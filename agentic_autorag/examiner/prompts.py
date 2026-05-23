"""Prompt templates for the open-ended exam pipeline.

Composition prompts (used during exam generation):
  - COMPOSITION_BATCH_SYSTEM_PROMPT: shared system prompt with the 5-type
    reasoning taxonomy, universal rules, and worked examples.
  - COMPOSITION_BATCH_USER_PROMPT: per-batch user prompt for cross-doc
    multi-hop seeds.

Eval-time prompts (used by the system-under-test and the validator):
  - ORACLE_OPEN_ENDED_PROMPT: feeds all spans concatenated; used by the
    answerability gate during exam generation.
  - NAIVE_RAG_PROMPT: sent to the RAG pipeline at evaluation time.
  - JUDGE_PROMPT: LLM-as-judge for paraphrased answers.
"""

from __future__ import annotations

COMPOSITION_BATCH_SYSTEM_PROMPT = """\
You generate exam questions for a retrieval-augmented generation (RAG) \
evaluation pipeline. A good question DISTINGUISHES a weak RAG (vector- \
only retrieval, no reranking) from a strong one (hybrid retrieval, \
reranker, multi-hop reasoning). Trivial questions don't separate \
configurations and are worse than refusals — when chunks only support \
an easy question a competent keyword search would already solve, \
REFUSE rather than compose.

For each item you receive one or two chunks plus a **preferred \
reasoning type**. Your job is one of:

(a) GENERATE the best possible question of the preferred type.
(b) GENERATE the best possible question of a DIFFERENT type from the \
closed taxonomy below, when the chunk(s) don't naturally support the \
preferred type.
(c) REFUSE, when the chunk(s) don't support any type cleanly, or only \
support a trivial question.

NEVER twist the chunks to fit a type that doesn't apply. NEVER lower \
the bar to avoid refusing. Quality of the question matters more than \
matching the requested type.

# REASONING-TYPE TAXONOMY (closed; choose exactly one)

Single-hop types (one chunk):

1. ``extraction`` — A factoid lookup answerable from one verbatim span. \
The question targets a specific value, name, date, or short phrase the \
chunk states explicitly.
   Answer style: a short factoid (a name, value, or phrase; at most 15 \
words). Typically a verbatim span from the chunk.

2. ``definitional`` — Asks for a definition or short description the \
chunk states. The answer paraphrases or quotes the definitional content.
   Answer style: a brief definition or description (at most 15 words).

Multi-hop types (two chunks):

3. ``bridge`` — Reference an entity in chunk_B via an indirect \
descriptor that chunk_A's content uniquely identifies; ask for an \
attribute of that entity from chunk_B.
   Answer style: a short factoid span from chunk_B (at most 15 words).

4. ``comparison`` — Read a comparable value from EACH chunk and \
compare. Both chunks' values must be necessary to produce the canonical \
answer.
   Answer style: a comparative phrase ("X is larger" / "Y was earlier" \
/ "the same"), or a numeric difference. Do NOT just ask which one is \
bigger / earlier in a way that's already stated in one chunk.

5. ``numeric`` — Compute across the chunks. Read numbers or dates and \
apply arithmetic (difference, sum, ratio, or duration). Subsumes the \
historical ``arithmetic`` and ``temporal`` types.
   Answer style: a numeric value with optional units \
("12", "$50 million", "27 points", "64 days", "12 years").

# FORMULA FIELD (numeric questions only)

For ``numeric`` questions, emit a ``formula`` and \
``formula_kind: "arithmetic"`` that the harness evaluates to verify the \
canonical answer. ``formula`` is a Python arithmetic expression over \
numeric literals only. Examples: ``2012 - 1948``, ``(300 + 250) / 2``, \
``11 * 27``, ``21 + 29``. No variable names, no function calls, no \
attribute access.

For temporal differences, the formula and answer must use the SAME \
unit, and you may only emit day-precision when the gap is small enough \
for a reader to verify mentally:

- Day-precision (≤ ~30 days): chunks must state day-precision dates \
AND the difference must be at most one month. Encode as integer day \
arithmetic against day-of-month numbers from the chunks (e.g. ``19 - \
5`` → ``"14 days"``). Day-precision arithmetic is not supported beyond \
~30 days.
- Year-precision (≥ ~1 year): integer year difference (``2011 - 2008``) \
and answer in years (``"3 years"``).
- Anything in between (~1 to ~12 months): if chunks state year and \
month, use month-precision arithmetic (``11 - 4`` → ``"7 months"``). \
Otherwise prefer a ``comparison`` question instead.

Do NOT manufacture day-precision by multiplying year differences by \
365, or month differences by 30 — the verifier catches the unit \
mismatch and rejects. The output unit must match what the formula \
computes.

For non-numeric types, set ``formula`` and ``formula_kind`` to null.

# HARD CONSTRAINTS (every accepted question must satisfy ALL)

R1. **Question integrity.** Every clue is load-bearing — removing it \
changes or eliminates the answer. For multi-hop seeds this means \
removing either chunk must break the question (an empty \
``source_span_B`` on a multi-hop seed is auto-rejected). Refer to \
entities of interest via descriptors, never naming them directly. The \
clues together must specify exactly ONE answer in the broader corpus — \
a reader searching the corpus, without the chunks in hand, must \
converge on the same canonical answer. If a different entity or \
document in the corpus could legitimately yield a different correct \
answer, the question lacks uniqueness — refuse.

Uniqueness contrast — apply this check before composing:

  Ambiguous clue (BAD):
    "the Phase 2 trial reporting a 23% reduction at week 12"
    — Many trials across compounds and indications report similar
      percentages; the descriptor matches multiple corpus documents.

  Unique clue (GOOD):
    "the Phase 2 trial of selumetinib in pediatric NF1-related
    plexiform neurofibromas"
    — Compound + population uniquely identifies one trial.

If you cannot find a uniquely-identifying clue without copying surface \
tokens, refuse the question.

R2. **No surface-token leakage from the chunks.** Do NOT include any \
of the document titles or any rare proper nouns that appear verbatim \
in the chunks. Refer to entities, events, dates, and titles \
indirectly — through their role, relationship, or definitional \
descriptor — even when this makes the question longer or more \
elliptical. The reader will not have any chunk in front of them; the \
question must work as a closed-book prompt that a search engine cannot \
trivially match by keyword overlap.

R3. **Self-contained.** No "the document", "the passage", "the above \
text", "the study", "the trial", "the experiment", "the analysis", \
"the present work", "according to the paper", or any phrase that \
implies the reader has the source in front of them. On research-paper \
corpora these phrases are natural in the chunks but make the question \
impossible to answer without seeing the chunk — identify the work by \
intervention, population, mechanism, or topic instead.

R4. **No meta-content.** Don't compose questions about author names, \
institutional affiliations, journal names, publishers, citations, \
references, acknowledgments, competing-interests declarations, \
contributor lists, funding statements, copyright notices, or any other \
publication-boilerplate content. Even when two chunks share an \
institution, an author, or identical "no competing interests" text, \
refuse instead — these are not substantive bridges.

R5. **Short canonical answer:** at most 15 words. Applies to every \
type. The harness rejects longer answers at parse time. \
``comparison``/``numeric`` answers are typically computed or \
synthesised; ``extraction``/``definitional``/``bridge`` answers are \
typically verbatim or near-verbatim from a chunk.

R6. **Canonical answer shape must match the per-type Answer style \
exactly.** The eval-time grader expects answers in the shape prescribed \
for ``reasoning_type``, and ranks RAG configs by how closely they \
match. A full English sentence ("Yes, both were played at the Lake \
Oval in Albert Park.") is NOT a valid ``comparison`` canonical — the \
shape is a phrase ("Same venue, Lake Oval"). For ``numeric``, emit \
just the value plus optional unit ("13 points", "12 years", "$50 \
million") — never wrap it in a sentence. For ``bridge``/ \
``extraction``, emit just the entity name or factoid span. \
``definitional`` admits a brief description, but still no leading \
"It is …" / "The term refers to …" hedges.

R7. **Anti-trivia.** Refuse rather than compose:
- Self-answering questions where the values needed for the answer \
appear in the question text itself. If a date, count, or quantity used \
as a descriptor in your question is ALSO the value the reader must \
extract, compare, or compute with, the question is trivially \
answerable without the chunks — refuse. Identify entities by role, \
distinguishing event, or relationship, NOT by the specific numeric \
value the question asks about. Examples to REFUSE: "which is earlier, \
the show that aired in January 2005 or the one in November 2010?" \
(dates supplied by the question); "by how much does the team's 21 \
consecutive seasons exceed the rival's 12 seasons?" (counts supplied \
by the question).
- Bare year/month subtraction where both dates are explicitly stated \
in the chunks. Numerics must require at least one non-trivial step \
beyond reading two dates (a multiplication, a sum, a ratio, or a \
derived value not directly stated).
- Comparisons whose alternative outcome is impossible from general \
world knowledge (e.g. asking whether a person's birth or one of their \
later works came first; whether an event preceded a film about that \
event; whether a relegation preceded a return from relegation).
- Comparisons based on coincident numbers across topically unrelated \
chunks ("both happen to be 3" between a glove-test count and a \
shoulder-implant count). The things compared must share a domain or \
framing.

# OUTPUT — fields per accepted question

Return:
  - ``reasoning``: 1-3 sentences explaining what each chunk contributes \
and why the question's clues uniquely identify one answer in the \
broader corpus (used internally to force explicit thinking; not stored).
  - ``reasoning_type``: one of {extraction, definitional, bridge, \
comparison, numeric}.
  - ``preferred_type_used``: ``true`` if you generated the preferred \
type the seed asked for, ``false`` if you fell back.
  - ``question``: the question text.
  - ``canonical_answer``: the answer (at most 15 words).
  - ``answer_variants``: 0-3 acceptable alternative surface forms.
  - ``formula``: arithmetic expression or null.
  - ``formula_kind``: ``"arithmetic"`` or null.
  - ``source_span_A``: a verbatim contiguous excerpt from chunk_A \
containing the evidence the answer relies on — typically 2-5 \
sentences, or the whole chunk verbatim if it is shorter. Must be an \
EXACT substring of chunk_A (do not paraphrase or normalise whitespace).
  - ``source_span_B``: a verbatim contiguous excerpt from chunk_B \
(multi-hop only) containing the evidence the answer relies on — \
typically 2-5 sentences, or the whole chunk verbatim if it is shorter. \
For single-hop, set this to the empty string.

When you REFUSE, return only ``explanation`` — one plain English \
sentence explaining why no type works.

# OUTPUT FORMAT

Return a JSON ARRAY of exactly K objects, in seed order. Each object \
is one of:

  // refusal
  {"seed_id": <int>, "linkable": false, "explanation": "<one sentence>"}

  // accepted
  {"seed_id": <int>, "linkable": true,
   "reasoning": "...",
   "reasoning_type": "...",
   "preferred_type_used": true|false,
   "question": "...",
   "canonical_answer": "...",
   "answer_variants": ["..."],
   "formula": null | "...",
   "formula_kind": null | "arithmetic",
   "source_span_A": "...",
   "source_span_B": "..."}

Return ONLY the JSON array. No commentary, no markdown fences.

# WORKED EXAMPLES

Example 1 — strong ``bridge`` (multi-hop, linkable: true):
  chunk_A: "Phoenix is a protocol proposed in 2018 to address the \
synchronisation problem in distributed databases."
  chunk_B: "The synchronisation problem in distributed databases was \
formally proven NP-hard by Müller in 2020."
  reasoning_type: "bridge"
  reasoning: "chunk_A identifies which problem the named protocol \
targets (synchronisation in distributed databases); chunk_B states \
that problem's complexity. Removing chunk_A leaves no link between \
Phoenix and the complexity result; removing chunk_B leaves no \
complexity to report. The descriptor 'the problem the Phoenix protocol \
was first proposed to address' uniquely identifies one problem."
  question: "What computational complexity has been formally proven \
for the problem the Phoenix protocol was first proposed to address?"
  canonical_answer: "NP-hard"

Example 2 — strong ``comparison`` (multi-hop, linkable: true):
  chunk_A: "In the active arm of a 412-adult cohort with refractory \
hypertension, all-cause mortality at year 5 was 11.2%."
  chunk_B: "In the matched control arm of the same 412-adult \
refractory-hypertension cohort, all-cause mortality at year 5 was \
14.1%."
  reasoning_type: "comparison"
  reasoning: "Both chunks supply year-5 all-cause mortality figures \
(11.2% active, 14.1% control) for distinguishable arms of the same \
cohort; the comparison requires reading both values. The descriptor \
'active arm of the 412-adult refractory-hypertension cohort' \
identifies a unique trial arm."
  question: "How did year-5 all-cause mortality in the active arm of \
the 412-adult refractory-hypertension cohort compare with the matched \
control arm?"
  canonical_answer: "2.9 percentage points lower"
  answer_variants: ["lower by 2.9 percentage points"]

Example 3 — strong ``numeric`` (multi-hop, NON-date arithmetic):
  chunk_A: "The active arm of the cohort enrolled 412 adults with \
refractory hypertension at three sites."
  chunk_B: "The matched control arm of the same cohort enrolled 298 \
adults at the same three sites."
  reasoning_type: "numeric"
  reasoning: "chunk_A gives active-arm enrollment (412); chunk_B gives \
control-arm enrollment (298). The total cohort size (710) is not \
stated in either chunk and requires summing both; neither chunk alone \
suffices."
  question: "What was the total enrollment across both arms of the \
three-site refractory-hypertension cohort?"
  canonical_answer: "710 adults"
  formula: "412 + 298"
  formula_kind: "arithmetic"

Example 4 — refusal (answer not unique in the corpus):
  chunk_A: "A Phase 2 trial in refractory hypertension reported a 23% \
systolic-blood-pressure reduction at week 12."
  chunk_B: "A Phase 2 trial in chronic kidney disease reported a 31% \
proteinuria reduction at week 24."
  linkable: false
  explanation: "Any natural question would refer to 'the Phase 2 \
trial' — but clinical corpora contain many Phase 2 trials, and the \
descriptors here (indication + percentage) don't uniquely identify \
either study without copying surface tokens. A reader searching the \
corpus could plausibly return a different trial with similar numbers."

Example 5 — refusal (meta-content / publication boilerplate):
  chunk_A: "Competing interests: We declare we have no competing \
interests."
  chunk_B: "Competing interests: We declare we have no competing \
interests."
  linkable: false
  explanation: "Both chunks are standard 'no competing interests' \
boilerplate found on most research papers — substantively empty and \
shared across many documents in any research-paper corpus."

Example 6 — refusal (avoid the fake-bridge trap):
  chunk_A: "Microsoft Decision Tree, although it has very low \
sensitivity and extremely high specificity, has the highest accuracy."
  chunk_B: "This research compared a closed-source algorithm \
(Microsoft Decision Tree) with open-source algorithms (CART and C4.5) \
using data from the U.S. Surveillance, Epidemiology, and End Results \
Program (SEERS)."
  linkable: false
  explanation: "A tempting bridge — 'what dataset underlies the \
evaluation of the decision tree with extremely high specificity?' — \
collapses because chunk_B alone names both Microsoft Decision Tree and \
SEERS. The chunk_A clue ('high specificity, low sensitivity') is \
decoration, not a load-bearing hop. Refuse rather than compose a \
multi-hop framing whose dependency is fake."

Example 7 — refusal (spurious comparison across unrelated chunks):
  chunk_A: "The wearable hand exoskeleton glove is experimentally \
validated through three different types of experiments: \
abduction/adduction tests, force exertion experiments, and grasp \
quality assessments."
  chunk_B: "There have been three major generations of anatomic \
humeral components based on their design."
  linkable: false
  explanation: "A comparison like 'how does the number of glove \
validation tests compare with the number of humeral-component design \
generations?' yields 'the same' purely because both happen to be \
three — but the two quantities share no domain or framing. \
Comparisons require quantities that are meaningfully comparable; \
coincident numbers across unrelated topics are not."
"""


COMPOSITION_BATCH_USER_PROMPT = """\
Domain context: {domain_description}

You will produce decisions for {k} seeds in this batch. Each seed has \
one chunk (single-hop) or two chunks (multi-hop, either from the same \
document or from different documents) plus a **preferred reasoning \
type**. The seed's ``Origin`` line tells you the chunk topology. Try \
first to generate a question of the preferred type; if the chunks \
don't support it, generate any other type; if no type fits or only a \
trivial question would result, refuse.

Origin guidance:
- ``single_chunk``: one chunk only. Aim for ``extraction`` or \
``definitional``. ``source_span_B`` must be the empty string.
- ``same_doc_pair``: two chunks from one document, typically different \
sections. Aim for ``bridge``, ``comparison``, or ``numeric``. Both \
chunks must contribute non-redundant facts.
- ``cross_doc_pair``: two chunks from different documents. Same \
multi-hop types as same_doc_pair.

{seed_blocks}

Reminders:
- The goal is HARD questions that distinguish strong RAG configs from \
weak ones. Trivial questions are worse than refusals.
- For multi-hop, refuse if either chunk alone suffices for the answer.
- The clues together must specify ONE answer in the corpus — a reader \
searching the corpus, without the chunks, must converge on the same \
canonical answer.
- Refer to entities indirectly. Do NOT copy distinctive surface tokens \
(document titles, rare proper nouns) verbatim into the question.
- For ``numeric`` questions, emit ``formula`` and ``formula_kind`` so \
the harness can verify the math.
- When in doubt, refuse — do not lower the bar.
"""


ORACLE_OPEN_ENDED_PROMPT = """\
Answer the following question using ONLY the context below. The \
context is known to contain the information needed to determine the \
correct answer.

Expected answer format: {answer_format_hint}

Output the answer and nothing else — no explanation, no quotes, no \
punctuation. Keep the answer to at most 15 words.

Context:
{context}

Question: {question}

Answer:"""


NAIVE_RAG_PROMPT = """\
Answer the following question. Use only the provided context if any \
was retrieved; otherwise answer to the best of your ability.

Expected answer format: {answer_format_hint}

Output the answer and nothing else — no explanation, no quotes, no \
punctuation. Keep the answer to at most 15 words.

Context:
{context}

Question: {question}

Answer:"""


JUDGE_PROMPT = """\
You are grading a question-answering system.

Question: {question}
Reference answer(s): {gold}
System answer: {pred}

Pick exactly one verdict and respond with that single token, nothing \
else:

  YES        — the system answer conveys the same factual information \
as any reference answer (paraphrasing is OK).
  NO         — the system answer asserts something different from the \
reference answer.
  NO_ANSWER  — the system did not attempt an answer (it said it cannot \
answer, that the context is insufficient, that it doesn't know, or \
its output is otherwise a refusal rather than an attempted factual \
claim)."""


# Per-(reasoning_type, formula_kind) hint embedded into eval-time prompts so
# the system-under-test knows what shape of answer to produce. Removes a
# known false-negative mode where the model emits "it was larger" when the
# canonical is "13 points larger".
ANSWER_FORMAT_HINTS: dict[tuple[str, str | None], str] = {
    ("extraction", None): "a short factoid: a name, value, or phrase (at most 15 words)",
    ("definitional", None): "a brief definition or description (at most 15 words)",
    ("bridge", None): "a short factoid identifying an entity or attribute (at most 15 words)",
    ("comparison", None): (
        "a comparative phrase such as 'X is larger', 'Y was earlier', "
        "'the same', or a numeric difference with units (at most 15 words)"
    ),
    ("numeric", "arithmetic"): (
        "a numeric value with optional units, e.g. '13', '$50 million', '27 points', '12 years' (at most 15 words)"
    ),
    ("numeric", None): "a numeric value with optional units (at most 15 words)",
}

_DEFAULT_ANSWER_FORMAT_HINT = "a short answer (at most 15 words)"


def answer_format_hint(reasoning_type: str | None, formula_kind: str | None) -> str:
    """Look up the eval-time format hint for a (reasoning_type, formula_kind) pair."""
    if reasoning_type is None:
        return _DEFAULT_ANSWER_FORMAT_HINT
    return ANSWER_FORMAT_HINTS.get((reasoning_type, formula_kind)) or ANSWER_FORMAT_HINTS.get(
        (reasoning_type, None), _DEFAULT_ANSWER_FORMAT_HINT
    )
