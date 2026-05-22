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
evaluation pipeline. We need questions that test what a real RAG \
system has to do: locate the right evidence in a corpus and synthesise \
an answer from it. Some questions need one chunk; some need two chunks \
reasoning together.

For each item you receive one or two chunks plus a **preferred reasoning \
type**. Your job is one of:

(a) GENERATE the best possible question of the preferred type.
(b) GENERATE the best possible question of a DIFFERENT type from the \
closed taxonomy below, when the chunk(s) don't naturally support the \
preferred type.
(c) REFUSE, when the chunk(s) don't support any type cleanly.

NEVER twist the chunks to fit a type that doesn't apply. The preferred \
type is a hint, not a constraint — quality of the question matters more \
than matching the requested type.

# REASONING-TYPE TAXONOMY (closed; choose exactly one)

Single-hop types (one chunk):

1. ``extraction`` — A factoid lookup answerable from one verbatim span. \
The question targets a specific value, name, date, or short phrase that \
the chunk states explicitly.
   Answer style: a short factoid (a name, value, or phrase; at most 15 \
words). Typically a verbatim span from the chunk.

2. ``definitional`` — Asks for a definition or short description that \
the chunk states. The answer paraphrases or quotes the definitional \
content.
   Answer style: a brief definition or description (at most 15 words).

Multi-hop types (two or more chunks):

3. ``bridge`` — Reference an entity in chunk_B via an indirect \
descriptor that chunk_A's content uniquely identifies; ask for an \
attribute of that entity from chunk_B.
   Answer style: a short factoid span from chunk_B (at most 15 words).

4. ``comparison`` — Read a comparable value from EACH chunk and \
compare. The question forces the answerer to read both values and \
synthesize a comparative answer.
   Answer style: a comparative phrase ("X is larger" / "Y was earlier" \
/ "the same"), or a numeric difference. Do NOT just ask which one is \
bigger / earlier in a way that's already stated in one chunk.

5. ``numeric`` — Compute across the chunks. The answerer reads numbers \
or dates and applies a simple arithmetic operation (difference, sum, \
ratio, or duration between two dates) to produce the answer. Subsumes \
the historical ``arithmetic`` and ``temporal`` types.
   Answer style: a numeric value with optional units \
("12", "$50 million", "27 points", "64 days", "12 years").

# FORMULA FIELD (numeric questions only)

For ``numeric`` questions, emit a ``formula`` and \
``formula_kind: "arithmetic"`` that the harness can evaluate to \
verify the canonical answer. ``formula`` is a Python arithmetic \
expression over numeric literals only. Examples: ``2012 - 1948``, \
``(300 + 250) / 2``, ``11 * 27``, ``21 + 29``. No variable names, no \
function calls, no attribute access.

For temporal differences, the formula and answer must use the SAME \
unit, and you may only emit day-precision when the gap is small \
enough for a reader to verify mentally:

- Day-precision (≤ ~30 days): the chunks must state day-precision \
dates AND the difference must be at most one month. Encode as \
integer day arithmetic against day-of-month numbers from the chunks \
(e.g. ``19 - 5`` → ``"14 days"``). Day-precision arithmetic is not \
supported beyond ~30 days.
- Year-precision (≥ ~1 year): write the integer year difference \
(``2011 - 2008``) and answer in years (``"3 years"``).
- Anything in between (~1 to ~12 months): if the chunks state both \
the year and month, use month-precision arithmetic (e.g. \
``11 - 4`` → ``"7 months"``). Otherwise prefer a ``comparison`` \
question instead.

Critically, do NOT manufacture day-precision by multiplying year \
differences by 365, or month differences by 30 — the verifier will \
catch the answer-unit mismatch and reject. The output unit must \
match what the formula computes.

The harness will compute the formula and reject the question if its \
result disagrees with ``canonical_answer``. Get the math right.

For non-numeric types, set ``formula`` and ``formula_kind`` to null.

# HARD CONSTRAINTS (every accepted question must satisfy ALL)

R1. **All chunks necessary**: For multi-hop types, every chunk must \
contribute a non-redundant fact that the answer requires. The question \
cannot be answerable from a strict subset of the chunks. If you cannot \
satisfy R1 on a multi-hop seed, REFUSE rather than emit a question \
that uses only one chunk (an empty ``source_span_B`` on a multi-hop \
seed will be auto-rejected).

R2. **Indirect reference**: Refer to entities of interest via \
descriptors, never by naming them directly. Even when the entity has a \
famous name and that name appears in a chunk, refer to it indirectly \
(e.g. "the team that won the 1907 final" instead of "Carlton").

R3. **No surface-token leakage from the chunks**: Do NOT include any of \
the document titles or any rare proper nouns that appear verbatim in \
the chunks. Refer to entities, events, dates, and titles indirectly — \
through their role, relationship, or definitional descriptor — even \
when this makes the question longer or more elliptical. The reader of \
this question will not have any chunk in front of them; the question \
must work as a closed-book prompt that a search engine cannot \
trivially match by keyword overlap.

R4. **Descriptor uniqueness across the broader knowledge base**: Each \
clue in your question must identify exactly one entity in the wider \
knowledge base — not just within the chunks you have been given. \
Before finalising, ask: could a reader plausibly substitute a \
different entity that also fits this clue?

  Ambiguous clue (BAD):
    "the cup competition that was restricted to non-finalists"
    — Many cup editions across years fit this category.

  Unique clue (GOOD):
    "the cup competition won by Geelong defeating North Melbourne in \
the final"
    — Specific to one edition (1961), without naming the year.

  Prefer narrow event-specific descriptors (winners, scores, specific \
venues, distinguishing achievements) over category descriptors. If \
you cannot find a uniquely-identifying clue without copying surface \
tokens, refuse the question.

R5. **Self-contained**: No "the document", "the passage", "the above \
text", "the study", "according to the paper", or any phrase that \
implies the reader has the source in front of them.

R6. **No bibliographic bridges**: Author names, institutional \
affiliations, journal names, publishers, citations, references, and \
acknowledgments are NOT valid bridges. Even when chunks share an \
institution or author, refuse instead.

R7. **Short canonical answer**: at most 15 words. Applies to every \
type — no exceptions. The harness rejects longer answers at parse \
time. ``comparison``/``numeric`` answers are typically computed or \
synthesised; ``extraction``/``definitional``/``bridge`` answers are \
typically verbatim or near-verbatim from a chunk.

R8. **Canonical answer shape must match the per-type Answer style \
exactly**. The eval-time grader expects answers in the shape \
prescribed for ``reasoning_type``, and ranks RAG configs by how \
closely they match. A full English sentence ("Yes, both were played \
at the Lake Oval in Albert Park.") is NOT a valid ``comparison`` \
canonical — the shape is a phrase ("Same venue, Lake Oval"). For \
``numeric``, emit just the value plus optional unit \
("13 points", "12 years", "$50 million") — never wrap it in a \
sentence. For ``bridge``/``extraction``, emit just the entity name \
or factoid span. ``definitional`` admits a brief description, but \
still no leading "It is …" / "The term refers to …" hedges.

# OUTPUT — fields per accepted question

Return:
  - ``reasoning``: 1-3 sentences explaining what each chunk contributes \
and why this is a good question (used internally to force explicit \
thinking; not stored).
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
exact substring of chunk_A.
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

# WORKED EXAMPLES (one per type)

Example 1 — preferred ``extraction`` (single-hop, linkable: true):
  chunk_A: "[Chunk: a clinical trial of MK-921 reported a maximum \
tolerated dose of 240 mg/kg in adult patients across cohorts B and C.]"
  reasoning_type: "extraction"
  question: "What was the maximum tolerated dose reported for the \
investigational compound studied in adult cohorts B and C of the \
referenced trial?"
  canonical_answer: "240 mg/kg"
  formula: null
  formula_kind: null

Example 2 — preferred ``definitional`` (single-hop, linkable: true):
  chunk_A: "[Chunk: an inclusion criterion for the trial required \
participants to have an estimated glomerular filtration rate above 60 \
mL/min/1.73m² and no prior immunosuppressive therapy in the preceding \
12 months.]"
  reasoning_type: "definitional"
  question: "How is participant eligibility characterised by the renal \
function and prior-therapy criteria of the referenced trial?"
  canonical_answer: "eGFR > 60 mL/min/1.73m² and no immunosuppressive \
therapy in 12 months"

Example 3 — preferred ``bridge`` (multi-hop, linkable: true):
  chunk_A: "[Chunk: Phoenix is a protocol proposed in 2018 to address \
the synchronisation problem in distributed databases.]"
  chunk_B: "[Chunk: the synchronisation problem in distributed \
databases was formally proven NP-hard by Müller in 2020.]"
  reasoning_type: "bridge"
  question: "What computational complexity has been formally proven \
for the problem the Phoenix protocol was first proposed to address?"
  canonical_answer: "NP-hard"

Example 4 — preferred ``comparison`` (multi-hop, linkable: true):
  chunk_A: "The 1907 VFL Grand Final was contested between Carlton and \
South Melbourne. Carlton won by 5 points."
  chunk_B: "The 1909 VFL Grand Final was contested between the same \
two clubs. South Melbourne won by 2 points."
  reasoning_type: "comparison"
  question: "How did the winning margin in the 1909 VFL Grand Final \
compare to the margin in the same fixture two years earlier?"
  canonical_answer: "3 points smaller"
  answer_variants: ["3 fewer points", "smaller by 3"]

Example 5 — preferred ``numeric`` arithmetic (multi-hop, linkable: \
true):
  chunk_A: "[Chunk: a regional NFL all-star game scored 24 points by \
the winning conference in its 1971-season edition.]"
  chunk_B: "[Chunk: the corresponding game two seasons later — the \
1973-season edition — was won by the same conference scoring 13 \
points.]"
  reasoning_type: "numeric"
  question: "What is the difference in the total number of points \
scored by the winning conference in the professional all-star game \
two seasons apart in the early 1970s?"
  canonical_answer: "11 points"
  answer_variants: ["11"]
  formula: "24 - 13"
  formula_kind: "arithmetic"

Example 6 — refusal (descriptor not unique):
  chunk_A: "[Chunk: the 1957 VFL Night Premiership Cup was won by \
South Melbourne by 51 points.]"
  chunk_B: "[Chunk: the 1961 VFL Night Premiership Cup, contested by \
the eight teams that did not make the finals, was won by Geelong by \
12 points.]"
  linkable: false
  explanation: "Any natural question would have to refer to chunk_B \
as 'the cup restricted to non-finalists' — but the corpus contains \
many non-finalists editions, so the descriptor is not unique. No \
uniquely-identifying clue is available without naming the year \
directly."

Example 7 — refusal (bibliographic-only overlap):
  chunk_A: "[Chunk: Department A, Institute X. Topic T1 in domain D1, \
outcome O1.]"
  chunk_B: "[Chunk: Department B, Institute X. Topic T2 in domain D2, \
outcome O2.]"
  linkable: false
  explanation: "The only overlap between these chunks is the shared \
institutional affiliation; the actual subject matter is unrelated, so \
no substantive 2-hop question is possible."
"""


COMPOSITION_BATCH_USER_PROMPT = """\
Domain context: {domain_description}

You will produce decisions for {k} seeds in this batch. Each seed has \
one chunk (single-hop) or two chunks (multi-hop, either from the same \
document or from different documents) plus a **preferred reasoning \
type**. The seed's ``Origin`` line tells you the chunk topology. Try \
first to generate a question of the preferred type; if the chunks \
don't support it, generate a question of any other type; if no type \
fits, refuse.

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
- For multi-hop types, BOTH chunks must be necessary; neither alone \
should suffice.
- For single-hop, the question must be answerable from the single \
chunk but NOT from no context (no surface-token leakage from chunk \
into question).
- Refer to entities indirectly. Do NOT copy distinctive surface \
tokens (document titles, rare proper nouns) verbatim into the question.
- For ``numeric`` questions, emit ``formula`` and ``formula_kind`` so \
the harness can verify the math.
- Do not bridge on bibliographic content (authors, citations, \
journals, acknowledgments).
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
