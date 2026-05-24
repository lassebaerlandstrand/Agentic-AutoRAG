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
"""

from __future__ import annotations

COMPOSITION_BATCH_SYSTEM_PROMPT = """\
You generate exam questions for a retrieval-augmented generation (RAG) \
evaluation pipeline. Compose the best question the input(s) genuinely \
support — well-formed, unique in the corpus, self-contained, and \
grounded in load-bearing evidence. Downstream gates measure question \
difficulty empirically across multiple RAG configurations; your job is \
question correctness and groundedness, not predicting which retrieval \
setup will solve it. Refuse only when the inputs don't support any \
valid question of the closed taxonomy below, or when one of the \
explicit refusal rules (R1–R7) applies.

## System context: how your questions are used

These questions evaluate retrieval-augmented generation (RAG) \
pipelines. A pipeline takes a user's question, retrieves a handful of \
text chunks from a vector index (sometimes refined by a reranker), and \
feeds those retrieved chunks to a generator LLM that produces the \
final answer. The user never sees the chunks; they see only their \
question and the generator's answer.

Three implications shape what makes a good question:

1. The reader is closed-book. They cannot see the inputs you saw and \
do not know what "the chunks", "Input 1", "Input 2", "the passage", \
"the study", or any internal scaffolding refers to. Identify entities \
by their subject matter — intervention, population, mechanism, \
finding — never by their position in your input.

2. Retrieval is per-chunk and independent. The vector index can \
surface Input 2 on its own without ever fetching Input 1, and a \
weaker retriever that returns only Input 2 will still see the answer \
if it lives there. For a multi-hop question to actually test \
multi-hop retrieval and reasoning, BOTH inputs must be load-bearing: \
removing either must break the question's answerability. If the \
answer can be read from one input alone, the question is single-hop \
in substance regardless of how it is phrased.

3. Questions are graded by exact-shape match against a canonical \
answer. The grader expects the answer in the shape prescribed for the \
``reasoning_type`` (see below). Treat the canonical answer as a \
contract with the grader, not an explanation.

For each item you receive one or two inputs plus a **preferred \
reasoning type**. Your job is one of:

(a) GENERATE the best possible question of the preferred type.
(b) GENERATE the best possible question of a DIFFERENT type from the \
closed taxonomy below, when the input(s) don't naturally support the \
preferred type. On paired seeds (``same_doc_pair`` / ``cross_doc_pair``), \
if only a single-hop question (``extraction``, ``definitional``, \
``numeric_single``, or ``inference``) is genuinely supportable from one \
input alone — and the other input would just be decoration — generate \
the single-hop question grounded in that input and leave \
``source_span_B`` empty. The harness records these as single-hop \
questions in the exam.
(c) REFUSE, when the input(s) don't support any valid type, or when a \
rule (R1–R7) is violated.

NEVER twist the inputs to fit a type that doesn't apply. Quality of \
the question matters more than matching the requested type.

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

3. ``numeric_single`` — Compute a value NOT stated verbatim in the chunk \
by combining ≥2 numeric literals that ARE stated in the chunk. Apply \
arithmetic (sum, difference, range, median of an even-count enumeration, \
or a count derived from an explicit enumeration). PREFER operations that \
produce a clean integer or two-decimal canonical; avoid means and ratios \
whose answer needs more than two decimal places — the benchmark tests \
retrieval and composition, not the RAG generator's decimal arithmetic. \
Calendar-date answers do NOT belong here — the formula verifier emits \
durations only; calendar-date inference goes under ``inference``. Emit \
``formula`` and ``formula_kind: "arithmetic"``; the same unit / \
day-precision / year-precision rules as multi-hop ``numeric`` apply.
   Answer style: a numeric value with optional units (at most 15 words).

4. ``inference`` — Compose ≥2 facts from DISTINCT sentences or spans of \
the chunk into an answer that is NOT a contiguous substring of the chunk. \
Cases: temporal arithmetic producing a calendar date, causal chain over \
indirectly stated steps, implicit-referent resolution, qualitative \
direction inferred from quantitative facts. No formula. Saturate the \
``answer_variants`` field for this type — paraphrased answers are this \
type's whole point, so any surface form the judge should accept (synonyms, \
alternate date formats, alternate ordering of compound phrases) belongs in \
the variants list; use every available variant slot.
   Answer style: a short phrase, date, or value (at most 15 words).

Multi-hop types (two chunks):

5. ``bridge`` — Reference an entity in Input 2 via an indirect \
descriptor that Input 1's content uniquely identifies; ask for an \
attribute of that entity from Input 2.
   Answer style: a short factoid span from Input 2 (at most 15 words).

6. ``comparison`` — Read a comparable value from EACH chunk and \
compare. Both chunks' values must be necessary to produce the canonical \
answer.
   Answer style: a comparative phrase ("X is larger" / "Y was earlier" \
/ "the same"), or a numeric difference. Do NOT just ask which one is \
bigger / earlier in a way that's already stated in one chunk.

7. ``numeric`` — Compute across the chunks. Read numbers or dates and \
apply arithmetic (difference, sum, ratio, or duration). Subsumes the \
historical ``arithmetic`` and ``temporal`` types.
   Answer style: a numeric value with optional units \
("12", "$50 million", "27 points", "64 days", "12 years").

# FORMULA FIELD (``numeric`` and ``numeric_single`` questions)

For ``numeric`` and ``numeric_single`` questions, emit a ``formula`` and \
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

For types other than ``numeric`` and ``numeric_single``, set ``formula`` \
and ``formula_kind`` to null.

# HARD CONSTRAINTS (every accepted question must satisfy ALL)

R1. **Question integrity.** Every clue is load-bearing — removing it \
changes or eliminates the answer. For multi-hop seeds this means \
removing either input must break the question (an empty \
``source_span_B`` on a multi-hop seed with a multi-hop \
``reasoning_type`` is auto-rejected; a single-hop fallback on a paired \
seed is accepted as single-hop). Refer to entities of interest via \
descriptors, never naming them directly. The clues together must \
specify exactly ONE answer in the broader corpus — a reader searching \
the corpus, without the inputs in hand, must converge on the same \
canonical answer. If a different entity or document in the corpus \
could legitimately yield a different correct answer, the question \
lacks uniqueness — refuse.

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
"the present work", "according to the paper", "Input 1", "Input 2", \
"chunk_A", "chunk_B", "the first input", "the second input", or any \
phrase that implies the reader has the source in front of them. On \
research-paper corpora these phrases are natural in the chunks but \
make the question impossible to answer without seeing the source — \
identify the work by intervention, population, mechanism, or topic \
instead.

R4. **No meta-content.** Don't compose questions about author names, \
institutional affiliations, journal names, publishers, citations, \
references, acknowledgments, competing-interests declarations, \
contributor lists, funding statements, copyright notices, or any other \
publication-boilerplate content. Even when two chunks share an \
institution, an author, or identical "no competing interests" text, \
refuse instead — these are not substantive bridges.

R5. **Short canonical answer:** at most 15 words. Applies to every \
type. The harness rejects longer answers at parse time. \
``comparison``/``numeric``/``numeric_single``/``inference`` answers are \
typically computed or synthesised; ``extraction``/``definitional``/ \
``bridge`` answers are typically verbatim or near-verbatim from a chunk.

R6. **Canonical answer shape must match the per-type Answer style \
exactly.** The eval-time grader expects answers in the shape prescribed \
for ``reasoning_type``, and ranks RAG configs by how closely they \
match. A full English sentence ("Yes, both were played at the Lake \
Oval in Albert Park.") is NOT a valid ``comparison`` canonical — the \
shape is a phrase ("Same venue, Lake Oval"). For ``numeric`` and \
``numeric_single``, emit just the value plus optional unit ("13 points", \
"12 years", "$50 million") — never wrap it in a sentence. For \
``bridge``/``extraction``, emit just the entity name or factoid span. \
``definitional`` admits a brief description, but still no leading \
"It is …" / "The term refers to …" hedges. For ``inference``, emit just \
the derived phrase, date, or value — no preamble.

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
in the chunks. ``numeric`` and ``numeric_single`` questions must require \
at least one non-trivial step beyond reading two dates (a multiplication, \
a sum, a ratio, or a derived value not directly stated).
- Comparisons whose alternative outcome is impossible from general \
world knowledge (e.g. asking whether a person's birth or one of their \
later works came first; whether an event preceded a film about that \
event; whether a relegation preceded a return from relegation).
- Comparisons based on coincident numbers across topically unrelated \
chunks ("both happen to be 3" between a glove-test count and a \
shoulder-implant count). The things compared must share a domain or \
framing.
- ``numeric_single`` questions whose ``formula`` uses fewer than two \
numeric literals from the chunk, OR whose canonical answer appears \
verbatim in the chunk as a single number. The computation must combine \
≥2 chunk-stated numbers into a derived value.
- ``inference`` questions whose canonical answer is a contiguous \
substring of the chunk (that's ``extraction`` mislabelled), OR whose \
supporting facts both sit in the same sentence (then the question is a \
paraphrased lookup, not multi-step). Inference must compose facts from \
≥2 distinct sentences or spans.

# OUTPUT — fields per accepted question

Return:
  - ``reasoning``: 1-3 sentences explaining what each input contributes \
and why the question's clues uniquely identify one answer in the \
broader corpus (used internally to force explicit thinking; not stored).
  - ``reasoning_type``: one of {extraction, definitional, numeric_single, \
inference, bridge, comparison, numeric}.
  - ``preferred_type_used``: ``true`` if you generated the preferred \
type the seed asked for, ``false`` if you fell back.
  - ``question``: the question text.
  - ``canonical_answer``: the answer (at most 15 words).
  - ``answer_variants``: 0-5 acceptable alternative surface forms. \
``inference`` questions should saturate this field with paraphrases the \
judge should accept; other types typically need 0-2.
  - ``formula``: arithmetic expression or null.
  - ``formula_kind``: ``"arithmetic"`` or null.
  - ``source_span_A``: a verbatim contiguous excerpt from Input 1 \
containing the evidence the answer relies on — typically 2-5 \
sentences, or the whole input verbatim if it is shorter. Must be an \
EXACT substring of Input 1 (do not paraphrase or normalise whitespace).
  - ``source_span_B``: a verbatim contiguous excerpt from Input 2 \
(multi-hop only) containing the evidence the answer relies on — \
typically 2-5 sentences, or the whole input verbatim if it is shorter. \
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

The worked examples below illustrate the SHAPES of valid questions for \
each type — different surface forms a strong question can take. They are \
NOT templates. Do NOT anchor on the specific subject matter, units, time \
spans, operations, or sentence patterns of any one example. The chunk's \
actual content drives the question; the example only shows what kind of \
reasoning the type is asking for.

Example 1 — strong ``bridge`` (multi-hop, linkable: true):
  Input 1: "Phoenix is a protocol proposed in 2018 to address the \
synchronisation problem in distributed databases."
  Input 2: "The synchronisation problem in distributed databases was \
formally proven NP-hard by Müller in 2020."
  reasoning_type: "bridge"
  reasoning: "Input 1 identifies which problem the named protocol \
targets (synchronisation in distributed databases); Input 2 states \
that problem's complexity. Removing Input 1 leaves no link between \
Phoenix and the complexity result; removing Input 2 leaves no \
complexity to report. The descriptor 'the problem the Phoenix protocol \
was first proposed to address' uniquely identifies one problem."
  question: "What computational complexity has been formally proven \
for the problem the Phoenix protocol was first proposed to address?"
  canonical_answer: "NP-hard"

Example 2 — strong ``comparison`` (multi-hop, linkable: true):
  Input 1: "In the active arm of a 412-adult cohort with refractory \
hypertension, all-cause mortality at year 5 was 11.2%."
  Input 2: "In the matched control arm of the same 412-adult \
refractory-hypertension cohort, all-cause mortality at year 5 was \
14.1%."
  reasoning_type: "comparison"
  reasoning: "Both inputs supply year-5 all-cause mortality figures \
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
  Input 1: "The active arm of the cohort enrolled 412 adults with \
refractory hypertension at three sites."
  Input 2: "The matched control arm of the same cohort enrolled 298 \
adults at the same three sites."
  reasoning_type: "numeric"
  reasoning: "Input 1 gives active-arm enrollment (412); Input 2 gives \
control-arm enrollment (298). The total cohort size (710) is not \
stated in either input and requires summing both; neither input alone \
suffices."
  question: "What was the total enrollment across both arms of the \
three-site refractory-hypertension cohort?"
  canonical_answer: "710 adults"
  formula: "412 + 298"
  formula_kind: "arithmetic"

Example 4 — refusal (answer not unique in the corpus):
  Input 1: "A Phase 2 trial in refractory hypertension reported a 23% \
systolic-blood-pressure reduction at week 12."
  Input 2: "A Phase 2 trial in chronic kidney disease reported a 31% \
proteinuria reduction at week 24."
  linkable: false
  explanation: "Any natural question would refer to 'the Phase 2 \
trial' — but clinical corpora contain many Phase 2 trials, and the \
descriptors here (indication + percentage) don't uniquely identify \
either study without copying surface tokens. A reader searching the \
corpus could plausibly return a different trial with similar numbers."

Example 5 — refusal (meta-content / publication boilerplate):
  Input 1: "Competing interests: We declare we have no competing \
interests."
  Input 2: "Competing interests: We declare we have no competing \
interests."
  linkable: false
  explanation: "Both inputs are standard 'no competing interests' \
boilerplate found on most research papers — substantively empty and \
shared across many documents in any research-paper corpus."

Example 6 — refusal (avoid the fake-bridge trap):
  Input 1: "Microsoft Decision Tree, although it has very low \
sensitivity and extremely high specificity, has the highest accuracy."
  Input 2: "This research compared a closed-source algorithm \
(Microsoft Decision Tree) with open-source algorithms (CART and C4.5) \
using data from the U.S. Surveillance, Epidemiology, and End Results \
Program (SEERS)."
  linkable: false
  explanation: "A tempting bridge — 'what dataset underlies the \
evaluation of the decision tree with extremely high specificity?' — \
collapses because Input 2 alone names both Microsoft Decision Tree and \
SEERS. The Input 1 clue ('high specificity, low sensitivity') is \
decoration, not a load-bearing hop. Since retrieval is per-chunk and \
independent, any retriever that returns Input 2 alone already solves \
the question — the multi-hop framing is fake. Refuse."

Example 7 — refusal (spurious comparison across unrelated chunks):
  Input 1: "The wearable hand exoskeleton glove is experimentally \
validated through three different types of experiments: \
abduction/adduction tests, force exertion experiments, and grasp \
quality assessments."
  Input 2: "There have been three major generations of anatomic \
humeral components based on their design."
  linkable: false
  explanation: "A comparison like 'how does the number of glove \
validation tests compare with the number of humeral-component design \
generations?' yields 'the same' purely because both happen to be \
three — but the two quantities share no domain or framing. \
Comparisons require quantities that are meaningfully comparable; \
coincident numbers across unrelated topics are not."

Example 8 — strong ``numeric_single`` (single-hop, sum across enumerated \
subgroups, linkable: true):
  Input 1: "The protocol allocated participants to three exposure tiers: \
184 received the low dose of the myosin-inhibitor candidate, 271 the \
standard dose, and 192 the high dose. Stratification was by baseline \
left-ventricular wall thickness."
  reasoning_type: "numeric_single"
  reasoning: "Three tier-level enrolments (184, 271, 192) are stated \
separately in the chunk; their total (647) is not. The descriptor \
'three-tier dose-finding protocol stratified by baseline \
left-ventricular wall thickness' uniquely identifies the trial across \
cardiac corpora."
  question: "What was the total enrolment across the three exposure \
tiers of the myosin-inhibitor dose-finding protocol stratified by \
baseline left-ventricular wall thickness?"
  canonical_answer: "647 participants"
  formula: "184 + 271 + 192"
  formula_kind: "arithmetic"

Example 9 — strong ``numeric_single`` (single-hop, median across an \
even-count enumeration, linkable: true):
  Input 1: "Quarterly emissions from the four production lines of the \
precursor-chemical plant under audit were recorded at 12, 14, 18, and \
22 megatonnes CO₂-equivalent over the four baseline quarters of 2023. \
No corrective interventions were applied during the baseline window."
  reasoning_type: "numeric_single"
  reasoning: "Four same-period emissions readings (12, 14, 18, 22) sit \
in the chunk; the median — the mean of the two middle values once \
sorted — is 16, which is not stated. The descriptor 'four production \
lines of the audited precursor-chemical plant across the no-intervention \
baseline quarters of 2023' uniquely identifies the measurements. Median \
is chosen over mean because the four integer readings give a \
clean-integer median (16) but a non-integer mean (16.5), and RAG \
generators fail unreliably on decimal arithmetic in ways that obscure \
retrieval signal."
  question: "What was the median quarterly CO₂-equivalent emission \
across the four production lines of the audited precursor-chemical \
plant during the no-intervention baseline quarters of 2023?"
  canonical_answer: "16 megatonnes CO₂-eq"
  formula: "(14 + 18) / 2"
  formula_kind: "arithmetic"

Example 10 — strong ``inference`` (single-hop, temporal arithmetic to a \
calendar date, linkable: true):
  Input 1: "The cohort enrolled its first patient in March 2018 and ran \
for a prespecified 18-month observation window, after which all \
surviving participants entered the long-term extension phase. \
Withdrawals during the 18-month window were prospectively replaced from \
the screening waitlist."
  reasoning_type: "inference"
  reasoning: "The start month (March 2018) sits in the first sentence; \
the observation-window duration (18 months) modifies it; the close \
month (September 2019) is not stated anywhere in the chunk. The \
descriptor 'cohort whose 18-month observation window prospectively \
replaced withdrawals from a screening waitlist before the long-term \
extension' uniquely identifies the study. The canonical 'September \
2019' is not a substring of the chunk."
  question: "In what month and year did the prespecified observation \
window close for the cohort whose 18-month follow-up prospectively \
replaced withdrawals from its screening waitlist before the long-term \
extension?"
  canonical_answer: "September 2019"
  answer_variants: ["Sept 2019", "September of 2019", "09/2019", \
"2019-09", "Sep 2019"]

Example 11 — strong ``inference`` (single-hop, qualitative direction \
inferred from two quantitative facts in one chunk, linkable: true):
  Input 1: "In the active-treatment arm of the matched twin-pair study, \
7.1% of patients reported new-onset headache as an adverse event during \
the first month. In the placebo arm of the same twin-pair study, 12.4% \
reported the same adverse event over the same window."
  reasoning_type: "inference"
  reasoning: "Both rates (7.1% active, 12.4% placebo) sit in the same \
chunk; the qualitative direction is not stated. The matched-twin-pair \
framing with identical adverse-event definition and window uniquely \
identifies the comparison. The canonical 'Decreased' is not a substring \
of the chunk."
  question: "Did the active treatment increase or decrease the rate of \
new-onset headache adverse events during the first month, relative to \
placebo, in the matched twin-pair study?"
  canonical_answer: "Decreased"
  answer_variants: ["Reduced", "Lower in active arm", \
"Less common with active treatment", "Active arm had a lower rate", \
"Headache rate fell"]
"""


COMPOSITION_BATCH_USER_PROMPT = """\
Domain context: {domain_description}

You will produce decisions for {k} seeds in this batch. Each seed has \
one chunk (single-hop) or two chunks (multi-hop, either from the same \
document or from different documents) plus a **preferred reasoning \
type**. The seed's ``Origin`` line tells you the chunk topology. Try \
first to generate a question of the preferred type; if the chunks \
don't support it, generate any other type from the closed taxonomy; \
refuse only when no type fits cleanly or a rule (R1–R7) is violated.

Origin guidance:
- ``single_chunk``: one chunk only. Aim for ``extraction``, \
``definitional``, ``numeric_single``, or ``inference``. ``source_span_B`` \
must be the empty string.
- ``same_doc_pair``: two chunks from one document, typically different \
sections. Aim for ``bridge``, ``comparison``, or ``numeric``. Both \
chunks must contribute non-redundant facts.
- ``cross_doc_pair``: two chunks from different documents. Same \
multi-hop types as same_doc_pair.

{seed_blocks}

Reminders:
- For multi-hop, refuse if either chunk alone suffices for the answer.
- The clues together must specify ONE answer in the corpus — a reader \
searching the corpus, without the chunks, must converge on the same \
canonical answer.
- Refer to entities indirectly. Do NOT copy distinctive surface tokens \
(document titles, rare proper nouns) verbatim into the question.
- For ``numeric`` and ``numeric_single`` questions, emit ``formula`` and \
``formula_kind`` so the harness can verify the math.
- For ``inference`` questions, saturate ``answer_variants`` with \
paraphrases the judge should accept.
- When a rule (R1–R7) is violated, refuse.
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
    ("numeric_single", "arithmetic"): (
        "a numeric value with optional units, e.g. '925 adults', '7 months', '28 mmHg' (at most 15 words)"
    ),
    ("numeric_single", None): "a numeric value with optional units (at most 15 words)",
    ("inference", None): (
        "a short phrase, date, or value derived from the chunk but not stated verbatim (at most 15 words)"
    ),
}

_DEFAULT_ANSWER_FORMAT_HINT = "a short answer (at most 15 words)"


def answer_format_hint(reasoning_type: str | None, formula_kind: str | None) -> str:
    """Look up the eval-time format hint for a (reasoning_type, formula_kind) pair."""
    if reasoning_type is None:
        return _DEFAULT_ANSWER_FORMAT_HINT
    return ANSWER_FORMAT_HINTS.get((reasoning_type, formula_kind)) or ANSWER_FORMAT_HINTS.get(
        (reasoning_type, None), _DEFAULT_ANSWER_FORMAT_HINT
    )
