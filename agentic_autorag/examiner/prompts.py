"""Prompt templates for the open-ended typed-2-hop exam pipeline.

Three prompts:
  - COMPOSITION_BATCH_PROMPT: takes K seeds (chunk_A, chunk_B, preferred_type)
    and asks the LLM to either generate the best 2-hop question of the
    preferred type — falling back to any other type from the closed
    taxonomy if the chunk pair doesn't naturally support it — or refuse
    with a free-text explanation.
  - SINGLE_HOP_SUFFICIENCY_PROBE: feeds only chunk_A's span as context and
    asks the LLM to answer. Used to verify the question is non-decomposable.
  - ORACLE_OPEN_ENDED_PROMPT: feeds both spans concatenated. Used as
    answerability gate and (in benchmark_eval form) the eval-time prompt.
"""

from __future__ import annotations

COMPOSITION_BATCH_SYSTEM_PROMPT = """\
You generate exam questions for a retrieval-augmented generation (RAG) \
evaluation framework. The framework needs questions that require \
**multi-hop reasoning across two documents**: a single keyword lookup \
cannot answer them, and one chunk alone cannot answer them.

For each item you receive two chunks from different documents and a \
**preferred question type**. Your job for each item is one of:

(a) GENERATE the best possible 2-hop question of the preferred type.
(b) GENERATE the best possible 2-hop question of a DIFFERENT type from \
the closed taxonomy below, when the chunk pair doesn't naturally support \
the preferred type.
(c) REFUSE, when the chunk pair doesn't support any type cleanly.

NEVER twist the chunks to fit a type that doesn't apply. The preferred \
type is a hint, not a constraint — quality of the question matters more \
than matching the requested type.

# QUESTION-TYPE TAXONOMY (closed; choose exactly one)

1. ``comparison`` — Read a comparable value from EACH chunk and compare. \
The question forces the answerer to read both values and synthesize a \
comparative answer.
   Answer style: comparative ("X" / "Y" / "the same"), or a numeric \
difference. Do NOT just ask which one is bigger / earlier in a way that's \
already stated in one chunk.

2. ``multi_constraint`` — Conjoin THREE OR MORE properties drawn from \
both chunks that uniquely identify a target. Phrase the question without \
naming the target. Each constraint should individually have several \
plausible candidates in a generic corpus, so the conjunction is what \
disambiguates.
   Answer style: a short proper-noun span (≤ 10 tokens).

3. ``exclusion`` — "Of the items listed in chunk_A, which is NOT \
[property described in chunk_B]?" or vice versa. The chunk that defines \
the exclusion is silent on the excluded item — the answerer must infer \
absence-of-evidence.
   Answer style: a short proper-noun span (≤ 10 tokens).

4. ``arithmetic`` — Compute across the two chunks. The answerer reads a \
number from each and applies a simple arithmetic operation (difference, \
sum, ratio, year-gap) to produce the answer.
   Answer style: a numeric value (≤ 10 tokens), expressed cleanly \
("12", "12 years", "$50 million").

5. ``bridge_chain`` — Reference an entity in chunk_B via an indirect \
descriptor that chunk_A's content uniquely identifies; ask for an \
attribute of that entity from chunk_B.
   Answer style: a short factoid span from chunk_B (≤ 10 tokens).

# HARD CONSTRAINTS (every accepted question must satisfy ALL)

R1. **Both-chunks-necessary**: Cannot be answered from chunk_A alone, \
and cannot be answered from chunk_B alone. Both contribute a \
non-redundant fact.

R2. **Indirect reference**: Refer to the entity-of-interest via a \
descriptor that chunk_A's content uniquely identifies, never by naming \
it directly. Even when the entity has a famous name and that name \
appears in chunk_A, refer to it indirectly (e.g. "the team that won the \
1907 final" instead of "Carlton").

R3. **No surface-token leakage from the chunks**: Do NOT include any of \
the document titles or any rare proper nouns that appear verbatim in \
either chunk. Refer to entities, events, dates, and titles indirectly — \
through their role, relationship, or definitional descriptor — even when \
this makes the question longer or more elliptical. The reader of this \
question will not have either chunk in front of them; the question must \
work as a closed-book prompt that a search engine cannot trivially \
match by keyword overlap.

R4. **Self-contained**: No "the document", "the passage", "the above \
text", "the study", "according to the paper", or any phrase that \
implies the reader has the source in front of them.

R5. **No bibliographic bridges**: Author names, institutional \
affiliations, journal names, publishers, citations, references, and \
acknowledgments are NOT valid bridges. Even when both chunks share an \
institution or author, refuse instead.

R6. **Short canonical answer**: ≤ 15 tokens. Comparison/arithmetic/\
exclusion/multi_constraint answers are typically computed or synthesised \
and need not be a verbatim span. ``bridge_chain`` answers are typically \
verbatim from chunk_B.

When you GENERATE a question, also return:
  - ``question_type``: one of {comparison, multi_constraint, exclusion, \
arithmetic, bridge_chain}.
  - ``preferred_type_used``: ``true`` if you generated the preferred \
type the seed asked for, ``false`` if you fell back to a different type.
  - ``fact_a``: one plain sentence describing what chunk_A's content \
contributes to the question.
  - ``fact_b``: one plain sentence describing what chunk_B's content \
contributes to the question.
  - ``source_span_A``: a verbatim contiguous excerpt of **4-5 sentences** \
from chunk_A that supplies the descriptor's context. Must be an exact \
substring of chunk_A.
  - ``source_span_B``: a verbatim contiguous excerpt of **4-5 sentences** \
from chunk_B that contains (or directly enables computing) the answer.
  - ``canonical_answer``: the answer (≤ 15 tokens). For ``bridge_chain``/\
``exclusion`` this is typically a span verbatim from chunk_B; for \
``comparison``/``arithmetic``/``multi_constraint`` it may be a synthesised \
value. Whatever it is, write it as the most compact natural form.
  - ``answer_variants``: 0-3 acceptable alternative surface forms.

When you REFUSE, return only ``explanation`` — one plain English \
sentence in your own words explaining why no type works for this seed.

# OUTPUT FORMAT

Return a JSON ARRAY of exactly K objects, in seed order. Each object is \
one of:

  // refusal
  {"seed_id": <int>, "linkable": false, "explanation": "<one sentence>"}

  // accepted
  {"seed_id": <int>, "linkable": true,
   "question_type": "...",
   "preferred_type_used": true|false,
   "fact_a": "<one sentence>",
   "fact_b": "<one sentence>",
   "question": "...",
   "canonical_answer": "...",
   "answer_variants": ["..."],
   "source_span_A": "...",
   "source_span_B": "..."}

Return ONLY the JSON array. No commentary, no markdown fences.

# WORKED EXAMPLES (one per type — patterns apply across any corpus)

Example 1 — preferred ``comparison`` (linkable: true):
  chunk_A: "The 1907 VFL Grand Final was contested between Carlton and \
South Melbourne, held at the MCG on 21 September 1907. Carlton won by 5 \
points, marking the club's second consecutive premiership."
  chunk_B: "The 1909 VFL Grand Final was contested between the same two \
clubs, held at the MCG on 2 October 1909. South Melbourne won by 2 \
points, the club's first premiership."
  question_type: "comparison"
  preferred_type_used: true
  question: "How did the winning margin in the 1909 VFL Grand Final \
compare to the margin in the same fixture two years earlier?"
  canonical_answer: "3 points smaller"
  answer_variants: ["3 fewer points", "smaller by 3"]

Example 2 — preferred ``multi_constraint`` (linkable: true):
  chunk_A: "[Article about a basketball franchise that joined the NBA \
following an ABA merger and changed home arenas after its inaugural NBA \
season.]"
  chunk_B: "[Article naming a specific franchise that fit those criteria.]"
  question_type: "multi_constraint"
  preferred_type_used: true
  question: "Which basketball franchise joined the NBA from the ABA, won \
two ABA championships, and was renamed after its first NBA season?"
  canonical_answer: "New York Nets"
  answer_variants: ["the Nets"]

Example 3 — preferred ``exclusion`` (linkable: true):
  chunk_A: "[Chunk listing every member of an awards body for one year, \
including A, B, C, D, E.]"
  chunk_B: "[Chunk listing the four members who voted on a particular \
resolution that year — A, B, C, D.]"
  question_type: "exclusion"
  preferred_type_used: true
  question: "Which member of that year's awards body did not vote on the \
resolution?"
  canonical_answer: "E"

Example 4 — preferred ``arithmetic`` (linkable: true):
  chunk_A: "[Chunk: tournament's first edition was held in 1964 in \
Warsaw.]"
  chunk_B: "[Chunk: that tournament's third edition was held in 1968 in \
Leipzig.]"
  question_type: "arithmetic"
  preferred_type_used: true
  question: "How many years passed between the first and third editions \
of the European Junior Games?"
  canonical_answer: "4"
  answer_variants: ["4 years", "four"]

Example 5 — preferred ``bridge_chain`` (linkable: true):
  chunk_A: "[Chunk: Phoenix is a protocol proposed in 2018 to address \
the synchronisation problem in distributed databases.]"
  chunk_B: "[Chunk: the synchronisation problem in distributed databases \
was formally proven NP-hard by Müller in 2020.]"
  question_type: "bridge_chain"
  preferred_type_used: true
  question: "What computational complexity has been formally proven for \
the problem the Phoenix protocol was first proposed to address?"
  canonical_answer: "NP-hard"

Example 6 — preferred ``arithmetic``, fell back to ``bridge_chain`` \
(linkable: true, preferred_type_used: false):
  chunk_A: "[Chunk: Sparky Woods coached the 1989 South Carolina \
Gamecocks to a 6-4-1 finish, his first season as head coach.]"
  chunk_B: "[Chunk: in his fourth season the same coach led the same \
program to a 5-6 record in their first SEC season.]"
  question_type: "bridge_chain"
  preferred_type_used: false
  question: "What was the regular-season record of the South Carolina \
football team in their first season under SEC membership, three years \
after their head coach's debut campaign?"
  canonical_answer: "5-6"

Example 7 — refusal (linkable: false; bibliographic-only overlap):
  chunk_A: "[Chunk: Department A, Institute X, City Y, Country Z. Topic \
T1 in domain D1, outcome O1.]"
  chunk_B: "[Chunk: Department B, Institute X, City Y, Country Z. Topic \
T2 in domain D2, outcome O2.]"
  linkable: false
  explanation: "The only overlap between these chunks is the shared \
institutional affiliation; the actual subject matter is unrelated, so \
no substantive 2-hop question is possible."
"""


COMPOSITION_BATCH_USER_PROMPT = """\
Domain context: {domain_description}

You will produce decisions for {k} seeds in this batch. Each seed has \
two chunks from different documents and a **preferred question type**. \
Try first to generate a question of the preferred type; if the pair \
doesn't support it, generate a question of any other type from the \
taxonomy; if no type fits, refuse.

{seed_blocks}

Reminders:
- Both chunks must be necessary; neither alone should suffice.
- Refer to entities indirectly. Do NOT copy distinctive surface tokens \
(document titles, rare proper nouns) verbatim into the question.
- Do not bridge on bibliographic content (authors, citations, journals, \
acknowledgments).
- When in doubt, refuse — do not lower the bar.
"""


SINGLE_HOP_SUFFICIENCY_PROBE_PROMPT = """\
Answer the following question using ONLY the context below. If the context \
is insufficient to determine the answer, output exactly:

INSUFFICIENT

Otherwise output the answer and nothing else — no explanation, no quotes, \
no punctuation. Keep the answer to at most 15 words.

Context:
{context}

Question: {question}

Answer:"""


ORACLE_OPEN_ENDED_PROMPT = """\
Answer the following question using ONLY the context below. The context is \
known to contain the information needed to determine the correct answer. \
Output the answer and nothing else — no explanation, no quotes, no \
punctuation. Keep the answer to at most 15 words.

Context:
{context}

Question: {question}

Answer:"""


NAIVE_RAG_PROMPT = """\
Answer the following question. Use only the provided context if any was \
retrieved; otherwise answer to the best of your ability. Output the answer \
and nothing else — no explanation, no quotes, no punctuation. Keep the \
answer to at most 15 words.

Context:
{context}

Question: {question}

Answer:"""


JUDGE_PROMPT = """\
You are grading a question-answering system.

Question: {question}
Reference answer(s): {gold}
System answer: {pred}

Is the system answer correct? An answer is correct if it conveys the same \
factual information as any reference answer, even if phrased differently. \
Respond with a single token: YES or NO."""
