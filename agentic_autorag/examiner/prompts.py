"""Prompt templates for the open-ended exam pipeline.

Composition prompts (used during exam generation):
  - COMPOSITION_BATCH_SYSTEM_PROMPT: shared system prompt with the
    hardness goal, 7-type reasoning taxonomy, hard rules (H1-H7),
    difficulty preferences (P1-P5), fallback policy, and worked
    examples. The composer receives a neighborhood of related chunks
    and emits as many questions as the chunks support; for each
    question it cites which chunks were used.
  - COMPOSITION_BATCH_USER_PROMPT: per-neighborhood user prompt.

Eval-time prompts (used by the system-under-test and the validator):
  - ORACLE_OPEN_ENDED_PROMPT: feeds all spans concatenated; used by the
    answerability gate during exam generation.
  - NAIVE_RAG_PROMPT: sent to the RAG pipeline at evaluation time.
"""

from __future__ import annotations

COMPOSITION_BATCH_SYSTEM_PROMPT = """\
You are building a DIFFICULT exam to discriminate between the very best \
RAG (retrieval-augmented generation) configurations. A weak RAG \
pipeline should fail many of your questions; even a strong RAG should \
not get them all. The gap between weak and strong RAG configurations \
is what your questions exist to measure — saturated benchmarks (where \
all strong configurations score the same) waste the optimisation \
signal we are trying to extract.

For each call you receive a NEIGHBORHOOD: an anchor chunk plus a \
handful of related chunks (same document, or topically related from \
other documents, depending on the corpus). Generate as many \
high-quality questions as the chunks GENUINELY support — there is no \
upper cap. If the neighborhood is rich, emit many; if sparse, emit a \
few. The single guiding principle is: produce the HARDEST questions \
the chunks support while keeping them well-formed and answerable \
from the cited chunks.

**Before drafting each question, scan the neighborhood for a chain of \
3+ chunks that genuinely depend on each other. If you find one, take \
it.** Only fall back to 2-hop when 3+ isn't reachable, and to 1-hop \
when even 2 isn't reachable. A deep chain across the neighborhood is \
harder for any retriever to assemble than a single bridge — the same \
neighborhood often supports both an easy 2-hop and a harder 3-hop \
framing, and the easy one is the wasted signal. A sharp 1-hop \
question on a rich chunk is still better than a contrived multi-hop \
that doesn't truly require its second chunk; but never default to \
shorter when the chunks support deeper.

For each question you emit, you MUST cite which chunks in the \
neighborhood are required to answer it, AND for each cited chunk \
attach the verbatim source span that supports the answer. The schema \
below ties citation and span together as one object per cited chunk \
so they can never get out of sync. Cite ONLY chunks the question \
genuinely needs — do not pad the selection with decorative chunks \
that aren't load-bearing.

## How your questions reach the reader

A RAG pipeline takes a user's question, retrieves a handful of text \
chunks from a vector index (sometimes refined by a reranker), and \
feeds those retrieved chunks to a generator LLM that produces the \
final answer. The reader sees only their question and the generator's \
answer — never the chunks you saw, never the rest of the corpus.

Three properties follow:

1. **The reader is closed-book.** They do not know what "the chunks", \
"the neighborhood", "Chunk 1", "the passage", "the study", or any \
internal scaffolding refers to. Identify entities by their subject \
matter — what they are, what they do, how they relate — never by \
their position in your input.

2. **Retrieval is per-chunk and independent.** The vector index can \
surface any chunk on its own. For a multi-hop question to actually \
test multi-hop retrieval and reasoning, every cited chunk must be \
LOAD-BEARING: removing any one of them must break the question's \
answerability. If a cited chunk's content can be skipped without \
losing the answer, it doesn't belong in the citation.

3. **The grader checks exact-shape match against a canonical \
answer.** The grader expects the answer in the shape prescribed for \
the ``reasoning_type`` (see below). Treat the canonical answer as a \
contract with the grader, not an explanation.

## Operational stance

When the chunks support a hard framing, take it. When they don't, \
generate the hardest framing the chunks DO support — never refuse \
just because the hardest possible framing isn't reachable. A \
moderately hard question is far better than a refusal; a refusal \
contributes zero signal to the benchmark.

If the entire neighborhood is pure boilerplate (publication metadata, \
single-sentence stubs, no substantive content), emit a single refusal \
entry. Otherwise, you should always emit at least one question.

# REASONING-TYPE TAXONOMY (closed; choose one per question)

Single-hop types (one cited chunk):

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
or a count derived from an explicit enumeration). PREFER operations \
that produce a clean integer or two-decimal canonical; avoid means and \
ratios whose answer needs more than two decimal places — the benchmark \
tests retrieval and composition, not the RAG generator's decimal \
arithmetic. Calendar-date answers do NOT belong here — the formula \
verifier emits durations only. They also don't belong in ``inference`` \
(date arithmetic tests LLM mental math, not retrieval). If your chunks \
only support a calendar-date framing, choose another reasoning_type or \
skip the question. Emit ``formula`` and ``formula_kind: "arithmetic"``; \
the same unit / day-precision / year-precision rules as multi-hop \
``numeric`` apply.
   Answer style: a numeric value with optional units (at most 15 words).

Single- or multi-hop types:

4. ``inference`` — Compose ≥2 facts from DISTINCT spans (either within \
one cited chunk, or across multiple cited chunks) into an answer that \
is NOT a contiguous substring of any single chunk. Allowed cases:
   - **causal chain** — chunk A states X causes Y; chunk B states Y \
causes Z; question asks what X ultimately produced.
   - **implicit-referent resolution** — pronouns or definite-article \
references that only resolve when both chunks are read together.
   - **qualitative direction from quantitative facts** — chunk A \
supplies a baseline measurement; chunk B supplies a follow-up; question \
asks the direction of change (improved / worsened / unchanged), not \
the numeric magnitude.

   NOT calendar-date arithmetic — that tests the LLM's mental math, \
not retrieval. NOT bare numeric arithmetic — use ``numeric`` or \
``numeric_single`` for those. NOT entity-attribute lookup through an \
indirect descriptor — use ``bridge``. NOT side-by-side metric reading \
— use ``comparison``. Use ``inference`` only when none of those fits. \
No formula. Saturate the ``answer_variants`` field — paraphrased \
answers are this type's whole point; any surface form the judge should \
accept (synonyms, alternate phrasings, alternate ordering of compound \
phrases) belongs in the variants list.
   Answer style: a short phrase, value, or qualitative direction (at \
most 15 words).

Multi-hop types (two or more cited chunks):

5. ``bridge`` — Reference an entity in one cited chunk via an indirect \
descriptor that another cited chunk's content uniquely identifies; ask \
for an attribute of that entity. Generalises to 3+ hop chains where \
each chunk's content describes a property of the bridge entity \
identified by the previous chunk.
   Answer style: a short factoid span (at most 15 words).

6. ``comparison`` — Read a comparable value from EACH cited chunk and \
compare. Each chunk's value must be necessary to produce the canonical \
answer.
   Answer style: a comparative phrase ("X is larger" / "Y was earlier" \
/ "the same"), or a numeric difference. Do NOT just ask which one is \
bigger / earlier in a way that's already stated in one chunk.

7. ``numeric`` — Compute across the cited chunks. Read numbers or dates \
and apply arithmetic (difference, sum, ratio, or duration). Subsumes \
the historical ``arithmetic`` and ``temporal`` types.
   Answer style: a numeric value with optional units \
("12", "$50 million", "27 points", "64 days", "12 years").

# FORMULA FIELD (``numeric`` and ``numeric_single`` questions)

For ``numeric`` and ``numeric_single`` questions, emit a ``formula`` \
and ``formula_kind: "arithmetic"`` that the harness evaluates to \
verify the canonical answer. ``formula`` is a Python arithmetic \
expression over numeric literals only. Examples: ``2012 - 1948``, \
``(300 + 250) / 2``, ``11 * 27``, ``21 + 29``. No variable names, \
no function calls, no attribute access.

For temporal differences, the formula and answer must use the SAME \
unit, and you may only emit day-precision when the gap is small \
enough for a reader to verify mentally:

- Day-precision (≤ ~30 days): chunks must state day-precision dates \
AND the difference must be at most one month. Encode as integer day \
arithmetic against day-of-month numbers from the chunks (e.g. \
``19 - 5`` → ``"14 days"``). Day-precision arithmetic is not \
supported beyond ~30 days.
- Year-precision (≥ ~1 year): integer year difference (``2011 - 2008``) \
and answer in years (``"3 years"``).
- Anything in between (~1 to ~12 months): if chunks state year and \
month, use month-precision arithmetic (``11 - 4`` → ``"7 months"``). \
Otherwise prefer a ``comparison`` question instead.

Do NOT manufacture day-precision by multiplying year differences by \
365, or month differences by 30 — the verifier catches the unit \
mismatch and rejects. The output unit must match what the formula \
computes.

For types other than ``numeric`` and ``numeric_single``, set \
``formula`` and ``formula_kind`` to null.

# HARD RULES (H1-H7) — refuse a question only if you cannot satisfy ALL of these

H1. **Multi-hop load-bearing.** For each chunk you cite in \
``cited_chunks``, removing that chunk must break the question's \
answerability. Do not pad citations with decorative chunks. If only \
one chunk is genuinely needed, cite only one and produce a \
single-hop question.

H2. **Self-contained closed-book phrasing.** No "the document", "the \
passage", "the above text", "the study", "the trial", "the \
experiment", "the analysis", "the present work", "according to the \
paper", "Chunk 1", "Chunk 2", "the first chunk", "the second chunk", \
or any phrase that implies the reader has the source in front of \
them. On research-paper corpora these phrases are natural in the \
chunks but make the question impossible to answer without seeing the \
source — identify the work by intervention, population, mechanism, \
or topic instead.

H3. **No meta-content.** Don't compose questions about author names, \
institutional affiliations, journal names, publishers, citations, \
references, acknowledgments, competing-interests declarations, \
contributor lists, funding statements, copyright notices, or any \
other publication-boilerplate content. Even when two chunks share \
an institution, an author, or identical "no competing interests" \
text, refuse the question instead — these are not substantive content.

H4. **Short canonical answer:** at most 15 words. Applies to every \
type. The harness rejects longer answers at parse time. \
``comparison``/``numeric``/``numeric_single``/``inference`` answers \
are typically computed or synthesised; \
``extraction``/``definitional``/``bridge`` answers are typically \
verbatim or near-verbatim from a chunk.

H5. **Canonical answer shape matches the per-type Answer style \
exactly.** The eval-time grader expects answers in the shape \
prescribed for ``reasoning_type``, and ranks RAG configs by how \
closely they match. A full English sentence ("Yes, both were played \
at the Lake Oval in Albert Park.") is NOT a valid ``comparison`` \
canonical — the shape is a phrase ("Same venue, Lake Oval"). For \
``numeric`` and ``numeric_single``, emit just the value plus \
optional unit ("13 points", "12 years", "$50 million") — never \
wrap it in a sentence. For ``bridge``/``extraction``, emit just the \
entity name or factoid span. ``definitional`` admits a brief \
description, but still no leading "It is …" / "The term refers \
to …" hedges. For ``inference``, emit just the derived phrase, \
date, or value — no preamble.

For ``comparison`` questions where the compared entities lack a \
short canonical name (only descriptions like "the triangular \
building on the corner" or "the town north of Orangetown"), use a \
**relational comparative phrase** that does not require naming the \
entity ("the older one", "the taller", "the earlier-built", "lower \
by 2.9 points") and saturate ``answer_variants`` with the \
descriptive surface forms a RAG generator might reasonably \
produce. NEVER emit a multi-sentence description as the \
``comparison`` canonical — that is bridge / extraction shape, not \
comparison shape, and the grader will reject correct relational \
answers against it.

H6. **Formula required for ``numeric`` and ``numeric_single``.** \
Emit ``formula`` and ``formula_kind: "arithmetic"`` so the verifier \
can check the math. The verifier rejects questions of these types \
without a formula.

H7. **The question must require the chunks.** The answer cannot be \
derivable from the question text alone. If a date, count, or \
quantity needed for the answer ALSO appears as a descriptor in your \
question text, the question is self-answering and tests nothing — \
rewrite the question, or only if no rewrite is possible, refuse. \
Identify entities by role, distinguishing event, or relationship, \
NOT by the specific numeric value the question asks about. Examples \
to REWRITE: "which is earlier, the show that aired in January 2005 \
or the one in November 2010?" (dates supplied by the question); \
"by how much does the team's 21 consecutive seasons exceed the \
rival's 12 seasons?" (counts supplied by the question).

# DIFFICULTY PREFERENCES (P1-P5) — try to satisfy; NEVER refuse for failing them

These are the levers for the hardness goal stated at the top. Try to \
satisfy each one — the more you satisfy, the harder the question. \
**Failing a preference is never grounds to refuse.** A simpler \
question still contributes signal; a refusal contributes none.

The examples below illustrate the kind of choice each preference \
involves. **They are NOT recipes — match the spirit, not the surface \
form.**

P1. **Prefer indirect descriptors over direct entity naming.** When \
the chunks allow it, reference entities through their role, \
relationship, or definitional descriptor rather than copying a \
distinctive proper noun verbatim. Indirect framing forces the \
retriever to do real semantic work; direct naming reduces the \
question to lexical matching.

  Concrete operational guidance: **for each cited chunk, prefer NOT \
to copy that chunk's distinctive proper nouns into the question \
text.** Describe the entity by its role, attribute, or relation \
instead. Verbatim proper-noun overlap between the question and the \
cited chunks lets a weak retriever find the gold chunks by lexical \
match alone — the question then tests neither retrieval nor reasoning.

  Harder framing: "Who founded the company that the acquirer of \
the early-stage biotech acquired in 1998?"
  OK framing (use when indirect would be awkward or the descriptor \
would be too long to be natural): "Who founded Beta Inc?"

  If indirect framing makes the question awkward or impossible, \
use the direct name — don't twist the question. Soft preference, \
not a hard rule.

  **One indirect descriptor is enough — do NOT stack.** Pick the \
SINGLE most indirect way to refer to the answer entity, not three \
attributes chained together. Stacking 3+ distinct descriptors of one \
entity ("the X founded in 1985 by Y in the city of Z" is three) is \
the most common way to leak gold chunks via lexical match — each \
attribute is usually a keyword cluster from the gold chunk, and the \
combination lets a weak retriever find the gold chunks trivially. \
Lean indirection is harder than dense attribution. As a guideline, \
questions usually land at ≤ 25 words; if you're past 30 words, you \
are probably stacking — rewrite leaner.

  Distinguish **two entity roles** in the question:
  - **The retrieval anchor.** A named entity, dated event, or \
specific multi-word descriptor that drives retrieval to the cited \
chunks. This SHOULD be present in the question. Without it ("the \
cohort", "the building", "the protocol"), retrieval has no signal \
and the question is unanswerable regardless of RAG quality.
  - **The answer entity.** What the question asks ABOUT — whose \
attribute or identity the canonical answer reveals. THIS one is \
described indirectly (by role, relation, or attribute) so the \
answer text isn't lexically present in the question.

  The "prefer NOT to copy distinctive proper nouns" rule above \
applies to the **answer entity's** identifying proper nouns (those \
leak the answer via lexical match). It does NOT apply to the \
retrieval anchor — strip the anchor and retrieval has nothing to \
lock onto.

P2. **Anchor the question; paraphrase the anchor; indirect-describe \
the answer.**

  (a) Your question needs ONE corpus-distinctive anchor — a named \
entity, dated event, or specific descriptor — so a vector retriever \
can find the cited chunks. Without one, no RAG can answer your \
question and you contribute noise instead of difficulty.

  (b) **Where possible, paraphrase the anchor rather than copying \
its lexical surface from the chunks.** If the chunk says "refractory \
hypertension", call it "treatment-resistant hypertension" in the \
question; if the chunk says "myosin-inhibitor", call it "cardiac- \
muscle-protein modulator". A semantic anchor forces the embedding \
model to bridge to the chunk's lexical surface (the real test of \
vector retrieval); a verbatim lexical anchor rewards BM25 and weak \
retrievers that pattern-match keywords. Paraphrase descriptors \
aggressively. **Proper nouns (people, places, named drugs/protocols) \
are fine verbatim** — they're the natural retrieval anchor \
regardless of phrasing, and forcing paraphrase risks contrived or \
wrong substitutions ("Phoenix protocol" → "the 2018 sync protocol" \
is forced).

  (c) The ANSWER ENTITY can be described indirectly (by role, \
relation, or attribute). The cited chunks identify the answer; you \
don't spell every distinguishing attribute into the question.

  Anchored + paraphrased + indirect (good): "How did year-5 \
mortality compare between the two arms of the 412-patient trial of \
treatment-resistant hypertension?" — paraphrased anchor; comparative \
answer not in question.
  Anchored + lexical-verbatim (weaker): "How did year-5 mortality \
compare between arms of the 412-adult refractory-hypertension \
cohort?" — verbatim anchor; BM25 wins trivially.
  Under-anchored (bad — kills retrieval): "How did year-5 mortality \
compare between the cohort's arms?" — no corpus signal.
  Over-disambiguated (bad — leaks answer): five descriptors stacked \
on the answer entity.

P3. **Prefer multi-step reasoning over one-step lookup.** When the \
neighborhood supports a 3-hop or 4-hop chain, take it — a longer \
chain of load-bearing chunks stresses retrieval more than a 2-hop \
bridge. For ``inference``, compose facts from distant sentences or \
spans of the chunk — single-sentence lookups are ``extraction`` \
mislabelled. For ``numeric_single``, combine ≥2 numeric literals \
when available. When the chunks only support a one-step factoid, \
fall back gracefully to ``extraction`` or ``definitional``.

P4. **Prefer non-obvious target attributes.** Don't always pick the \
title, the headline date, or the first sentence — look for \
attributes the chunk states but doesn't foreground.

P5. **Prefer comparisons whose answer isn't obvious from general \
world knowledge.** Birth-before-later-work comparisons, comparisons \
based on coincident numbers across topically unrelated chunks \
("both happen to be 3" between a glove-test count and a \
shoulder-implant count), and bare year/month subtraction where the \
answer is mentally obvious — these are weaker than comparisons that \
require the chunks to resolve. If only such comparisons fit the \
inputs, try another multi-hop type (the same chunks may support a \
stronger ``bridge`` or ``numeric``); if no other type fits either, \
generate the weak comparison rather than refuse.

# FALLBACK POLICY

If the chunks support no multi-hop framing, generate the hardest \
single-hop question on the richest chunk. If the entire neighborhood \
is pure boilerplate (publication metadata, single-sentence stubs, no \
substantive content), emit a single refusal entry. Otherwise, you \
should always emit at least one question.

NEVER twist the chunks to fit a type that doesn't apply.

# OUTPUT — fields per question entry

For each accepted question, return:
  - ``reasoning``: 1-3 sentences explaining what each cited chunk \
contributes and how the cited chunks together pin a unique canonical \
answer. The question must contain a corpus-distinctive anchor (a \
named entity, dated event, or specific descriptor — paraphrased from \
the chunks' wording where possible) so retrieval can locate them; \
the answer entity itself can be described indirectly to hide the \
answer from lexical match. (Internal — forces explicit thinking; \
not stored.)
  - ``reasoning_type``: one of {extraction, definitional, \
numeric_single, inference, bridge, comparison, numeric}.
  - ``cited_chunks``: an array of objects, one per cited chunk. Each \
object has two fields:
      - ``chunk_id``: integer position from the ``[Chunk N]`` label \
in the user prompt.
      - ``span``: verbatim contiguous excerpt from that chunk \
containing the evidence the answer relies on — typically 2-5 \
sentences, or the whole chunk if shorter. Must be an EXACT substring \
of the chunk (do not paraphrase or normalise whitespace).
    Cite ONLY chunks the question genuinely needs (H1). Every cited \
chunk MUST come with its span — they live inside the same object so \
they can never get out of sync.
  - ``question``: the question text.
  - ``canonical_answer``: the answer (at most 15 words).
  - ``answer_variants``: 0-5 acceptable alternative surface forms. \
``inference`` questions should saturate this field with paraphrases \
the judge should accept; other types typically need 0-2.
  - ``formula``: arithmetic expression or null.
  - ``formula_kind``: ``"arithmetic"`` or null.

For a refusal (only when the entire neighborhood is unusable), \
return a single entry of the form:
  ``{"linkable": false, "explanation": "<one sentence>"}``

# OUTPUT FORMAT

Return a JSON ARRAY of question entries. Each entry is one of:

  // accepted question
  {"linkable": true,
   "reasoning": "...",
   "reasoning_type": "...",
   "cited_chunks": [
     {"chunk_id": <int>, "span": "<verbatim excerpt>"},
     {"chunk_id": <int>, "span": "<verbatim excerpt>"}
   ],
   "question": "...",
   "canonical_answer": "...",
   "answer_variants": ["..."],
   "formula": null | "...",
   "formula_kind": null | "arithmetic"}

  // refusal (only one such entry, only when the whole neighborhood is unusable)
  {"linkable": false, "explanation": "<one sentence>"}

Return ONLY the JSON array. No commentary, no markdown fences.

# WORKED EXAMPLES

The worked examples below illustrate the SHAPES of valid questions \
and the KINDS of reasoning each type calls for. **THEY ARE NOT \
TEMPLATES.** Do NOT copy the surface form, subject matter, units, \
time spans, operations, or sentence patterns of any one example — \
the chunk's actual content drives the question; the example only \
shows what good output for the type looks like.

The "Between the X and Y, which has more Z?" construction in \
Example 3 is ONE valid comparison shape; many alternatives exist — \
"Of the two Italian-born…", "Of the films released in 1985, \
which…", "Was the X earlier than the Y?", or a relational form \
like "X precedes Y by how many years?". **Vary phrasing \
aggressively.** If your question's sentence structure matches an \
example almost verbatim, rewrite it. The example teaches what \
reasoning_type the chunks support; the chunks themselves drive the \
question wording.

Example 1 — strong ``bridge`` (multi-hop):
  Neighborhood includes:
    [Chunk 2] "Phoenix is a protocol proposed in 2018 to address the \
synchronisation problem in distributed databases."
    [Chunk 5] "The synchronisation problem in distributed databases \
was formally proven NP-hard by Müller in 2020."
  reasoning_type: "bridge"
  cited_chunks:
    - {chunk_id: 2, span: "Phoenix is a protocol proposed in 2018 to \
address the synchronisation problem in distributed databases."}
    - {chunk_id: 5, span: "The synchronisation problem in distributed \
databases was formally proven NP-hard by Müller in 2020."}
  reasoning: "Chunk 2 identifies which problem the named protocol \
targets (synchronisation in distributed databases); Chunk 5 states \
that problem's complexity. Removing Chunk 2 leaves no link between \
Phoenix and the complexity result; removing Chunk 5 leaves no \
complexity to report. The descriptor 'the problem the Phoenix \
protocol was first proposed to address' uniquely identifies one \
problem."
  question: "What computational complexity has been formally proven \
for the problem the Phoenix protocol was first proposed to address?"
  canonical_answer: "NP-hard"

Example 2 — strong ``comparison`` (multi-hop):
  Neighborhood includes:
    [Chunk 0] "In the active arm of a 412-adult cohort with \
refractory hypertension, all-cause mortality at year 5 was 11.2%."
    [Chunk 3] "In the matched control arm of the same 412-adult \
refractory-hypertension cohort, all-cause mortality at year 5 was \
14.1%."
  reasoning_type: "comparison"
  cited_chunks:
    - {chunk_id: 0, span: "In the active arm of a 412-adult cohort \
with refractory hypertension, all-cause mortality at year 5 was 11.2%."}
    - {chunk_id: 3, span: "In the matched control arm of the same \
412-adult refractory-hypertension cohort, all-cause mortality at \
year 5 was 14.1%."}
  reasoning: "Both chunks supply year-5 all-cause mortality figures \
(11.2% active, 14.1% control) for distinguishable arms of the same \
cohort; the comparison requires reading both values."
  question: "How did year-5 mortality compare between the two arms \
of the 412-patient trial of treatment-resistant hypertension?"
  canonical_answer: "2.9 percentage points lower"
  answer_variants: ["lower by 2.9 percentage points"]

Example 3 — strong ``comparison`` over entities without canonical \
short names (multi-hop):
  Neighborhood includes:
    [Chunk 1] "Anchoring the southern apex of Madison Square in \
Manhattan, a triangular masonry building was completed in 1902 and \
rises twenty-two storeys above the avenue."
    [Chunk 4] "Diagonally across Manhattan's Madison Square, a \
steel-frame office tower was finished in 1924 and tops out at \
nineteen storeys."
  reasoning_type: "comparison"
  cited_chunks:
    - {chunk_id: 1, span: "Anchoring the southern apex of Madison \
Square in Manhattan, a triangular masonry building was completed in \
1902 and rises twenty-two storeys above the avenue."}
    - {chunk_id: 4, span: "Diagonally across Manhattan's Madison \
Square, a steel-frame office tower was finished in 1924 and tops \
out at nineteen storeys."}
  question: "Of the two pre-Depression-era towers around Manhattan's \
Madison Square, which has more storeys?"
  canonical_answer: "the older one"
  answer_variants: ["the earlier-built one", "the triangular one", \
"the 1902 building", "the masonry building", "the one with 22 storeys"]

Example 4 — strong ``numeric`` (multi-hop, NON-date arithmetic):
  Neighborhood includes:
    [Chunk 0] "The active arm of the cohort enrolled 412 adults \
with refractory hypertension at three sites."
    [Chunk 6] "The matched control arm of the same cohort enrolled \
298 adults at the same three sites."
  reasoning_type: "numeric"
  cited_chunks:
    - {chunk_id: 0, span: "The active arm of the cohort enrolled 412 \
adults with refractory hypertension at three sites."}
    - {chunk_id: 6, span: "The matched control arm of the same \
cohort enrolled 298 adults at the same three sites."}
  question: "What was the total enrolment across both arms of the \
three-centre trial of treatment-resistant hypertension?"
  canonical_answer: "710 adults"
  formula: "412 + 298"
  formula_kind: "arithmetic"

Example 5 — strong 3-hop chain (multi-hop, three cited chunks):
  Neighborhood includes:
    [Chunk 2] "Helios Therapeutics was founded in 2009 by Dr. \
Anika Rao, a former Genentech immunologist."
    [Chunk 7] "In 2015, Helios Therapeutics was acquired by \
NorthStar Biosciences in a $1.2B cash-and-stock deal."
    [Chunk 9] "NorthStar Biosciences is headquartered in Boston, \
Massachusetts, and operates research facilities in Cambridge and \
Watertown."
  reasoning_type: "bridge"
  cited_chunks:
    - {chunk_id: 2, span: "Helios Therapeutics was founded in 2009 \
by Dr. Anika Rao, a former Genentech immunologist."}
    - {chunk_id: 7, span: "In 2015, Helios Therapeutics was acquired \
by NorthStar Biosciences in a $1.2B cash-and-stock deal."}
    - {chunk_id: 9, span: "NorthStar Biosciences is headquartered in \
Boston, Massachusetts, and operates research facilities in Cambridge \
and Watertown."}
  reasoning: "Chunk 2 establishes the founder identity for an \
indirectly-described biotech; Chunk 7 identifies the 2015 acquirer; \
Chunk 9 provides the headquarters location. Removing any one chunk \
breaks the chain — the question cannot be answered from any pair \
alone."
  question: "In which city is the parent company that acquired the \
biotech founded by the former Genentech immunologist headquartered?"
  canonical_answer: "Boston"
  answer_variants: ["Boston, Massachusetts", "Boston, MA"]

Example 6 — refusal (meta-content / publication boilerplate, H3):
  Neighborhood is dominated by:
    [Chunk 0] "Competing interests: We declare we have no competing \
interests."
    [Chunk 1] "Competing interests: We declare we have no competing \
interests."
  linkable: false
  explanation: "The neighborhood is dominated by 'no competing \
interests' boilerplate found on most research papers — \
substantively empty and shared across many documents in any \
research-paper corpus."

Example 7 — refusal (fake-bridge trap, H1 multi-hop load-bearing):
  A composer was tempted by:
    [Chunk 2] "Microsoft Decision Tree, although it has very low \
sensitivity and extremely high specificity, has the highest accuracy."
    [Chunk 8] "This research compared a closed-source algorithm \
(Microsoft Decision Tree) with open-source algorithms (CART and \
C4.5) using data from the U.S. Surveillance, Epidemiology, and End \
Results Program (SEERS)."
  The composer drafted: "What dataset underlies the evaluation of \
the decision tree with extremely high specificity?" — but Chunk 8 \
ALONE names both Microsoft Decision Tree and SEERS. The Chunk 2 \
clue ('high specificity, low sensitivity') is decoration, not a \
load-bearing hop. The composer must not cite Chunk 2 in \
``cited_chunks`` for this question — only Chunk 8 is needed, making \
it single-hop. Either reframe as a single-hop question citing only \
Chunk 8, or find a different question in the neighborhood that \
genuinely needs both chunks.

Example 8 — strong ``numeric_single`` (single-hop, sum across \
enumerated subgroups):
  Neighborhood includes:
    [Chunk 0] "The protocol allocated participants to three \
exposure tiers: 184 received the low dose of the myosin-inhibitor \
candidate, 271 the standard dose, and 192 the high dose. \
Stratification was by baseline left-ventricular wall thickness."
  reasoning_type: "numeric_single"
  cited_chunks:
    - {chunk_id: 0, span: "The protocol allocated participants to \
three exposure tiers: 184 received the low dose of the \
myosin-inhibitor candidate, 271 the standard dose, and 192 the high \
dose. Stratification was by baseline left-ventricular wall thickness."}
  question: "What was the total enrolment across the three exposure \
tiers of the cardiac-muscle-protein-modulator dose-finding trial?"
  canonical_answer: "647 participants"
  formula: "184 + 271 + 192"
  formula_kind: "arithmetic"

Example 9 — strong ``inference`` (multi-hop, qualitative direction \
from quantitative facts across chunks):
  Neighborhood includes:
    [Chunk 0] "Across the Greenland summit ice cores, the mean \
annual surface temperature anomaly relative to 1961-1990 averaged \
+0.4°C during the 1990s."
    [Chunk 4] "By the 2010s, the same Greenland summit cores \
recorded a mean annual surface temperature anomaly of +2.1°C against \
the 1961-1990 baseline."
  reasoning_type: "inference"
  cited_chunks:
    - {chunk_id: 0, span: "Across the Greenland summit ice cores, \
the mean annual surface temperature anomaly relative to 1961-1990 \
averaged +0.4°C during the 1990s."}
    - {chunk_id: 4, span: "By the 2010s, the same Greenland summit \
cores recorded a mean annual surface temperature anomaly of +2.1°C \
against the 1961-1990 baseline."}
  reasoning: "Chunk 0 supplies the 1990s baseline (+0.4°C); Chunk 4 \
supplies the 2010s follow-up (+2.1°C). Composing the two yields the \
qualitative direction (grew). Neither chunk alone says 'the anomaly \
grew' — that has to be inferred by comparing the two numeric values \
across chunks. The question asks direction, not magnitude, so this \
is ``inference``, not ``numeric``."
  question: "Between the 1990s and the 2010s, did the Greenland \
summit temperature anomaly grow or shrink?"
  canonical_answer: "grew"
  answer_variants: ["grew larger", "increased", "got bigger", \
"rose", "expanded"]
"""


COMPOSITION_BATCH_USER_PROMPT = """\
Compose the hardest valid questions this neighborhood supports — \
your job is to widen the gap between weak and strong RAG \
configurations. Generate as many high-quality questions as the \
chunks genuinely support; multi-hop wherever the chunks allow it. \
There is no upper cap on the number of questions you emit.

Domain context: {domain_description}

=== Neighborhood (anchor: chunk_id={anchor_chunk_id}) ===

{chunk_blocks}

Reminders:
- Scan the neighborhood for a 3+ hop chain BEFORE drafting; if one \
exists, take it. Fall back to fewer hops only if the chunks don't \
support more.
- For each question, list the chunks you used in ``cited_chunks`` as \
``[{{"chunk_id": <int>, "span": "<verbatim excerpt>"}}]``. Use the \
integer position from the ``[Chunk N]`` labels. Cite ONLY chunks the \
question genuinely needs (H1 load-bearing).
- Self-contained closed-book phrasing (H2). Never reference the \
chunks, the neighborhood, "the document", etc.
- Prefer NOT to copy distinctive proper nouns from the cited chunks \
into the question text (P1) — describe entities by role/attribute \
instead, unless indirect framing would be awkward or ambiguous.
- For ``numeric`` and ``numeric_single``, emit ``formula`` and \
``formula_kind: "arithmetic"`` (H6).
- For ``inference``, saturate ``answer_variants`` with paraphrases \
the judge should accept.
- Refuse only if the ENTIRE neighborhood is pure boilerplate. \
Otherwise, always emit at least one question.
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


MULTI_HOP_DEPENDENCY_AND_ORACLE_PROMPT = """\
You are evaluating a multi-hop reading comprehension question.

You will see a question and {num_spans} supporting text spans, each \
extracted from a different passage. Your tasks:

1. STRUCTURAL JUDGMENT — Determine the minimum set of spans needed to \
fully answer the question. A span set is sufficient ONLY IF a reader \
using only those spans, with no outside knowledge or training data, \
could derive the answer.
   - For each span you claim is part of the sufficient set, quote the \
exact literal text from that span that supports your reasoning. If you \
cannot quote it, the span is not contributing.
   - Comparison questions need data from EACH item being compared.
   - "Both X and Y" or conjunction questions need verification of each \
conjunct.
   - Bridge questions need the bridge fact (the link from the question's \
premise to the answer entity) to be explicitly quotable from a span, not \
inferred from training.
   - If a span contains the answer entity but the question's reasoning \
requires a fact that is NOT quotable from that span, the span is not \
sufficient on its own.

2. ANSWER — Provide the answer to the question using all the spans \
together.

Expected answer format: {answer_format_hint}

Output a single JSON object and nothing else:
{{
  "reasoning": "<one short sentence per span describing what it contributes>",
  "supporting_quotes": {{"<span_idx>": "<exact quote from that span>", ...}},
  "sufficient_spans": [<sorted list of span indices that together are necessary AND sufficient>],
  "answer": "<your answer, at most 15 words, no quotes or punctuation>"
}}

Question: {question}

{spans_block}"""


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
        "a comparative phrase such as 'X is larger' or 'Y was earlier', "
        "or a numeric difference with units (at most 15 words)"
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
