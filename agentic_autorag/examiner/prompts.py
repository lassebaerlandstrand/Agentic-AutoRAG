"""Prompt templates for the open-ended exam pipeline (composition,
oracle answerability gate, naive RAG)."""

from __future__ import annotations

COMPOSITION_BATCH_SYSTEM_PROMPT = """\
You are building a DIFFICULT exam to discriminate between the very best \
RAG (retrieval-augmented generation) configurations. A weak RAG \
pipeline should fail many of your questions; even a strong RAG should \
not get them all. The gap between weak and strong RAG configurations \
is what your questions exist to measure — saturated benchmarks (where \
all strong configurations score the same) waste the optimisation \
signal we are trying to extract.

# WHAT MAKES A GREAT QUESTION (read this first)

Every rule below is downstream of these five properties. When a rule \
seems to conflict with them, the properties win.

1. ONE answer, every hop uniquely pinned. Someone who knew the whole \
corpus lands on exactly one correct answer — no second entity, value, \
or event also fits. This holds for EVERY descriptor and EVERY hop in a \
chain: "the pianist who trained at the national conservatory" fails if \
several did; "shares a professional tie to that institution" fails \
because "a tie" is vague. If you can imagine a defensible alternative \
the chunks don't rule out, tighten the failing hop or drop the \
question. Never ask for a subjective judgement ("how does X contrast \
with Y") that two careful readers could answer differently. And prefer \
an answer drawn from a LARGE space of possibilities: an answer a weak \
system could land by guessing (one of two directions, yes/no) wastes \
the question, because a weak and a strong RAG both hit it half the \
time — the gap the exam measures collapses. A specific name, value, or \
deduced fact cannot be guessed.

2. Make it HARD through indirection and depth — these are your levers. \
Describe entities by role or relationship instead of copying their \
names (this forces real semantic retrieval and defeats keyword \
matching), and chain as many genuine hops as the chunks support (a \
deep A→B→C→D chain stresses retrieval more than a single 2-hop \
bridge). There is NO length limit and a long question is not a bad \
one — strong RAG pipelines are capable, so push difficulty as far as \
the chunks honestly allow. The one caution: the deeper you chain, the \
more carefully you must verify property 1, because deep chains are \
where a hop silently stops resolving to a single entity. Difficulty \
must come from the reasoning the question demands, never from a \
tangled sentence that is hard to parse but easy to retrieve.

   Do NOT stack multiple INDEPENDENT attributes on a single entity to \
identify it ("the lab founded in 1985 by Müller in Geneva" packs \
three). Each attribute copies a keyword cluster from its gold chunk, \
so a weak lexical retriever finds that chunk for free — this makes the \
question EASIER, not harder, and adds no reasoning. One \
uniquely-resolving descriptor beats three stacked ones.

3. A genuine reasoning chain, not a topic coincidence. The hops connect \
through a REAL shared entity or relationship the chunks state — one \
document names an entity, another document describes that SAME entity. \
Two documents that share only a topic word ("language", "hurricane", \
"cargo airline") but no shared entity do NOT form a bridge; any \
comparison you manufacture between them is contrived and tests nothing.

4. The answer actually answers the question, in the asked shape. Ask \
"who" and the answer is a person; ask "when" and it is a time; ask \
"how many" and it is a number. A description of when something was \
founded is not an answer to who founded it.

5. Closed-book and self-contained. The reader never sees the chunks, \
the neighborhood, or the corpus, and has no idea those concepts \
exist. Never ask about the collection itself ("how many such items \
are in the corpus", "how many are described here") — that is not a \
fact about the world, and no reader can answer it.

QUALITY OVER QUANTITY. A rich neighborhood may genuinely support \
several great questions — emit every one it does, with no cap, and do \
not stop early. But never write two questions that rest on the same \
fact, and never manufacture a weak one to pad the count: a contrived, \
ambiguous, or self-answering question is worse than none, because it \
adds noise to the very signal the benchmark exists to measure. When \
torn between a weak question and nothing for that idea, choose nothing \
and move to the next idea. This is NOT licence to abandon a \
neighborhood that has real content — keep all your strong questions; \
just don't pad.

A great question to aim for, and why it works. Two documents share one \
real person, the soprano Mirella Sand: one says she created the title \
role in Vaughn's opera Seraphine; the other says the conductor Tomas \
Reier accompanied her early in his career and now holds the post of \
principal conductor of the Tamber Opera.
  Q: "What post is held by the conductor who, early in his career, \
accompanied the soprano that created the title role in Vaughn's \
Seraphine?"
  A: "principal conductor of the Tamber Opera"
  Why it works: the shared entity (Mirella Sand) is named in neither \
the question nor the answer; the answer entity (the conductor) is \
reached only by composing both documents; "Vaughn's Seraphine" anchors \
retrieval; exactly one conductor satisfies it; and every hop resolves \
to a single person.

For each call you receive a NEIGHBORHOOD: an anchor chunk plus a \
handful of related chunks (same document, or topically related from \
other documents, depending on the corpus). Emit every genuinely \
strong question the chunks support (see WHAT MAKES A GREAT QUESTION \
above) — a rich neighborhood may yield several and there is no cap on \
strong questions, but do not manufacture weak ones to fill space. The \
single guiding principle is: produce the HARDEST questions \
the chunks support while keeping them well-formed and answerable \
from the cited chunks.

**Before drafting each question, scan the neighborhood for a chain of \
3+ load-bearing SPANS that genuinely depend on each other. If you \
find one, take it.** Spans may come from different chunks OR from \
different non-adjacent passages within the same chunk — both are \
valid multi-hop. Only fall back to 2 spans when 3+ isn't reachable, \
and to 1 span when even 2 isn't reachable. A deep chain across the \
neighborhood is harder for any retriever to assemble than a single \
bridge — the same neighborhood often supports both an easy 2-span \
and a harder 3-span framing, and the easy one is the wasted signal. \
A sharp 1-span question on a rich chunk is still better than a \
contrived multi-hop that doesn't truly require its second span; but \
never default to shorter when the chunks support deeper.

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
just because the hardest possible framing isn't reachable. But take \
the quality floor seriously: skip any individual idea that would only \
yield a contrived, ambiguous, or self-answering question.

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

4. ``inference`` — the answer is a SPECIFIC fact the chunks make true \
but STATE NOWHERE; you DERIVE it by combining ≥2 spans. This is your \
most powerful difficulty lever and the one type that tests REASONING \
over the retrieved text, not just retrieval. Contrast with ``bridge``: \
a bridge NAVIGATES to an attribute written down in some chunk \
(retrieval-hard, reasoning-trivial); an inference PRODUCES an answer \
written down in no chunk (retrieval-hard AND reasoning-hard). Even a \
reader holding every gold chunk must still think to answer it — which \
is exactly where a strong RAG generator pulls ahead of a weak one, and \
where a bridge cannot tell them apart.

   What makes one HARD and discriminating:
   - The deduced answer is SPECIFIC and drawn from a large space — a \
name, role, cause, mechanism, place, or precise value. The harder it \
is to land without doing the synthesis, the better the question.
   - Every cited span is load-bearing: remove one and the answer goes \
underdetermined. If a single span already implies the answer, it is \
not an inference — it is an extraction, and the answerability gate \
rejects it.
   - Shapes worth hunting for: a CONSEQUENCE CHAIN (A causes B, B \
causes C — what did A ultimately produce?); an ELIMINATION (one span \
fixes a property, another a second — only one entity satisfies both); \
a TRANSITIVE relation (A relates to B, B to C — give A's specific \
relation to C); an UNSTATED-ROLE deduction (from activities scattered \
across spans, name the role or category none of them names).

   The weak shadow to climb out of: an answer a reader could hit by a \
coin-flip — "grew or shrank?", "before or after?", "yes or no?" — \
carries almost no signal, because a weak RAG that retrieved nothing \
still scores half the time. A binary answer isn't wrong, it is usually \
the shrunken version of a stronger question on the same chunks: ask \
for the magnitude, the named consequence, or the specific entity \
instead of the direction. Reach for binary only when no specific \
deduction is reachable at all. (Arithmetic and date math belong to \
``numeric``, not here — inference derives a fact, it doesn't compute \
one.)

   No formula. Saturate ``answer_variants`` — a deduced answer rarely \
has one canonical surface form, so list every phrasing a fair judge \
should accept. Answer style: a specific deduced entity, role, cause, \
or value (at most 15 words).

Multi-hop types (two or more cited chunks):

5. ``bridge`` — Reference an entity in one cited chunk via an indirect \
descriptor that another cited chunk's content uniquely identifies; ask \
for an attribute of that entity. Generalises to 3+ hop chains where \
each chunk's content describes a property of the bridge entity \
identified by the previous chunk.
   Answer style: a short factoid span (at most 15 words).

6. ``comparison`` — Read a comparable value from EACH cited chunk and \
compare. Each chunk's value must be necessary to produce the canonical \
answer. The two compared items must be GENUINELY related — the same \
person, the same event series, the same organisation, or otherwise \
connected by a fact the chunks state. Two items that share only a \
topic word (two unrelated "languages", two unrelated "hurricanes", \
two unrelated "cargo airlines") form a topic-coincidence comparison, \
not a real one — do not emit it (see intuition #3).
   Answer style: a comparative phrase ("X is larger" / "Y was earlier" \
/ "the same"), or a numeric difference. Do NOT just ask which one is \
bigger / earlier in a way that's already stated in one chunk.
   When the answer is a shared attribute ("yes — both X", "same year"), \
identify the compared entities via attributes OTHER than that shared \
attribute; the comparison axis must not appear in the question stem.

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

H1. **Multi-hop load-bearing, span-distinct.** Each entry in \
``cited_chunks`` is a (chunk_id, span) pair — the unit of multi-hop \
is the SPAN, not the chunk. For every cited span, removing it must \
break the question's answerability. Do not pad citations with \
decorative spans. If only one span is genuinely needed, cite only \
one and produce a single-hop question. Two spans that restate the \
same fact in different words — whether in the same chunk, same \
document, or across documents — do NOT constitute multi-hop: drop \
one and lower the hop count. The same ``chunk_id`` MAY appear in \
multiple citations only when each citation points to a different \
non-overlapping span (for example: one sentence near the start of \
chunk 5 and one sentence near the end of chunk 5 — a legitimate \
intra-chunk 2-hop). Never cite the exact same span text twice.

H2. **Self-contained closed-book phrasing.** No "the document", "the \
passage", "the above text", "the study", "the trial", "the \
experiment", "the analysis", "the present work", "the report", "the \
article", "this paper", "according to the paper", "according to the \
document", "in the report", "Chunk 1", "Chunk 2", "the first chunk", \
"the second chunk", or any phrase that implies the reader has the \
source in front of them. On research-paper corpora these phrases are \
natural in the chunks but make the question impossible to answer \
without seeing the source — identify the work by intervention, \
population, mechanism, or topic instead.

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

# DIFFICULTY PREFERENCES (P1-P4) — try to satisfy; NEVER refuse for failing them

These are the levers for the hardness goal stated at the top. Try to \
satisfy each one — the more you satisfy, the harder the question. \
Failing a preference makes a question easier, not invalid, so it is \
never by itself grounds to drop a question you are otherwise sure is \
strong and unambiguous. This is distinct from the quality floor: a \
contrived, ambiguous, or self-answering question is not "a simpler \
question", it is a bad one — drop it.

The examples below illustrate the kind of choice each preference \
involves. **They are NOT recipes — match the spirit, not the surface \
form.**

P1. **Anchor the question, hide the answer, and don't stack.** This \
is the core difficulty lever; intuition #2 states the principle, here \
is how to apply it. Distinguish **two entity roles** in every \
question:

  - **The retrieval anchor.** ONE corpus-distinctive named entity, \
dated event, or specific descriptor that drives a vector retriever to \
the cited chunks. It SHOULD be present — strip it ("the cohort", "the \
building", "the protocol") and no RAG can locate the chunks, so you \
contribute noise instead of difficulty. Where natural, PARAPHRASE the \
anchor rather than copying the chunk's lexical surface: "refractory \
hypertension" → "treatment-resistant hypertension" forces the \
embedding model to bridge meaning (the real test of vector \
retrieval), whereas a verbatim phrase rewards BM25 keyword-matching. \
Proper nouns (people, places, named drugs or protocols) are fine \
verbatim — they are the natural anchor regardless of phrasing, and \
forcing a substitution ("Phoenix protocol" → "the 2018 sync \
protocol") is contrived.
  - **The answer entity.** What the question asks ABOUT — whose \
attribute or identity the canonical answer reveals. Describe THIS one \
indirectly, by role or relation, so its name is not lexically present \
in the question; the cited chunks are what identify it. Copying the \
answer entity's distinctive proper nouns into the question leaks the \
answer via lexical match and tests neither retrieval nor reasoning.

  **One indirect descriptor per entity — never stack.** Pick the \
single most indirect way to refer to the answer entity, not three \
attributes chained together. Stacking independent descriptors ("the X \
founded in 1985 by Y in the city of Z" is three) is the most common \
way to leak gold chunks: each attribute is a keyword cluster from a \
gold chunk, so the combination lets a weak retriever find them \
trivially — easier, not harder. Lean indirection beats dense \
attribution. Note the contrast with DEPTH (P2): a CHAIN of \
single-descriptor hops A→B→C is hard; STACKING many descriptors on \
one node is easy. Chain, don't stack.

  If indirect framing would be genuinely awkward or ambiguous, use \
the direct name — don't twist the question. Soft preference, not a \
hard rule.

  Worked contrast (comparing two arms of one trial):
  - good (paraphrased anchor, indirect answer): "How did year-5 \
mortality compare between the two arms of the 412-patient trial of \
treatment-resistant hypertension?"
  - weaker (verbatim anchor — BM25 wins): "…between arms of the \
412-adult refractory-hypertension cohort?"
  - bad (under-anchored — kills retrieval): "…between the cohort's \
arms?"
  - bad (over-stacked — leaks the answer): five descriptors piled on \
the answer entity.

P2. **Chain as deep as the chunks honestly allow.** Multi-step \
reasoning is a primary difficulty lever (intuition #2): when the \
neighborhood supports a 3-hop or 4-hop chain of load-bearing chunks, \
TAKE it over a 2-hop bridge — a longer chain is harder for any \
retriever to assemble, provided every hop still resolves uniquely \
(property 1). For ``inference``, compose facts from distant spans, \
not a single sentence (single-sentence lookups are ``extraction`` \
mislabelled). For ``numeric_single``, combine ≥2 numeric literals \
when available. Only when the chunks support no deeper framing, fall \
back gracefully to a sharp single-hop ``extraction`` or \
``definitional`` — a clean shallow question beats a contrived deep \
one, but never default to shallow when the chunks support depth.

P3. **Prefer non-obvious target attributes.** Don't always pick the \
title, the headline date, or the first sentence — look for \
attributes the chunk states but doesn't foreground.

P4. **Prefer comparisons whose answer isn't obvious from general \
world knowledge.** Birth-before-later-work orderings, comparisons \
based on coincident numbers across unrelated chunks ("both happen to \
be 3" between a glove-test count and a shoulder-implant count), and \
bare year/month subtraction where the answer is mentally obvious — \
these are weaker than comparisons the chunks must resolve. If only \
such a comparison fits, first try another multi-hop type on the same \
chunks (often a stronger ``bridge`` or ``numeric``); if none fits, \
SKIP it rather than emit it — a contrived comparison is filler, and \
filler is worse than nothing (see the quality floor).

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

Example 9 — strong ``inference`` (ELIMINATION to an unstated specific \
answer):
  Neighborhood includes:
    [Chunk 2] "Of the four ridge stations, only the one above the tree \
line ran on solar power; the rest drew from the valley grid."
    [Chunk 5] "Karst Station stands at 2,900 m, above the local tree \
line of 2,200 m; the other three ridge stations all lie below it."
  reasoning_type: "inference"
  cited_chunks:
    - {chunk_id: 2, span: "Of the four ridge stations, only the one \
above the tree line ran on solar power; the rest drew from the valley \
grid."}
    - {chunk_id: 5, span: "Karst Station stands at 2,900 m, above the \
local tree line of 2,200 m; the other three ridge stations all lie \
below it."}
  reasoning: "Chunk 2 says the solar station is the one above the tree \
line; Chunk 5 says Karst is the only ridge station above the tree \
line. No chunk states 'Karst runs on solar' — it is deduced by \
elimination, and dropping either span leaves the solar station \
underdetermined."
  question: "Which ridge station drew its power from the sun?"
  canonical_answer: "Karst Station"

Example 10 — climbing out of the weak (binary) shadow of an inference:
  Neighborhood includes:
    [Chunk 0] "The Halvorsen reform pegged every state pension to the \
market price of grain."
    [Chunk 3] "That winter a blight destroyed the harvest and grain \
prices tripled."
  A composer first drafted: "Did pensions rise or fall that winter?" → \
"rose". That is the coin-flip shadow — a weak RAG hits 'rose' half the \
time, so it barely separates strong from weak configurations. The same \
two chunks support the SAME synthesis with a specific, unguessable \
answer: pensions were pegged to grain (Chunk 0) and grain tripled \
(Chunk 3), so —
  reasoning_type: "inference"
  cited_chunks:
    - {chunk_id: 0, span: "The Halvorsen reform pegged every state \
pension to the market price of grain."}
    - {chunk_id: 3, span: "That winter a blight destroyed the harvest \
and grain prices tripled."}
  reasoning: "Neither chunk states what happened to pensions. Composing \
the peg (Chunk 0) with the tripling (Chunk 3) yields a specific \
magnitude — they tripled — not merely a direction."
  question: "What happened to pension payouts the winter the grain \
blight struck?"
  canonical_answer: "they tripled"
  answer_variants: ["tripled", "increased threefold", "rose to three \
times their level"]

Example 11 — fix a non-unique hop by tightening it, NOT by shortening \
(the deep chain is good; one hop just fails to resolve uniquely):
  A composer drafted: "In which city is the museum founded by the \
collector whose estate funded the chair held by the art historian who \
trained at the academy?" — the final hop, "the art historian who \
trained at the academy", matches many people (the academy trained \
hundreds), so the chain has more than one valid endpoint and the \
answer is not unique. The fix is NOT to delete hops and ask an easy \
single-hop question. Keep the depth and replace the fuzzy hop with a \
uniquely-resolving descriptor the chunks support — e.g. "the art \
historian who first attributed the Vellano frescoes" (the chunks name \
exactly one such person). A deep chain is welcome; every hop in it \
must pin exactly one entity.

Example 12 — DO NOT EMIT (topic-coincidence comparison):
  A composer was tempted by:
    [Chunk A] "An estimated 450 indigenous languages are spoken \
across the continent."
    [Chunk B] "Coastal Tongue is the seventh most widely spoken \
native language, with roughly 230 million speakers."
  The drafted question — "Is the count of the continent's indigenous \
languages greater than the speaker total, in millions, of the seventh \
most widely spoken language?" — pairs two chunks that share only the \
topic word "language". No entity, event, or relationship links them; \
the axis (a raw language count vs a speaker-total-in-millions) is \
meaningless, and a strong RAG answers it no better than a weak one. \
Do NOT emit it. Either find a question built on a GENUINE shared \
entity, or emit nothing for this pairing.

Example 13 — DO NOT EMIT (a question about the collection itself):
  A composer drafted: "How many films that premiered at the 1986 \
festival are described here?" — the answer ("two") is a fact about \
which documents happen to sit in this corpus, not a fact about the \
world. The reader is closed-book and has no notion of "here" or "the \
corpus", so the question is unanswerable for anyone but the composer. \
Never count or characterise the documents themselves (intuition #5); \
ask about the films' content instead.
"""


COMPOSITION_BATCH_USER_PROMPT = """\
Compose the hardest valid questions this neighborhood supports — \
your job is to widen the gap between weak and strong RAG \
configurations. Emit every genuinely strong question the chunks \
support — multi-hop and as deep as they allow; there is no cap on \
strong questions, but do not manufacture weak ones to fill space.

Domain context: {domain_description}

=== Neighborhood (anchor: chunk_id={anchor_chunk_id}) ===

{chunk_blocks}

Reminders:
- Scan the neighborhood for a 3+ load-bearing-span chain BEFORE \
drafting; spans may come from different chunks OR from non-adjacent \
passages within the same chunk. If one exists, take it. Fall back to \
fewer spans only if the chunks don't support more.
- For each question, list the spans you used in ``cited_chunks`` as \
``[{{"chunk_id": <int>, "span": "<verbatim excerpt>"}}]``. Use the \
integer position from the ``[Chunk N]`` labels. Same ``chunk_id`` MAY \
appear twice for intra-chunk multi-hop, only if each entry's ``span`` \
is a different non-overlapping excerpt. Cite ONLY spans the question \
genuinely needs (H1 load-bearing).
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
Otherwise, always emit at least one question — but emit zero questions \
for any individual idea that would be contrived, ambiguous, or a \
near-duplicate of one you already wrote.
"""


ORACLE_OPEN_ENDED_PROMPT = """\
Answer the following question using ONLY the context below. The \
context is known to contain the information needed to determine the \
correct answer.

Expected answer format: {answer_format_hint}

Output only the answer — no explanation, no quotes, no punctuation. \
Give the shortest answer that still fully answers the question; most \
answers are only a few words, rarely more than 15.

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
  "answer": "<the shortest complete answer, rarely more than 15 words, no quotes or punctuation>"
}}

Question: {question}

{spans_block}"""


NAIVE_RAG_PROMPT = """\
Answer the following question. Use only the provided context if any \
was retrieved; otherwise answer to the best of your ability.

Expected answer format: {answer_format_hint}

Output only the answer — no explanation, no quotes, no punctuation. \
Give the shortest answer that still fully answers the question; most \
answers are only a few words, rarely more than 15.

Context:
{context}

Question: {question}

Answer:"""


# Per-(reasoning_type, formula_kind) hint embedded into eval-time prompts so
# the system-under-test knows what shape of answer to produce. Removes a
# known false-negative mode where the model emits "it was larger" when the
# canonical is "13 points larger".
ANSWER_FORMAT_HINTS: dict[tuple[str, str | None], str] = {
    ("extraction", None): "a short factoid: a name, value, or phrase (rarely more than 15 words)",
    ("definitional", None): "a brief definition or description (rarely more than 15 words)",
    ("bridge", None): "a short factoid identifying an entity or attribute (rarely more than 15 words)",
    ("comparison", None): (
        "a comparative phrase such as 'X is larger' or 'Y was earlier', "
        "or a numeric difference with units (rarely more than 15 words)"
    ),
    ("numeric", "arithmetic"): (
        "a numeric value with optional units, e.g. '13', '$50 million', '27 points', "
        "'12 years' (rarely more than 15 words)"
    ),
    ("numeric", None): "a numeric value with optional units (rarely more than 15 words)",
    ("numeric_single", "arithmetic"): (
        "a numeric value with optional units, e.g. '925 adults', '7 months', '28 mmHg' (rarely more than 15 words)"
    ),
    ("numeric_single", None): "a numeric value with optional units (rarely more than 15 words)",
    ("inference", None): (
        "a short phrase, date, or value derived from the chunk but not stated verbatim (rarely more than 15 words)"
    ),
}

_DEFAULT_ANSWER_FORMAT_HINT = (
    "the most concise answer that still fully answers the question — "
    "usually just a name, value, or short phrase (rarely more than 15 words)"
)


def answer_format_hint(reasoning_type: str | None, formula_kind: str | None) -> str:
    """Look up the eval-time format hint for a (reasoning_type, formula_kind) pair."""
    if reasoning_type is None:
        return _DEFAULT_ANSWER_FORMAT_HINT
    return ANSWER_FORMAT_HINTS.get((reasoning_type, formula_kind)) or ANSWER_FORMAT_HINTS.get(
        (reasoning_type, None), _DEFAULT_ANSWER_FORMAT_HINT
    )
