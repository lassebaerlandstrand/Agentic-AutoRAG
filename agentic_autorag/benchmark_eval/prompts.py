"""Prompt templates for free-form QA answering and LLM-as-judge grading.

``ANSWER_PROMPT`` is the single answer template for both scoring paths — the
examiner's exam (which fills ``{answer_format_line}`` with a per-question format
hint) and held-out gold scoring (which leaves it empty). Keeping one template
avoids a train/serve skew between the optimizer's objective and the reported
metric.
"""

ANSWER_PROMPT = """\
Answer the following question using only the provided context. If the context \
does not contain enough information to answer, reply exactly: Insufficient \
information. Otherwise answer directly from the context and do not decline.

{answer_format_line}Output only the answer — no explanation, no quotes, no \
punctuation. Give the shortest answer that still fully answers the question; \
most answers are only a few words, rarely more than 15.

Context:
{context}

Question: {question}

Answer:"""


JUDGE_PROMPT = """\
You are grading a question-answering system. Grade the system answer ONLY \
against the reference answer(s) below — never use outside knowledge, and never \
reward an answer just because it looks plausible.

Question: {question}
Reference answer(s): {gold}
System answer: {pred}

First decide whether the REFERENCE answer is itself a statement that the \
question cannot be answered (for example: "Insufficient information", "cannot \
be determined", "not enough information", "unknown"). Then pick exactly one \
verdict and respond with that single token, nothing else.

If the reference answer IS such a statement (the question is unanswerable), the \
correct behavior is to decline:
  YES        — the system answer also declines: it reports that the \
information is insufficient / cannot be determined, or otherwise does not \
commit to a specific answer.
  NO         — the system answer commits to ANY specific answer (a name, \
entity, value, letter, date, "yes", or "no"). This is NO even if that answer \
looks plausible or you believe it is factually true — the reference says the \
question cannot be answered from the given material. Do not use NO_ANSWER here.

If the reference answer is a real answer (the question is answerable):
  YES        — the system answer conveys the same factual information as any \
reference answer (paraphrasing is OK). For numeric answers, values that agree \
to the reference's displayed precision count as YES — e.g. "33.33%" matches \
"33.3%" and "200.0" matches "200", but "30%" does NOT match "33.3%" and "210" \
does NOT match "200". A prediction that contains a matching answer alongside \
additional context still counts as YES — judge whether the question was \
answered correctly, not whether the extra context is independently verifiable.
  NO         — the system answer asserts something different from, or \
contradicts, the reference answer.
  NO_ANSWER  — the system did not attempt an answer: it said the information \
is insufficient, that it cannot answer, that it doesn't know, or otherwise \
refused. Such a refusal is NEVER equivalent to "yes", "no", or any specific \
reference answer — do not grade it YES. In particular, for a yes/no question, \
"insufficient information" or "cannot be determined" is NOT the same as "no" \
(or "yes"): grade it NO_ANSWER."""


DIAGNOSIS_JUDGE_PROMPT = """\
A retrieval-augmented system answered a question wrongly. Decide WHERE it \
failed.

Question: {question}
Reference answer(s): {gold}

Retrieved context:
{context}

First decide whether the REFERENCE answer is itself a statement that the \
question cannot be answered (for example: "Insufficient information", "cannot \
be determined"). Then pick exactly one verdict and respond with that single \
token, nothing else.

If the reference answer IS such a statement (the question is unanswerable):
  CONTEXT_PRESENT       — always choose this. There is no answer to retrieve, \
so the system failed by asserting an answer instead of declining: a generation \
(over-answering) failure, not a retrieval failure.

If the reference answer is a real answer:
  CONTEXT_INSUFFICIENT  — the retrieved context does NOT contain the \
information needed to reach the reference answer (a retrieval failure: the \
right passages were not retrieved).
  CONTEXT_PRESENT       — the retrieved context DOES contain the information \
needed to reach the reference answer, so answering wrongly is a generation \
failure (the passages were there but the model did not use them correctly)."""
