"""Prompt templates for free-form QA generation and LLM-as-judge grading."""

FREE_FORM_ANSWER_PROMPT = """\
Answer the question using only the provided context. If the context is insufficient, \
answer with your best guess in 1-5 words. Respond with only the answer itself: \
no explanation, no quotes, no punctuation.

Context:
{context}

Question: {question}
Answer:"""


JUDGE_PROMPT = """\
You are grading a question-answering system.

Question: {question}
Reference answer(s): {gold}
System answer: {pred}

Pick exactly one verdict and respond with that single token, nothing else:

  YES        — the system answer conveys the same factual information as \
any reference answer (paraphrasing is OK). For numeric answers, values \
that agree to the reference's displayed precision count as YES — e.g. \
"33.33%" matches "33.3%" and "200.0" matches "200", but "30%" does NOT \
match "33.3%" and "210" does NOT match "200". A prediction that \
contains a matching answer alongside additional context still counts as \
YES — judge whether the question was answered correctly, not whether the extra \
context is verifiable. Mark NO only when the prediction directly \
contradicts any part of a reference answer or is plain wrong.
  NO         — the system answer asserts something different from the \
reference answer.
  NO_ANSWER  — the system did not attempt an answer (it said it cannot \
answer, that the context is insufficient, that it doesn't know, or its \
output is otherwise a refusal rather than an attempted factual claim)."""


DIAGNOSIS_JUDGE_PROMPT = """\
A retrieval-augmented system answered a question wrongly. Decide WHERE it \
failed by judging only whether the retrieved context below contains the \
information needed to reach the reference answer.

Question: {question}
Reference answer(s): {gold}

Retrieved context:
{context}

Pick exactly one verdict and respond with that single token, nothing else:

  CONTEXT_INSUFFICIENT  — the retrieved context does NOT contain the \
information needed to reach the reference answer (a retrieval failure: the \
right passages were not retrieved).
  CONTEXT_PRESENT       — the retrieved context DOES contain the information \
needed to reach the reference answer, so answering wrongly is a generation \
failure (the passages were there but the model did not use them correctly)."""
