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
match "33.3%" and "210" does NOT match "200".
  NO         — the system answer asserts something different from the \
reference answer.
  NO_ANSWER  — the system did not attempt an answer (it said it cannot \
answer, that the context is insufficient, that it doesn't know, or its \
output is otherwise a refusal rather than an attempted factual claim)."""
