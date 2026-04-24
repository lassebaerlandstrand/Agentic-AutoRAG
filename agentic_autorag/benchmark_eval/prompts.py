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

Is the system answer correct? An answer is correct if it conveys the same factual \
information as any reference answer, even if phrased differently. Respond with a \
single token: YES or NO."""
