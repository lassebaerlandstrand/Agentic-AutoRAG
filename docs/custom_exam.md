# Bring your own exam

By default Agentic AutoRAG generates a synthetic exam from your corpus and
optimizes against it. If you already have questions with known answers — a
hand-written set, an exported evaluation set, or a labelled QA dataset — you can
optimize against **those instead**: point the examiner at a JSON file and skip
generation entirely.

```yaml
examiner:
  custom_exam_path: path/to/exam.json
```

When `custom_exam_path` is set, `optimize` loads your file, skips corpus
composition / probe selection, and runs the whole reasoning loop against your
questions. Nothing is dropped — every question you provide is evaluated.

## Exam format

The file is a **JSON list of question records**. Each record has, at minimum, a
unique `id`, the `question`, and its `canonical_answer`. How much *evidence* you
attach to a question defines its **grounding tier**, and the tier decides how
much diagnostic detail the optimizer can give you.

| Tier | You provide | The optimizer can score |
| ---- | ----------- | ----------------------- |
| **A** | question + answer | answer accuracy; a judge attributes each wrong answer to retrieval vs. generation |
| **B** | + `supporting_doc_ids` | above **+ document-level retrieval** (did the right documents come back?) |
| **C** | + verbatim evidence `source_spans` | above **+ span-level retrieval** (did the exact evidence come back?) |

You can mix tiers in one file. Fields:

- `id` *(str, required)* — unique per question.
- `question` *(str, required)*.
- `canonical_answer` *(str, required)* — the primary correct answer.
- `answer_variants` *(list[str], optional)* — other acceptable surface forms.
- `supporting_doc_ids` *(list[str])* — ids of the corpus documents that support
  the answer (**tier B**). An id is a corpus filename without its extension.
- `reasoning_type` *(str, optional)* — one of `extraction`, `definitional`,
  `inference`, `bridge`, `comparison`, `numeric_single`, `numeric`. Used only to
  tailor the answer-format hint and group the diagnosis; safe to omit. If omitted,
  the model is asked for the most concise answer that still fully answers the
  question — usually just a name, value, or short phrase (rarely more than 15 words).

### Tier A — question + answer

```json
[
  { "id": "q1", "question": "In what year did Apollo 11 land humans on the Moon?", "canonical_answer": "1969" }
]
```

### Tier B — add the supporting documents

```json
[
  {
    "id": "q2",
    "question": "What is the capital of the country that hosted the 2016 Summer Olympics?",
    "canonical_answer": "Brasília",
    "answer_variants": ["Brasilia"],
    "supporting_doc_ids": ["summer_olympics_2016", "brazil"]
  }
]
```

### Tier C — add verbatim evidence spans

Tier C adds two **aligned, parallel** lists — one entry per supporting
document: `source_doc_ids` (which document each piece of evidence comes from) and
`source_spans` (a verbatim substring copied from that document).

```json
[
  {
    "id": "q2",
    "question": "What is the capital of the country that hosted the 2016 Summer Olympics?",
    "canonical_answer": "Brasília",
    "answer_variants": ["Brasilia"],
    "source_doc_ids": ["summer_olympics_2016", "brazil"],
    "source_spans": [
      "The 2016 Summer Olympics were held in Rio de Janeiro, Brazil.",
      "The capital of Brazil is Brasília, inaugurated in 1960."
    ]
  }
]
```

You rarely need to hand-write tier C — see below.

## Upgrade a tier-B exam to tier C automatically

If your questions already name their supporting documents (tier B), let an LLM
extract the evidence spans for you and verify each one is actually present in its
document:

```bash
uv run agentic-autorag ground-exam \
  --config configs/my_project.yaml \
  --exam my_exam_tierB.json \
  --output my_exam_tierC.json
```

The command reads the documents from your config's `meta.corpus_path` and uses
its examiner model (override with `--extractor-model`). Every question whose
spans verify is upgraded to tier C; any that can't be verified is kept unchanged
as tier B — nothing is dropped. Then point `custom_exam_path` at the output.
