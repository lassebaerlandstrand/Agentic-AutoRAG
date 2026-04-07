# MCQ Exam Generation for RAG Pipeline Evaluation: Methodology and Rationale

## 1. Introduction

Evaluating Retrieval-Augmented Generation (RAG) pipelines requires a scoring mechanism that is fast, deterministic, and sensitive to retrieval quality. Open-ended generation evaluation via LLM-as-judge introduces stochasticity and conflates retrieval with generation quality (Liu et al., 2023; Zheng et al., 2023). Multiple-choice question (MCQ) exams offer deterministic scoring and isolate retrieval effectiveness: a pipeline either retrieves the right context to answer a question, or it does not.

Our exam generation approach draws on three research threads: (1) automated question generation from educational measurement (Anderson & Krathwohl, 2001; Haladyna et al., 2002), (2) LLM-as-examiner evaluation from Guinet et al. (2024), and (3) parametric knowledge detection to prevent test contamination (Oren et al., 2024; Golchin & Surdeanu, 2024). This document describes the current implementation, the reasoning behind each design choice, and the empirical evidence supporting it.

## 2. Document-Level Question Generation

### 2.1 Full Documents, Not Chunks

Prior work on LLM-based exam generation for RAG evaluation (Guinet et al., 2024) generates questions from individual chunks. This creates a systematic bias toward dense passage retrieval: a question derived from a specific chunk shares vocabulary and semantic space with that chunk, making it trivially findable by vector similarity. Graph-based and hybrid retrieval strategies, which surface information through entity relationships rather than lexical overlap, receive no advantage.

Our approach passes the **full parsed document** text to the examiner LLM. This allows the model to:
- Generate questions that span multiple sections (testing multi-hop retrieval)
- Create questions whose vocabulary diverges from the source passage (testing semantic understanding)
- Produce more realistic questions that resemble actual user queries

For documents exceeding the examiner model's context window, we split at configurable thresholds (`doc_split_word_threshold`, default 24,000 words) into overlapping sections. Models with large context windows (e.g., Gemini 3.x with 1M+ tokens) can process documents without splitting.

### 2.2 Realistic Question Design

Questions are designed to resemble what a real user would ask a document AI assistant. This design principle is informed by the UniDoc-Bench benchmark (Feng et al., 2024), which contains real user questions about enterprise documents across healthcare and legal domains.

**Self-containment** is enforced through both prompt instructions and post-hoc regex filtering. Questions must never reference "the document", "the study", "the report", or similar phrases that presuppose the reader has the source material. A set of 18 regex patterns catches violations, and a positive rewriting rule in the system prompt teaches the LLM to rephrase study-referencing questions into direct queries.

## 3. Cognitive Diversity via Bloom's Revised Taxonomy

### 3.1 Theoretical Foundation

Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001) classifies cognitive processes into six levels: Remember, Understand, Apply, Analyze, Evaluate, and Create. The taxonomy is the most widely used framework in educational measurement for ensuring assessment items span multiple cognitive levels (Krathwohl, 2002).

We use five of the six levels (excluding Create, which is not feasible to assess via MCQ) and assign each question a cognitive level. This serves two purposes:

1. **Retrieval diversity.** Lower-order questions (Remember) test whether a pipeline can locate specific facts. Higher-order questions (Analyze, Evaluate) test whether it can retrieve and synthesize information from multiple passages. Different retrieval strategies may excel at different cognitive levels.

2. **Discrimination quality.** IRT-based analysis (Lord, 1980; Guinet et al., 2024) shows that exams with diverse difficulty produce higher Fisher information (measurement precision). Cognitive level diversity naturally creates difficulty diversity.

### 3.2 Weighted Distribution

Our distribution weights higher-order cognitive levels more heavily:

| Level | Target % | Rationale |
|-------|----------|-----------|
| Remember | 10% | Most vulnerable to parametric leaks (isolated facts often in LLM training data) |
| Understand | 20% | Tests interpretation, less leak-prone than pure recall |
| Apply | 25% | Scenario-based questions are naturally self-contained and leak-resistant |
| Analyze | 25% | Multi-piece synthesis requires retrieval of multiple passages |
| Evaluate | 20% | Judgment questions test the quality of retrieved evidence |

The down-weighting of Remember-level questions is empirically motivated: in our runs, Remember questions exhibit the highest parametric leak rate because they target isolated facts that may be common knowledge. Apply and Analyze questions, which embed domain-specific scenarios, are significantly more resistant to parametric leaks because the scenarios are grounded in document-specific details that general-purpose LLMs have not memorized.

Each level includes a specific prompt instruction and example that guides the examiner LLM in producing questions at the correct cognitive depth. The level assignment cycles through a weighted index table (`BLOOM_LEVEL_WEIGHTS`) so each wave of generation produces a deterministic distribution.

## 4. Corpus Diversity via Clustering

### 4.1 Document Clustering

To ensure the exam covers the full breadth of the corpus rather than over-sampling a single topic cluster, we cluster documents before allocating question slots. The approach:

1. **Document embeddings.** Each document is embedded by mean-pooling embeddings of overlapping 1000-token windows. This produces a single vector capturing the document's semantic content.

2. **KMeans clustering.** Documents are clustered into `k = min(sqrt(n_docs), target_size)` clusters. The sqrt heuristic balances granularity against cluster stability.

3. **Square-root proportional allocation.** Question slots are distributed across clusters using Hamilton's method (largest remainder) with square-root weights. This means smaller clusters receive proportionally more questions than their size would suggest, improving coverage of minority topics. This is the same proportional representation method used in apportionment problems (Balinski & Young, 2001).

### 4.2 Per-Document Capacity

When a corpus has fewer documents than the target exam size, multiple questions must be generated from individual documents. We cap the number of questions per document based on its length:

```
capacity = min(word_count // 1500, max_questions_per_doc)
```

A 500-word document supports at most 1 question. A 5000-word document supports up to 3. This prevents the system from attempting to extract multiple distinct questions from short documents where only one or two facts are available.

## 5. Quality Assurance Pipeline

Questions pass through a multi-layer validation pipeline. The pipeline follows the principle of escalating cost: cheap structural checks run first, expensive LLM-based checks run later on the surviving candidates.

### 5.1 Layer 1: Structural Checks (During Generation)

Applied during the generation loop itself, before candidates enter the validation pipeline:

- **JSON parse verification.** The LLM response must be valid JSON with all required fields. Handles markdown fences, trailing commas, and mixed text/JSON.
- **Self-containment filter.** 18 regex patterns catch document-referencing language. A positive rewriting rule in the prompt reduces the failure rate.
- **Source fact contextuality.** The source_fact field must be at least 60 characters and not consist primarily of table headers or list fragments.
- **Option shuffling.** Answer positions are randomized to prevent positional bias (Attali & Bar-Hillel, 2003).

### 5.2 Layer 2: Discriminator Quality Analysis (Guinet et al., 2024)

Following Guinet et al. (2024, Section 3.2), we analyze the quality of incorrect answer options (distractors) using four metrics computed from Jaccard and embedding similarity:

- **Extra-candidate similarity.** If a distractor is more similar to the source document than the correct answer (at Jaccard or embedding level), the distractor may be a rephrased correct answer. Questions exceeding the 95th percentile threshold are removed.
- **Intra-candidate similarity.** If a distractor is too similar to the correct answer itself, it may be a trivial rephrase. Same percentile-based removal.

Thresholds are auto-calibrated at the 95th percentile of each metric across all candidates, targeting approximately 5% removal per metric. This follows the paper's recommendation of adaptive thresholds that match the corpus rather than hard-coded values.

### 5.3 Layer 3: Source Fact Verification (No LLM)

The examiner LLM extracts a `source_fact` field: the passage from the document that answers the question. We verify this is grounded in the actual document using a three-strategy cascade:

1. **Normalized substring match.** Whitespace-normalized source_fact found verbatim in the document. Handles minor formatting differences by stripping pipe characters and collapsing whitespace.

2. **Token overlap match.** For table-derived source_facts where the LLM synthesizes prose from structured data, we compute the fraction of non-stop-word tokens that appear in the source document. A threshold of 70% catches most valid syntheses while rejecting fabrications.

3. **Embedding similarity.** Source_fact and overlapping document windows are embedded, and the maximum cosine similarity is checked against a threshold (default 0.65). For explicitly synthesized source_facts (prefixed with "From the document's data:"), the threshold is relaxed by 0.10.

### 5.4 Layer 4: Parametric Leak Detection (LLM-Based)

A parametric leak occurs when an LLM can answer a question correctly using only its training data, without retrieving any context. Such questions provide zero signal about retrieval quality and waste exam slots.

**Detection method.** Each question is sent to the examiner LLM with no context ("No context available.") across multiple independent trials with varied temperatures (0.3, 0.7, 1.0). If a **majority** of trials answer correctly (threshold: `n_trials // 2 + 1`, e.g., 2 out of 3), the question is flagged as a parametric leak and removed.

**Rationale for majority voting.** The original Guinet et al. design does not include parametric leak detection. Oren et al. (2024) propose detecting benchmark contamination by checking whether LLMs can reproduce test examples, but this targets memorization of specific benchmarks rather than general knowledge overlap. Our approach is more conservative: we test whether the question is answerable from general domain knowledge, not whether the exact question was memorized.

We use majority voting rather than unanimous agreement because unanimous voting (all `n_trials` correct) has a high false negative rate. An LLM that "knows" the answer from training data will answer correctly most of the time but may occasionally select the wrong option due to temperature-induced stochasticity. Majority voting catches these cases.

Variable temperatures across trials further improve detection: low temperature (0.3) reveals confident parametric knowledge, while high temperature (1.0) distinguishes genuine knowledge from lucky guesses.

### 5.5 Layer 5: Oracle Verification (LLM-Based)

The complement of the parametric leak check: we verify that questions are answerable *when given the right context*. The examiner LLM receives the question with a context window centered on the source_fact (default 300 words from the source document) and an escape option ("E: insufficient context").

If the LLM cannot answer correctly even with the source context, the question is structurally broken: ambiguous wording, mismatched distractors, or insufficient source_fact. Such questions are removed.

**Two-pass retry.** If the LLM selects "E" (insufficient context) on the first pass with the windowed context, a second pass retries with the full source document. This catches questions that are valid but whose source_fact is too narrow to be self-explanatory. If the second pass still fails, the question is removed.

## 6. Adaptive Generation with Backfill

The validation pipeline removes a significant fraction of candidates (typically 40-60%). To guarantee hitting the target exam size, we use an adaptive generation loop:

1. **Wave 0.** Generate `exam_size * initial_candidate_multiplier` candidates (default 2.5x).
2. **Validate.** Run the full validation pipeline.
3. **Check.** If `len(validated) >= exam_size`, stop. Otherwise:
4. **Backfill.** Estimate the survival rate from previous waves, compute the deficit per cluster, and generate a targeted backfill wave. Repeat up to `max_backfill_rounds` times.

Backfill waves target under-represented clusters specifically, using per-cluster deficit tracking. This ensures the final exam maintains the desired corpus coverage even when specific topic clusters have higher failure rates.

The entire exam is **frozen** after generation. No modifications occur during the optimization loop. Every trial scores against identical questions, making scores directly comparable across all configurations.

## 7. Multi-Question Document Handling

When a corpus has fewer documents than the target exam size, or when the candidate multiplier requires multiple questions per document, we use a sequential generation strategy:

- **Single-slot documents** (one question per doc) are generated concurrently for maximum throughput.
- **Multi-slot documents** generate questions sequentially within each document, while different documents run concurrently. This prevents the race condition where concurrent calls for the same document don't see each other's results.
- **Source fact avoidance.** Each subsequent question for the same document receives both the question text and source_fact of all previous questions, with the instruction to target a completely different passage.

## 8. Scoring During Optimization

The primary score is simple accuracy on the frozen exam:

```
score = n_correct / n_total
```

Each question produces a `QuestionResult` with the selected answer, correct answer, retrieved context, and generated response. The diagnostic agent analyzes failed questions to identify retrieval failure patterns (e.g., "questions requiring multi-hop reasoning consistently fail") and proposes configuration changes.

## 9. Probe-Based Question Selection

### 9.1 Motivation

When the validation pipeline produces more candidates than `exam_size`, the surplus must be reduced. Random truncation discards potentially discriminating questions. Probe-based selection instead evaluates all candidates against diverse RAG pipeline configurations and retains questions that best differentiate strong from weak setups.

A question that all pipeline configurations answer correctly provides no signal — it is too easy. A question that none can answer is too hard. The most informative questions are those where some configurations succeed and others fail: these are the questions that will actually discriminate between the pipeline configurations explored during optimization.

### 9.2 Probe Configuration Generation

We construct 2–4 probe configurations representing the extremes of the user's search space:

| Probe | Chunk Size | top_k | Embedding | LLM | Reranker |
|-------|-----------|-------|-----------|-----|----------|
| Weak | min | min | weakest | weakest | none |
| Strong | max | max | strongest | strongest | best |
| Balanced | midpoint | midpoint | weakest | weakest | none |
| Cross | max | max | strongest | weakest | best |

**Model ranking.** Weak/strong model selection uses the KnowledgeBase benchmark data: Intelligence Index for LLMs (a pre-computed aggregate of MMLU Pro, GPQA, IFBench), Retrieval score for embedding models, and MTEB-R for rerankers. Only benchmarks available for each model are used in the average (fair comparison — models missing a benchmark are not penalised). When the KnowledgeBase covers fewer than 3 models in a category, a single LLM call to the optimizer model provides the ranking as a fallback.

Probes are deduplicated by structural fingerprint + LLM + reranker, so narrow search spaces (e.g., one LLM, one embedding) produce fewer than 4 unique probes.

### 9.3 Discrimination Scoring

Each validated candidate question is evaluated against every probe configuration. The discrimination score for question $q$ is the variance of the binary response vector:

$$\text{disc}(q) = \text{Var}\left([r_1, r_2, \ldots, r_P]\right)$$

where $r_p \in \{0, 1\}$ indicates whether probe $p$ answered correctly. Maximum variance (0.25 for 2 probes) occurs when exactly half the probes answer correctly — the ideal discriminating question.

### 9.4 Cluster-Aware Selection

The final selection uses a greedy algorithm that maximises total discrimination while preserving cluster diversity:

1. Compute proportional cluster quotas via Hamilton's method (largest remainder).
2. Fill each cluster's quota with its highest-scoring candidates.
3. Fill remaining slots globally from the highest-scoring unused candidates.

This ensures the final exam maintains corpus coverage while preferring discriminating questions within each topic cluster.

### 9.5 Cost Analysis

Probe-based selection adds:
- **Index builds:** 2–4 vector indexes (deduplicated by structural fingerprint, so shared structural configs are built once).
- **Evaluation calls:** $|C| \times P$ LLM calls, where $|C|$ is the number of candidates and $P$ is the number of probes. For typical values ($|C| = 75$, $P = 4$), this is ~300 calls with short MCQ prompts.

The surplus available for selection is controlled by `initial_candidate_multiplier`. Higher values produce more candidates for probes to select from, at the cost of more generation and validation calls.

---

## References

Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives*. Longman.

Attali, Y., & Bar-Hillel, M. (2003). Guess where: The position of correct answers in multiple-choice test items as a psychometric variable. *Journal of Educational Measurement*, 40(2), 109-128.

Balinski, M. L., & Young, H. P. (2001). *Fair representation: Meeting the ideal of one man, one vote* (2nd ed.). Brookings Institution Press.

Feng, Z., et al. (2024). UniDoc-Bench: A unified benchmark for document understanding. *arXiv preprint arXiv:2406.04906*.

Golchin, S., & Surdeanu, M. (2024). Time travel in LLMs: Tracing data contamination in large language models. *ICLR 2024*.

Guinet, G., Pouplin, A., & Perez, V. (2024). Automated evaluation of retrieval-augmented language models with task-specific exam generation. *ICML 2024*. arXiv:2405.13622.

Haladyna, T. M., Downing, S. M., & Rodriguez, M. C. (2002). A review of multiple-choice item-writing guidelines for classroom assessment. *Applied Measurement in Education*, 15(3), 309-333.

Krathwohl, D. R. (2002). A revision of Bloom's taxonomy: An overview. *Theory into Practice*, 41(4), 212-218.

Liu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *EMNLP 2023*.

Lord, F. M. (1980). *Applications of item response theory to practical testing problems*. Lawrence Erlbaum Associates.

Oren, Y., et al. (2024). Proving test set contamination in black box language models. *ICLR 2024*.

Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.
