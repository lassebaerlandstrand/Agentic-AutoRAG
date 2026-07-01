"""Tests for the trial history module."""

from __future__ import annotations

import json

import numpy as np

from agentic_autorag.config.models import (
    IndexType,
    TrialConfig,
)
from agentic_autorag.examiner.evaluator import QuestionResult
from agentic_autorag.optimizer.diagnosis import (
    Diagnosis,
    ProposalMeta,
    TrialMetrics,
)
from agentic_autorag.optimizer.history import HistoryLog, TrialRecord

# Every lever treated as tunable — exercises the renderers without a
# SearchSpace. Production passes ``SearchSpace.tunable_levers()``.
_ALL_TUNABLE: set[str] = {
    "chunking_strategy",
    "chunk_token_size",
    "chunk_token_overlap",
    "embedding_model",
    "index_type",
    "top_k",
    "hybrid_alpha",
    "bm25_vector_fusion",
    "long_context_reorder",
    "passage_compressor",
    "reranker",
    "reranker_top_n",
    "query_expansion",
    "generator_llm",
    "compressor_llm",
    "expander_llm",
    "temperature",
    "reasoning",
    "graph_query_mode",
    "graph_top_k",
}


def _make_config(**overrides) -> TrialConfig:
    defaults = dict(
        chunking_strategy="recursive",
        chunk_token_size=512,
        chunk_token_overlap=64,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_type=IndexType.VECTOR_ONLY,
        top_k=5,
        reranker="none",
        generator_llm="ollama/llama3.2",
        temperature=0.0,
    )
    defaults.update(overrides)
    return TrialConfig(**defaults)


def _make_question_result(qid: str, *, correct: bool) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        correct=correct,
        selected_answer="A" if correct else "B",
        correct_answer="A",
        retrieved_context="some context",
        generated_response="A" if correct else "B",
    )


def _make_trial_metrics(retrieval_complete: float = 0.7) -> TrialMetrics:
    return TrialMetrics(
        answer_accuracy=0.6,
        retrieval_complete=retrieval_complete,
        retrieval_partial=0.2,
        retrieval_miss=0.1,
        refusal_rate=0.05,
        answer_correct_given_complete_retrieval=0.85,
        n_valid=25,
        mean_llm_cost_per_query_usd=0.012,
    )


def _make_diagnosis() -> Diagnosis:
    return Diagnosis(
        trial_metrics=_make_trial_metrics(),
        narrative="retrieval looks weak",
        confirmed_findings=["12 of 20 failures are retrieval_miss"],
    )


def _make_meta() -> ProposalMeta:
    from agentic_autorag.optimizer.diagnosis import Strategy

    return ProposalMeta(
        rationale="diagnoser flagged retrieval primary; widening helps",
        strategy=Strategy(
            phase="ceiling",
            plan="ranging over strong configs; ceiling not yet established; retrieval limits now",
            notes="MiniLM misses span_B",
        ),
    )


def _make_record(
    trial_number: int,
    score: float,
    question_ids: list[str] | None = None,
    *,
    with_structured: bool = True,
) -> TrialRecord:
    if question_ids is None:
        question_ids = ["q1", "q2", "q3"]
    return TrialRecord(
        trial_number=trial_number,
        config=_make_config(),
        answer_accuracy=score,
        question_results=[_make_question_result(qid, correct=(score > 0.5)) for qid in question_ids],
        trial_metrics=_make_trial_metrics() if with_structured else None,
        diagnosis=_make_diagnosis() if with_structured else None,
        meta=_make_meta() if with_structured else None,
    )


class TestTrialRecord:
    def test_to_dict_roundtrip_with_structured(self) -> None:
        record = _make_record(1, 0.8)

        data = record.to_dict()
        restored = TrialRecord.from_dict(data)

        assert restored.trial_number == record.trial_number
        assert restored.answer_accuracy == record.answer_accuracy
        assert restored.trial_metrics is not None
        assert restored.trial_metrics.retrieval_complete == record.trial_metrics.retrieval_complete
        assert restored.diagnosis is not None
        assert restored.diagnosis.narrative == "retrieval looks weak"
        assert restored.diagnosis.confirmed_findings == ["12 of 20 failures are retrieval_miss"]
        assert restored.meta is not None
        assert restored.meta.rationale == "diagnoser flagged retrieval primary; widening helps"
        assert restored.meta.strategy is not None
        assert restored.meta.strategy.phase == "ceiling"

    def test_to_dict_roundtrip_without_structured(self) -> None:
        record = _make_record(1, 0.5, with_structured=False)

        data = record.to_dict()
        restored = TrialRecord.from_dict(data)

        assert restored.trial_metrics is None
        assert restored.diagnosis is None
        assert restored.meta is None

    def test_to_dict_is_json_serializable(self) -> None:
        record = _make_record(1, 0.5)

        json.dumps(record.to_dict())


class TestHistoryLog:
    def test_empty_log(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))

        assert log.records == []
        assert log.get_best() is None

    def test_add_and_get_best(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))

        log.add(_make_record(1, 0.5))
        log.add(_make_record(2, 0.8))
        log.add(_make_record(3, 0.6))
        best = log.get_best()

        assert len(log.records) == 3
        assert best is not None
        assert best.trial_number == 2
        assert best.answer_accuracy == 0.8

    def test_persistence_preserves_structured_fields(self, tmp_path) -> None:
        path = str(tmp_path / "history.jsonl")
        log1 = HistoryLog(path=path)
        log1.add(_make_record(1, 0.5))
        log1.add(_make_record(2, 0.9))

        log2 = HistoryLog(path=path)

        assert len(log2.records) == 2
        assert log2.records[0].trial_number == 1
        assert log2.records[1].answer_accuracy == 0.9
        assert log2.records[1].diagnosis is not None
        assert log2.records[1].meta is not None

    def test_add_strips_large_fields_from_memory(self, tmp_path) -> None:
        path = str(tmp_path / "history.jsonl")
        log = HistoryLog(path=path)
        log.add(_make_record(1, 0.8))

        qr = log.records[0].question_results[0]
        assert qr.retrieved_context == ""
        assert qr.generated_response == ""
        assert qr.question_id == "q1"
        assert qr.correct is True

        reloaded = HistoryLog(path=path)
        qr_disk = reloaded.records[0].question_results[0]
        assert qr_disk.retrieved_context == "some context"

    def test_format_for_agent_empty(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))

        result = log.format_for_agent(tunable=_ALL_TUNABLE)

        assert result == "No previous trials."

    def test_format_for_agent_includes_full_trial_block_and_phase(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        # Two trials with different configs so the mechanical "changes vs prior"
        # line has something to render on trial 2.
        log.add(_make_record(1, 0.5))
        rec2 = _make_record(2, 0.6)
        rec2.config = rec2.config.model_copy(update={"embedding_model": "BAAI/bge-m3", "top_k": 10})
        log.add(rec2)

        text = log.format_for_agent(tunable=_ALL_TUNABLE)

        # Header + score/cost line
        assert "Trial 1" in text
        assert "Trial 2" in text
        assert "accuracy=0.600" in text
        assert "cost=$" in text
        # Verdict breakdown
        assert "verdicts: EM=" in text
        assert "judge_yes=" in text
        assert "judge_no_answer=" in text
        # Retrieval rates. The old `quality:` line (off-objective mean_em/mean_f1
        # + an undefined retrieval_quality scalar) is no longer rendered.
        assert "quality:" not in text
        assert "retrieval rates: complete=" in text
        # Config block renders the tunable levers applicable to this vector-only,
        # no-reranker, no-compressor, no-expansion trial.
        assert "config (tunable levers):" in text
        for field_name in (
            "index_type",
            "embedding_model",
            "chunking_strategy",
            "chunk_token_size",
            "chunk_token_overlap",
            "top_k",
            "reranker",
            "query_expansion",
            "generator_llm",
            "reasoning",
        ):
            assert field_name in text, f"missing applicable lever {field_name} in rendered block"
        # Levers moot for this trial's structural choices are omitted (not n/a'd).
        for field_name in (
            "graph_query_mode",
            "graph_top_k",
            "hybrid_alpha",
            "bm25_vector_fusion",
            "reranker_top_n",
        ):
            assert field_name not in text, f"inapplicable lever {field_name} should be omitted"
        # Mechanical diff between trial 1 and trial 2 configs.
        assert "changes vs prior:" in text
        assert "embedding_model:" in text and "BAAI/bge-m3" in text
        assert "top_k:" in text
        assert "rationale:" in text
        # The per-trial block surfaces the campaign phase; the full plan/notes
        # live in the state card's plan carry-over, not in the history dump.
        assert "phase: ceiling" in text

    def test_format_for_agent_drops_fixed_and_derived_levers(self, tmp_path) -> None:
        # The Proposer view shows ONLY the run's tunable levers — fixed
        # (temperature) and derived (expander_llm) values are constant or
        # auto-resolved, so they belong in the search-space prompt, not in
        # every trial block. Inapplicable levers (hybrid_alpha under rrf) drop too.
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        rec = _make_record(1, 0.6)
        rec.config = rec.config.model_copy(
            update={
                "index_type": IndexType.HYBRID_BM25_VECTOR,
                "bm25_vector_fusion": "rrf",
                "query_expansion": "hyde",
                "expander_llm": "azure/gpt-4o-mini",
            }
        )
        log.add(rec)
        tunable = {
            "chunk_token_size",
            "embedding_model",
            "index_type",
            "top_k",
            "bm25_vector_fusion",
            "query_expansion",
            "generator_llm",
        }

        text = log.format_for_agent(tunable=tunable)

        # Tunable + applicable levers render...
        assert "chunk_token_size=" in text
        assert "bm25_vector_fusion=rrf" in text
        assert "query_expansion=hyde" in text
        # ...fixed / derived / not-searched levers do not.
        assert "temperature=" not in text
        assert "expander_llm=" not in text
        assert "reasoning=" not in text
        assert "chunk_token_overlap=" not in text
        # hybrid_alpha is both untuned and inapplicable under rrf fusion.
        assert "hybrid_alpha=" not in text

    def test_format_for_agent_appends_current_trial_preview(self, tmp_path) -> None:
        # The orchestrator persists the just-completed trial to history AFTER
        # the Proposer runs, so during proposal the current trial is passed as
        # a preview record. format_for_agent must render it as the last block.
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.5))
        preview = _make_record(2, 0.82)
        preview.meta = None  # proposer hasn't emitted meta for the current trial yet

        text = log.format_for_agent(tunable=_ALL_TUNABLE, current_trial=preview)

        assert "Trial 1" in text
        assert "Trial 2" in text
        # The preview's score should appear and trial 2 should carry the best-score tag.
        assert "accuracy=0.820" in text
        assert "★best accuracy" in text
        # No persisted records were mutated.
        assert len(log.records) == 1

    def test_format_for_agent_empty_history_with_current_trial(self, tmp_path) -> None:
        # First trial of a run: no prior history, but the Proposer still needs
        # the current trial's block (and a "No previous trials" sentinel
        # would lie to it).
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        preview = _make_record(1, 0.6)
        preview.meta = None

        text = log.format_for_agent(tunable=_ALL_TUNABLE, current_trial=preview)

        assert text != "No previous trials."
        assert "Trial 1" in text
        assert "accuracy=0.600" in text

    def test_format_for_diagnoser_is_slim_and_cost_free(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        record = _make_record(1, 0.6)
        record.cross_tab_snapshot = "retrieval_miss × bridge(n=2): 4"
        log.add(record)

        text = log.format_for_diagnoser()

        # One correctness line per trial (accuracy + retrieval rates).
        assert "trial 1: acc=0.600" in text
        assert "retrieval complete=" in text
        assert "acc_given_complete=" in text
        # Recent cross-tab snapshot is surfaced for failure-mode migration.
        assert "Recent failure-mode cross-tabs" in text
        assert "retrieval_miss × bridge(n=2): 4" in text
        # No cost, no per-trial config block, no configs-already-tried index,
        # no Proposer-side fields — the Diagnoser is objective-agnostic.
        assert "$" not in text
        assert "cost" not in text
        assert "config:" not in text
        assert "Configs already tried" not in text
        assert "rationale:" not in text
        assert "phase:" not in text

    def test_format_for_diagnoser_empty(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        assert log.format_for_diagnoser() == "No previous trials."

    def test_format_for_agent_marks_pareto_and_best(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        # Two trials: trial 1 cheap+ok, trial 2 expensive+best. After Pareto
        # recomputation both should be on the frontier; trial 2 is the best
        # score. The knee marker was dropped — the optimizer loop uses
        # best-score as the universal anchor instead.
        rec1 = _make_record(1, 0.6)
        rec1.mean_llm_cost_per_query_usd = 0.001
        rec2 = _make_record(2, 0.9)
        rec2.mean_llm_cost_per_query_usd = 0.05
        log.add(rec1)
        log.add(rec2)
        log.recompute_pareto_flags()

        text = log.format_for_agent(tunable=_ALL_TUNABLE)

        assert "★on Pareto frontier" in text
        assert "★best accuracy" in text
        assert "(knee)" not in text

    def test_format_for_agent_score_only_drops_cost_and_pareto_tag(self, tmp_path) -> None:
        # show_cost=False (score-only Proposer): cost/token columns and the
        # Pareto-frontier tag are suppressed; accuracy and the best tag stay.
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        rec1 = _make_record(1, 0.6)
        rec1.mean_llm_cost_per_query_usd = 0.001
        rec2 = _make_record(2, 0.9)
        rec2.mean_llm_cost_per_query_usd = 0.05
        log.add(rec1)
        log.add(rec2)
        log.recompute_pareto_flags()

        text = log.format_for_agent(tunable=_ALL_TUNABLE, show_cost=False)

        assert "accuracy=0.900" in text
        assert "cost=$" not in text
        assert "in_tok=" not in text
        assert "★on Pareto frontier" not in text
        # Best-accuracy tag is not a cost concept — it stays.
        assert "★best accuracy" in text

        # The default (cost-aware) view still shows cost + the Pareto tag.
        text_cost = log.format_for_agent(tunable=_ALL_TUNABLE, show_cost=True)
        assert "cost=$" in text_cost
        assert "★on Pareto frontier" in text_cost

    def test_format_for_agent_tiers_old_trials_to_index(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        # 15 trials: the first 7 are flat-and-mediocre (not best, not movers, no
        # Pareto recompute), the last 8 improve so trial 15 is best. With the
        # default recent window of 8, trials 8-15 stay full-detail and 1-7 must
        # collapse to the configs-tried index.
        for n in range(1, 16):
            score = 0.30 if n <= 7 else 0.30 + 0.05 * (n - 7)
            log.add(
                TrialRecord(
                    trial_number=n,
                    config=_make_config(top_k=n),
                    answer_accuracy=score,
                    question_results=[_make_question_result("q1", correct=True)],
                    trial_metrics=_make_trial_metrics(),
                    meta=_make_meta(),
                )
            )

        text = log.format_for_agent(tunable=_ALL_TUNABLE)

        # Recent window renders as full trial blocks.
        assert "### Trial 15" in text
        assert "### Trial 8" in text
        # Old trials outside the keep-set get no full block...
        assert "### Trial 1\n" not in text
        assert "### Trial 5\n" not in text
        # ...but every trial appears in the complete configs-tried index.
        assert "Configs already tried" in text
        for n in range(1, 16):
            assert f"trial {n} (acc=" in text

    def test_config_signature_uses_same_vocabulary_as_block(self) -> None:
        from agentic_autorag.optimizer.history import _config_lines, _config_signature

        # The index line is exactly the block's config lines flattened to one
        # line — same canonical key=value vocabulary, no alias dialect.
        cfg = _make_config(chunk_token_overlap=64)
        sig = _config_signature(cfg, _ALL_TUNABLE)
        # Same key=value tokens in the same order; only whitespace layout differs.
        block_tokens = " ".join(_config_lines(cfg, _ALL_TUNABLE)).split()

        assert sig.split() == block_tokens
        assert "chunk_token_size=512" in sig
        assert "chunk_token_overlap=64" in sig
        # No legacy aliases / presence-flags.
        for legacy in ("chunk=512/64", "embed=", "qexp=", "fusion=", "reorder=on", "reasoning=on", "llm=gen"):
            assert legacy not in sig

    def test_config_signature_distinguishes_levers_the_summary_omits(self) -> None:
        from agentic_autorag.optimizer.history import _config_signature

        # chunk_token_overlap is absent from the one-line summary() but must
        # distinguish configs in the no-repeat index.
        a = _make_config(chunk_token_overlap=0)
        b = _make_config(chunk_token_overlap=64)

        assert _config_signature(a, _ALL_TUNABLE) != _config_signature(b, _ALL_TUNABLE)

    def test_config_signature_omits_inapplicable_reranker_top_n(self) -> None:
        from agentic_autorag.optimizer.history import _config_signature

        # reranker_top_n is moot when no reranker runs — the signature omits it
        # (matching the full block), so configs differing only in a dead lever
        # share a signature rather than masquerading as distinct.
        a = _make_config(reranker="none", reranker_top_n=3)
        b = _make_config(reranker="none", reranker_top_n=9)

        sig_a = _config_signature(a, _ALL_TUNABLE)
        assert sig_a == _config_signature(b, _ALL_TUNABLE)
        assert "reranker=none" in sig_a
        assert "reranker_top_n=" not in sig_a

    def test_config_signature_respects_tunable_set(self) -> None:
        from agentic_autorag.optimizer.history import _config_signature

        # A pinned lever (not in the tunable set) is dropped from the signature.
        cfg = _make_config(top_k=7)
        full = _config_signature(cfg, _ALL_TUNABLE)
        without_top_k = _config_signature(cfg, _ALL_TUNABLE - {"top_k"})

        assert "top_k=7" in full
        assert "top_k=" not in without_top_k

    def test_get_response_matrix_none_for_few_trials(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        assert log.get_response_matrix() is None

        log.add(_make_record(1, 0.5))
        assert log.get_response_matrix() is None

    def test_get_response_matrix_shape(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2", "q3"]))
        log.add(_make_record(2, 0.4, question_ids=["q1", "q2", "q3"]))

        matrix = log.get_response_matrix()

        assert matrix is not None
        assert matrix.shape == (2, 3)

    def test_get_response_matrix_values(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2"]))
        log.add(_make_record(2, 0.3, question_ids=["q1", "q2"]))

        matrix = log.get_response_matrix()

        assert matrix is not None
        np.testing.assert_array_equal(matrix[0], [1, 1])
        np.testing.assert_array_equal(matrix[1], [0, 0])

    def test_get_response_matrix_for_exam_filters_columns(self, tmp_path) -> None:
        log = HistoryLog(path=str(tmp_path / "history.jsonl"))
        log.add(_make_record(1, 0.8, question_ids=["q1", "q2", "q3"]))
        log.add(_make_record(2, 0.3, question_ids=["q2", "q3", "q4"]))

        matrix = log.get_response_matrix_for_exam({"q2", "q4"})

        assert matrix is not None
        assert matrix.shape == (2, 2)
        np.testing.assert_array_equal(matrix[0], [1, 0])
        np.testing.assert_array_equal(matrix[1], [0, 0])
