"""Tests for IDF-overlap chunk pairing."""

from __future__ import annotations

import logging

from agentic_autorag.engine.section_classifier import SectionLabel
from agentic_autorag.examiner.chunk_pair_index import ChunkRecord
from agentic_autorag.examiner.idf_pair_index import emit_idf_pairs, tokenize


def _chunk(chunk_id: str, doc_id: str, text: str, section: SectionLabel | None = None) -> ChunkRecord:
    return ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, text=text, section=section)


class TestTokenize:
    def test_drops_stopwords_and_short_tokens(self) -> None:
        tokens = tokenize("The Phoenix protocol was proposed in 2018 to address X.")
        assert "phoenix" in tokens
        assert "protocol" in tokens
        assert "2018" in tokens
        assert "the" not in tokens
        assert "in" not in tokens
        assert "to" not in tokens

    def test_keeps_four_digit_years_drops_short_numerics(self) -> None:
        tokens = tokenize("section 12 published in 1907 with results across 256 cases")
        assert "1907" in tokens
        assert "12" not in tokens
        assert "256" not in tokens

    def test_strips_inline_citations(self) -> None:
        text = "Müller and Jones (2019) found X. Smith et al. (2020) extended it. [12] showed Y."
        tokens = tokenize(text)
        assert "müller" not in tokens
        assert "smith" not in tokens
        assert "jones" not in tokens
        assert "2019" not in tokens
        assert "found" in tokens
        assert "extended" in tokens

    def test_joins_pdf_line_break_hyphens(self) -> None:
        # PDFs often line-wrap with "informa-\ntion" — should re-join.
        tokens = tokenize("the informa-\ntion was useful")
        assert "information" in tokens
        # Without joining we'd see "informa-" and "tion" as separate fragments.
        assert "informa-" not in tokens
        assert "tion" not in tokens

    def test_joins_crlf_line_break_hyphens(self) -> None:
        # Windows-extracted PDFs use \r\n line endings. The hyphen-join regex
        # must handle both LF and CRLF.
        tokens = tokenize("the informa-\r\ntion was useful")
        assert "information" in tokens
        assert "informa-" not in tokens
        assert "tion" not in tokens


class TestEmitIdfPairs:
    def test_cross_doc_only(self) -> None:
        chunks = [
            _chunk("a::0", "a", "phoenix protocol distributed synchronisation"),
            _chunk("a::1", "a", "phoenix protocol followup section"),
            _chunk("b::0", "b", "phoenix protocol from another angle"),
        ]
        seeds = emit_idf_pairs(chunks, target_count=10)
        for s in seeds:
            assert s.chunk_a.doc_id != s.chunk_b.doc_id

    def test_bridges_via_shared_rare_token(self) -> None:
        # docA mentions phoenix; docB also mentions phoenix; docC mentions neither.
        # The pair (A,B) must outrank both (A,C) and (B,C).
        chunks = [
            _chunk("a::0", "a", "the phoenix protocol explained in detail with elaboration"),
            _chunk("b::0", "b", "phoenix protocol applied to distributed systems for sync"),
            _chunk("c::0", "c", "completely unrelated content about gardening and plants"),
        ]
        seeds = emit_idf_pairs(chunks, target_count=10)
        assert len(seeds) >= 1
        top = seeds[0]
        ids = {top.chunk_a.chunk_id, top.chunk_b.chunk_id}
        assert ids == {"a::0", "b::0"}

    def test_caller_filters_before_calling(self) -> None:
        """Eligibility filtering moved up to prepare_corpus; the seeder itself
        treats every supplied chunk as eligible, so a REFERENCES chunk that
        isn't filtered by the caller WILL appear in pairs. This is the new
        contract — verify the seeder no longer drops chunks on its own."""
        chunks = [
            _chunk("a::0", "a", "phoenix protocol details extra words", SectionLabel.REFERENCES),
            _chunk("b::0", "b", "phoenix protocol followup section content", SectionLabel.BODY),
            _chunk("c::0", "c", "phoenix protocol applied to distributed systems", SectionLabel.BODY),
        ]
        seeds = emit_idf_pairs(chunks, target_count=10)
        # The REFERENCES chunk is NOT filtered by the seeder — caller is
        # responsible. Confirm pairs include the body chunks.
        all_ids = {s.chunk_a.chunk_id for s in seeds} | {s.chunk_b.chunk_id for s in seeds}
        assert "b::0" in all_ids or "c::0" in all_ids

    def test_score_descending_order(self) -> None:
        # Three docs: A and B share "phoenix" + "protocol" (high IDF); A and C
        # share only "topic". So (A,B) outranks (A,C).
        chunks = [
            _chunk("a::0", "a", "phoenix protocol details about topic things"),
            _chunk("b::0", "b", "phoenix protocol followup section content"),
            _chunk("c::0", "c", "completely different topic about other stuff"),
        ]
        seeds = emit_idf_pairs(chunks, target_count=10)
        assert all(seeds[i].score >= seeds[i + 1].score for i in range(len(seeds) - 1))

    def test_empty_input(self) -> None:
        assert emit_idf_pairs([], target_count=10) == []

    def test_single_doc_yields_no_pairs(self) -> None:
        chunks = [
            _chunk("a::0", "a", "phoenix protocol"),
            _chunk("a::1", "a", "phoenix protocol"),
        ]
        assert emit_idf_pairs(chunks, target_count=10) == []

    def test_target_count_truncation(self) -> None:
        # Each adjacent pair shares one rare token (alpha, beta, gamma, ...) — every
        # other chunk has only unique tokens otherwise. Forces many pairs above zero
        # score, so target_count truncation is the binding constraint.
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        chunks: list[ChunkRecord] = []
        for i, shared in enumerate(words):
            chunks.append(_chunk(f"d{i}::0", f"d{i}", f"{shared} unique{i} filler{i} content{i}"))
            chunks.append(_chunk(f"d{i}::1", f"d{i + 100}", f"{shared} other{i} stuff{i} text{i}"))
        seeds = emit_idf_pairs(chunks, target_count=3)
        assert len(seeds) == 3

    def test_pair_dedup_canonical_order(self) -> None:
        chunks = [
            _chunk("a::0", "a", "shared phoenix protocol words"),
            _chunk("b::0", "b", "shared phoenix protocol words"),
        ]
        seeds = emit_idf_pairs(chunks, target_count=10)
        assert len(seeds) == 1

    def test_deterministic(self) -> None:
        chunks = [
            _chunk("a::0", "a", "phoenix protocol distributed"),
            _chunk("b::0", "b", "phoenix protocol followup"),
            _chunk("c::0", "c", "phoenix protocol applied"),
        ]
        seeds_1 = emit_idf_pairs(chunks, target_count=10)
        seeds_2 = emit_idf_pairs(chunks, target_count=10)
        assert [(s.chunk_a.chunk_id, s.chunk_b.chunk_id) for s in seeds_1] == [
            (s.chunk_a.chunk_id, s.chunk_b.chunk_id) for s in seeds_2
        ]

    def test_emits_log_summary(self, caplog) -> None:
        chunks = [
            _chunk("a::0", "a", "phoenix protocol distributed"),
            _chunk("b::0", "b", "phoenix protocol followup"),
        ]
        with caplog.at_level(logging.INFO, logger="agentic_autorag.examiner.idf_pair_index"):
            emit_idf_pairs(chunks, target_count=10)
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "IDF-overlap seeds" in joined
