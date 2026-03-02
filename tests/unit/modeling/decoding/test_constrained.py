"""Tests for SIDConstrainedLogitsProcessor (T033)."""

from __future__ import annotations

import torch

from saegenrec.modeling.decoding.constrained import SIDConstrainedLogitsProcessor
from saegenrec.modeling.decoding.trie import SIDTrie

VOCAB_SIZE = 20
SID_BEGIN = 15
SID_END = 16


def _make_trie() -> SIDTrie:
    trie = SIDTrie()
    trie.insert([1, 2, 3])
    trie.insert([1, 2, 4])
    trie.insert([1, 5, 6])
    return trie


def _make_processor(trie: SIDTrie | None = None) -> SIDConstrainedLogitsProcessor:
    return SIDConstrainedLogitsProcessor(
        trie=trie or _make_trie(),
        sid_begin_token_id=SID_BEGIN,
        sid_end_token_id=SID_END,
    )


class TestValidTokensPreserved:
    def test_valid_next_tokens_keep_original_scores(self):
        processor = _make_processor()
        input_ids = torch.tensor([[SID_BEGIN]])
        scores = torch.zeros(1, VOCAB_SIZE)
        scores[0, 1] = 5.0

        result = processor(input_ids, scores)

        assert result[0, 1].item() == 5.0

    def test_after_partial_prefix_valid_tokens_preserved(self):
        processor = _make_processor()
        input_ids = torch.tensor([[SID_BEGIN, 1]])
        scores = torch.zeros(1, VOCAB_SIZE)
        scores[0, 2] = 3.0
        scores[0, 5] = 4.0

        result = processor(input_ids, scores)

        assert result[0, 2].item() == 3.0
        assert result[0, 5].item() == 4.0


class TestInvalidTokensMasked:
    def test_invalid_tokens_set_to_neg_inf(self):
        processor = _make_processor()
        input_ids = torch.tensor([[SID_BEGIN]])
        scores = torch.ones(1, VOCAB_SIZE)

        result = processor(input_ids, scores)

        assert result[0, 1].item() == 1.0
        assert result[0, 0].item() == float("-inf")
        assert result[0, 7].item() == float("-inf")

    def test_after_prefix_only_valid_continuations_allowed(self):
        processor = _make_processor()
        input_ids = torch.tensor([[SID_BEGIN, 1, 2]])
        scores = torch.ones(1, VOCAB_SIZE)

        result = processor(input_ids, scores)

        assert result[0, 3].item() == 1.0
        assert result[0, 4].item() == 1.0
        for t in [0, 1, 2, 5, 6, 7, 8, 9, 10]:
            assert result[0, t].item() == float("-inf")


class TestBatchSupport:
    def test_batch_sequences_processed_independently(self):
        processor = _make_processor()
        # Pad to same length (0 is a padding token, not in trie root)
        input_ids = torch.tensor([
            [0, SID_BEGIN, 1],
            [SID_BEGIN, 1, 2],
        ])
        scores = torch.ones(2, VOCAB_SIZE)

        result = processor(input_ids, scores)

        # Seq 0: prefix=[1] → valid next: [2, 5]
        assert result[0, 2].item() == 1.0
        assert result[0, 5].item() == 1.0
        assert result[0, 0].item() == float("-inf")

        # Seq 1: prefix=[1, 2] → valid next: [3, 4]
        assert result[1, 3].item() == 1.0
        assert result[1, 4].item() == 1.0
        assert result[1, 0].item() == float("-inf")


class TestSidBeginEndTokens:
    def test_no_sid_begin_allows_root_tokens(self):
        """When no sid_begin_token is found, processor uses empty prefix."""
        processor = _make_processor()
        input_ids = torch.tensor([[99, 98]])
        scores = torch.ones(1, VOCAB_SIZE)

        result = processor(input_ids, scores)

        assert result[0, 1].item() == 1.0
        assert result[0, 0].item() == float("-inf")

    def test_extract_sid_prefix_after_begin_token(self):
        processor = _make_processor()
        seq = [99, 98, SID_BEGIN, 1, 2]
        prefix = processor._extract_sid_prefix(seq)
        assert prefix == [1, 2]

    def test_extract_sid_prefix_uses_last_begin(self):
        processor = _make_processor()
        seq = [SID_BEGIN, 7, 8, SID_BEGIN, 1]
        prefix = processor._extract_sid_prefix(seq)
        assert prefix == [1]

    def test_extract_sid_prefix_no_begin_returns_empty(self):
        processor = _make_processor()
        prefix = processor._extract_sid_prefix([99, 98, 97])
        assert prefix == []
