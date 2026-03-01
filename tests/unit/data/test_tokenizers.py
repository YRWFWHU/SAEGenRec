"""Tests for ItemTokenizer and PassthroughTokenizer."""

import pytest

from saegenrec.data.tokenizers.base import TOKENIZER_REGISTRY, get_tokenizer
from saegenrec.data.tokenizers.passthrough import PassthroughTokenizer


class TestTokenizerRegistry:
    def test_passthrough_registered(self):
        assert "passthrough" in TOKENIZER_REGISTRY

    def test_get_tokenizer_valid(self):
        tok = get_tokenizer("passthrough", num_items=100)
        assert isinstance(tok, PassthroughTokenizer)

    def test_get_tokenizer_invalid(self):
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            get_tokenizer("nonexistent")


class TestPassthroughTokenizer:
    def test_tokenize(self):
        tok = PassthroughTokenizer(num_items=100)
        assert tok.tokenize(42) == [42]

    def test_detokenize(self):
        tok = PassthroughTokenizer(num_items=100)
        assert tok.detokenize([42]) == 42

    def test_round_trip(self):
        """tokenize(detokenize(tokens)) == tokens and detokenize(tokenize(id)) == id"""
        tok = PassthroughTokenizer(num_items=100)
        for item_id in [0, 1, 50, 99]:
            tokens = tok.tokenize(item_id)
            assert tok.detokenize(tokens) == item_id
            assert tok.tokenize(tok.detokenize(tokens)) == tokens

    def test_detokenize_wrong_length(self):
        tok = PassthroughTokenizer(num_items=100)
        with pytest.raises(ValueError, match="Expected 1 token"):
            tok.detokenize([1, 2])

    def test_vocab_size(self):
        tok = PassthroughTokenizer(num_items=100)
        assert tok.vocab_size == 100

    def test_token_length(self):
        tok = PassthroughTokenizer(num_items=100)
        assert tok.token_length == 1

    def test_tokenize_batch(self):
        tok = PassthroughTokenizer(num_items=100)
        result = tok.tokenize_batch([0, 5, 99])
        assert result == [[0], [5], [99]]

    def test_token_values_in_range(self):
        """All token values should be in [0, vocab_size)."""
        tok = PassthroughTokenizer(num_items=50)
        for item_id in range(50):
            tokens = tok.tokenize(item_id)
            assert all(0 <= t < tok.vocab_size for t in tokens)

    def test_fixed_token_length(self):
        """All tokenized outputs should have the same length."""
        tok = PassthroughTokenizer(num_items=100)
        for item_id in range(100):
            assert len(tok.tokenize(item_id)) == tok.token_length
