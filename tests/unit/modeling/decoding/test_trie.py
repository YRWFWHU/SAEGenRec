"""Tests for SIDTrie (T032)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from saegenrec.modeling.decoding.trie import SIDTrie


class TestSIDTrieInsertAndSearch:
    def test_insert_single_sequence(self):
        trie = SIDTrie()
        trie.insert([10, 20, 30])
        assert trie.search_prefix([]) == [10]
        assert trie.search_prefix([10]) == [20]
        assert trie.search_prefix([10, 20]) == [30]

    def test_insert_multiple_sequences_shared_prefix(self):
        trie = SIDTrie()
        trie.insert([10, 20, 30])
        trie.insert([10, 20, 40])
        trie.insert([10, 50, 60])

        assert sorted(trie.search_prefix([])) == [10]
        assert sorted(trie.search_prefix([10])) == [20, 50]
        assert sorted(trie.search_prefix([10, 20])) == [30, 40]
        assert trie.search_prefix([10, 50]) == [60]

    def test_insert_disjoint_sequences(self):
        trie = SIDTrie()
        trie.insert([1, 2, 3])
        trie.insert([4, 5, 6])
        assert sorted(trie.search_prefix([])) == [1, 4]
        assert trie.search_prefix([1]) == [2]
        assert trie.search_prefix([4]) == [5]

    def test_search_prefix_complete_sequence_returns_empty(self):
        trie = SIDTrie()
        trie.insert([10, 20, 30])
        assert trie.search_prefix([10, 20, 30]) == []

    def test_search_prefix_invalid_prefix_returns_empty(self):
        trie = SIDTrie()
        trie.insert([10, 20, 30])
        assert trie.search_prefix([99]) == []
        assert trie.search_prefix([10, 99]) == []


class TestSIDTrieEmpty:
    def test_empty_trie_search_returns_empty(self):
        trie = SIDTrie()
        assert trie.search_prefix([]) == []
        assert trie.search_prefix([1, 2, 3]) == []


class TestSIDTrieFromSidMap:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()

        token_map = {
            "<s_a_0><s_b_1><s_c_2><s_d_3>": [100, 101, 102, 103],
            "<s_a_0><s_b_1><s_c_2><s_d_4>": [100, 101, 102, 104],
            "<s_a_0><s_b_2><s_c_3><s_d_1>": [100, 105, 106, 101],
            "<s_a_1><s_b_0><s_c_2><s_d_3>": [107, 108, 102, 103],
            "<s_a_1><s_b_1><s_c_0><s_d_2>": [107, 101, 109, 110],
        }
        tokenizer.encode = lambda text, add_special_tokens=True: token_map[text]
        return tokenizer

    def test_from_sid_map_builds_correct_trie(self, mock_sid_map, mock_tokenizer):
        trie = SIDTrie.from_sid_map(mock_sid_map, mock_tokenizer)

        assert sorted(trie.search_prefix([])) == [100, 107]
        assert sorted(trie.search_prefix([100])) == [101, 105]
        assert sorted(trie.search_prefix([100, 101, 102])) == [103, 104]

    def test_from_sid_map_complete_paths(self, mock_sid_map, mock_tokenizer):
        trie = SIDTrie.from_sid_map(mock_sid_map, mock_tokenizer)

        assert trie.search_prefix([100, 101, 102, 103]) == []
        assert trie.search_prefix([107, 108, 102, 103]) == []

    def test_from_sid_map_calls_tokenizer_correctly(self, mock_sid_map, mock_tokenizer):
        token_map = {
            "<s_a_0><s_b_1><s_c_2><s_d_3>": [100, 101, 102, 103],
            "<s_a_0><s_b_1><s_c_2><s_d_4>": [100, 101, 102, 104],
            "<s_a_0><s_b_2><s_c_3><s_d_1>": [100, 105, 106, 101],
            "<s_a_1><s_b_0><s_c_2><s_d_3>": [107, 108, 102, 103],
            "<s_a_1><s_b_1><s_c_0><s_d_2>": [107, 101, 109, 110],
        }
        tracked_tokenizer = MagicMock()
        tracked_tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=True: token_map[text]
        )

        SIDTrie.from_sid_map(mock_sid_map, tracked_tokenizer)

        assert tracked_tokenizer.encode.call_count == 5
        for call in tracked_tokenizer.encode.call_args_list:
            assert call.kwargs.get("add_special_tokens") is False
