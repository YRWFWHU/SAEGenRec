"""SID prefix trie for constrained decoding."""

from __future__ import annotations

from datasets import Dataset


class SIDTrie:
    """Prefix trie mapping token ID sequences for valid SID paths."""

    def __init__(self) -> None:
        self.root: dict = {}

    def insert(self, token_ids: list[int]) -> None:
        """Insert a sequence of token IDs into the trie."""
        node = self.root
        for tid in token_ids:
            if tid not in node:
                node[tid] = {}
            node = node[tid]

    def search_prefix(self, prefix: list[int]) -> list[int]:
        """Return valid next token IDs given a prefix."""
        node = self.root
        for tid in prefix:
            if tid not in node:
                return []
            node = node[tid]
        return list(node.keys())

    @classmethod
    def from_sid_map(cls, sid_map: Dataset, tokenizer) -> SIDTrie:
        """Build trie from item_sid_map dataset.

        Args:
            sid_map: HF Dataset with ``sid_tokens`` column.
            tokenizer: HuggingFace tokenizer (or compatible) to convert SID
                token strings to IDs.
        """
        trie = cls()
        for row in sid_map:
            token_ids = tokenizer.encode(row["sid_tokens"], add_special_tokens=False)
            trie.insert(token_ids)
        return trie
