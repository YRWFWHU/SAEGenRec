"""Constrained logits processor for SID-guided decoding."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from saegenrec.modeling.decoding.trie import SIDTrie


class SIDConstrainedLogitsProcessor(LogitsProcessor):
    """Constrain LLM generation to valid SID token sequences."""

    def __init__(
        self,
        trie: SIDTrie,
        sid_begin_token_id: int,
        sid_end_token_id: int,
    ) -> None:
        self.trie = trie
        self.sid_begin_token_id = sid_begin_token_id
        self.sid_end_token_id = sid_end_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()
            sid_prefix = self._extract_sid_prefix(seq)
            valid_tokens = self.trie.search_prefix(sid_prefix)

            if valid_tokens:
                mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
                for token_id in valid_tokens:
                    mask[token_id] = False
                scores[i][mask] = float("-inf")
            elif not sid_prefix:
                root_tokens = self.trie.search_prefix([])
                if root_tokens:
                    mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
                    for token_id in root_tokens:
                        mask[token_id] = False
                    scores[i][mask] = float("-inf")

        return scores

    def _extract_sid_prefix(self, seq: list[int]) -> list[int]:
        """Extract SID tokens from the sequence (after last sid_begin_token)."""
        try:
            last_begin = len(seq) - 1 - seq[::-1].index(self.sid_begin_token_id)
            return seq[last_begin + 1 :]
        except ValueError:
            return []
