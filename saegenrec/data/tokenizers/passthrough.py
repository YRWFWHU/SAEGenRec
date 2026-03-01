"""PassthroughTokenizer: identity tokenizer that returns item ID as a single token."""

from __future__ import annotations

from saegenrec.data.tokenizers.base import ItemTokenizer, register_tokenizer


@register_tokenizer("passthrough")
class PassthroughTokenizer(ItemTokenizer):
    """Passthrough tokenizer — item integer ID is used directly as a single token."""

    def __init__(self, num_items: int):
        self._num_items = num_items

    def tokenize(self, item_id: int) -> list[int]:
        return [item_id]

    def detokenize(self, tokens: list[int]) -> int:
        if len(tokens) != 1:
            raise ValueError(f"Expected 1 token, got {len(tokens)}")
        return tokens[0]

    @property
    def vocab_size(self) -> int:
        return self._num_items

    @property
    def token_length(self) -> int:
        return 1
