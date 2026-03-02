"""Generative recommendation modeling package.

Public API:
- ItemTokenizer: Abstract base for item embedding → SID tokenization
- SFTDatasetBuilder: Multi-task SFT data orchestrator
- GenRecModel: Abstract base for generative recommendation models
- SIDTrie: Prefix trie for constrained decoding
"""

from saegenrec.modeling.decoding.trie import SIDTrie
from saegenrec.modeling.genrec.base import GenRecModel
from saegenrec.modeling.sft.builder import SFTDatasetBuilder
from saegenrec.modeling.tokenizers.base import ItemTokenizer

__all__ = [
    "GenRecModel",
    "ItemTokenizer",
    "SFTDatasetBuilder",
    "SIDTrie",
]
