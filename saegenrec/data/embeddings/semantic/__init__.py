"""Semantic embedding sub-package."""

from saegenrec.data.embeddings.semantic.base import (
    SEMANTIC_EMBEDDER_REGISTRY,
    SemanticEmbedder,
    get_semantic_embedder,
    register_semantic_embedder,
)
import saegenrec.data.embeddings.semantic.sentence_transformer  # noqa: F401

__all__ = [
    "SEMANTIC_EMBEDDER_REGISTRY",
    "SemanticEmbedder",
    "get_semantic_embedder",
    "register_semantic_embedder",
]
