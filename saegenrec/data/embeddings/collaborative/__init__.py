"""Collaborative embedding sub-package."""

from saegenrec.data.embeddings.collaborative.base import (
    COLLABORATIVE_EMBEDDER_REGISTRY,
    CollaborativeEmbedder,
    get_collaborative_embedder,
    register_collaborative_embedder,
)
import saegenrec.data.embeddings.collaborative.sasrec  # noqa: F401

__all__ = [
    "COLLABORATIVE_EMBEDDER_REGISTRY",
    "CollaborativeEmbedder",
    "get_collaborative_embedder",
    "register_collaborative_embedder",
]
