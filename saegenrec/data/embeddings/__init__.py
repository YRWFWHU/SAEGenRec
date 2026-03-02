"""Embedding module — semantic and collaborative sub-systems."""

from saegenrec.data.embeddings.collaborative.base import (
    COLLABORATIVE_EMBEDDER_REGISTRY,
    CollaborativeEmbedder,
    get_collaborative_embedder,
    register_collaborative_embedder,
)
from saegenrec.data.embeddings.semantic.base import (
    SEMANTIC_EMBEDDER_REGISTRY,
    SemanticEmbedder,
    get_semantic_embedder,
    register_semantic_embedder,
)

__all__ = [
    "COLLABORATIVE_EMBEDDER_REGISTRY",
    "CollaborativeEmbedder",
    "SEMANTIC_EMBEDDER_REGISTRY",
    "SemanticEmbedder",
    "get_collaborative_embedder",
    "get_semantic_embedder",
    "register_collaborative_embedder",
    "register_semantic_embedder",
]
