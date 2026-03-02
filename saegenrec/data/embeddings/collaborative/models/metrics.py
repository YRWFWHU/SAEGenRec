"""Recommendation evaluation metrics: Hit Rate@K and NDCG@K.

GPU-accelerated full-ranking evaluation (reference: RecBole / MiniOneRec).
"""

from __future__ import annotations

import torch


def hit_rate_at_k(
    scores: torch.Tensor, targets: torch.Tensor, k: int
) -> float:
    """Compute Hit Rate@K (Recall@K with single ground-truth).

    Args:
        scores: Prediction scores, shape [B, num_items].
        targets: Ground-truth item indices, shape [B].
        k: Top-K cutoff.

    Returns:
        Mean hit rate over the batch.
    """
    rank = _get_target_rank(scores, targets)
    hits = (rank < k).float()
    return hits.mean().item()


def ndcg_at_k(
    scores: torch.Tensor, targets: torch.Tensor, k: int
) -> float:
    """Compute NDCG@K (single ground-truth per query).

    Args:
        scores: Prediction scores, shape [B, num_items].
        targets: Ground-truth item indices, shape [B].
        k: Top-K cutoff.

    Returns:
        Mean NDCG over the batch.
    """
    rank = _get_target_rank(scores, targets)
    mask = (rank < k).float()
    dcg = (1.0 / torch.log2(rank.float() + 2.0)) * mask
    return dcg.mean().item()


def _get_target_rank(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Get the rank (0-indexed) of each target item in descending score order.

    Args:
        scores: Shape [B, num_items].
        targets: Shape [B], values are item indices.

    Returns:
        Tensor of shape [B] with the rank of each target.
    """
    sorted_indices = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)
    target_rank = sorted_indices.gather(1, targets.unsqueeze(1)).squeeze(1)
    return target_rank
