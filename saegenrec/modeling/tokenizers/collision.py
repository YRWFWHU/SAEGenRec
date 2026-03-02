"""Collision resolution strategies for Semantic IDs (T012)."""

from __future__ import annotations

from collections import defaultdict

from loguru import logger
import torch


def resolve_collisions(
    codes: torch.Tensor, strategy: str = "append_level", **kwargs
) -> list[list[int]]:
    """Resolve duplicate SID codes across items."""
    if strategy == "append_level":
        return _append_level_resolve(codes)
    elif strategy == "sinkhorn":
        return _sinkhorn_resolve(codes, **kwargs)
    raise ValueError(f"Unknown collision strategy: '{strategy}'")


def _append_level_resolve(codes: torch.Tensor) -> list[list[int]]:
    """For groups sharing the same codes, append an incrementing disambiguation index."""
    code_tuples = [tuple(row.tolist()) for row in codes]
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, ct in enumerate(code_tuples):
        groups[ct].append(idx)

    result: list[list[int] | None] = [None] * len(codes)
    num_collisions = 0
    for ct, indices in groups.items():
        if len(indices) == 1:
            result[indices[0]] = list(ct)
        else:
            num_collisions += len(indices)
            for rank, idx in enumerate(indices):
                result[idx] = list(ct) + [rank]

    if num_collisions:
        logger.info(f"Resolved {num_collisions} collisions via append_level")

    return result  # type: ignore[return-value]


def _sinkhorn_resolve(codes: torch.Tensor, **kwargs) -> list[list[int]]:
    """Sinkhorn resolve — falls back to append_level for post-hoc resolution."""
    logger.info("Sinkhorn resolve: using append_level fallback for post-hoc resolution")
    return _append_level_resolve(codes)


def sinkhorn_knopp(cost: torch.Tensor, max_iter: int = 20, epsilon: float = 0.05) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm: iterative row/col normalisation of exp(-cost/epsilon).

    Returns a doubly-stochastic transport plan.
    """
    M = torch.exp(-cost / epsilon)

    for _ in range(max_iter):
        M = M / M.sum(dim=1, keepdim=True).clamp(min=1e-10)
        M = M / M.sum(dim=0, keepdim=True).clamp(min=1e-10)

    return M
