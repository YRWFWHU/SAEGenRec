"""Tests for collision resolution (T008)."""

from __future__ import annotations

import torch

from saegenrec.modeling.tokenizers.collision import resolve_collisions, sinkhorn_knopp


class TestAppendLevel:
    def test_resolves_duplicates(self):
        codes = torch.tensor([[0, 1], [0, 1], [0, 1], [2, 3]])
        result = resolve_collisions(codes, strategy="append_level")
        assert len(result) == 4
        unique_tuples = {tuple(r) for r in result}
        assert len(unique_tuples) == 4
        assert result[3] == [2, 3]

    def test_no_collisions(self):
        codes = torch.tensor([[0, 1], [0, 2], [1, 0], [2, 3]])
        result = resolve_collisions(codes, strategy="append_level")
        assert result == [[0, 1], [0, 2], [1, 0], [2, 3]]

    def test_all_same_codes(self):
        codes = torch.tensor([[0, 0], [0, 0], [0, 0]])
        result = resolve_collisions(codes, strategy="append_level")
        unique_tuples = {tuple(r) for r in result}
        assert len(unique_tuples) == 3

    def test_items_fewer_than_codebook(self):
        codes = torch.tensor([[0], [1]])
        result = resolve_collisions(codes, strategy="append_level")
        assert result == [[0], [1]]


class TestSinkhorn:
    def test_convergence(self):
        cost = torch.tensor(
            [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]
        )
        transport = sinkhorn_knopp(cost, max_iter=100, epsilon=1.0)
        assert transport.shape == (3, 3)
        assert torch.allclose(transport.sum(dim=1), torch.ones(3), atol=0.05)
        assert torch.allclose(transport.sum(dim=0), torch.ones(3), atol=0.05)

    def test_sinkhorn_strategy_runs(self):
        codes = torch.tensor([[0, 1], [0, 1], [0, 2]])
        result = resolve_collisions(codes, strategy="sinkhorn")
        assert len(result) == 3
