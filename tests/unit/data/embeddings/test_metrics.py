"""Tests for recommendation evaluation metrics (Hit Rate@K, NDCG@K)."""

import math

import torch

from saegenrec.data.embeddings.collaborative.models.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
)


class TestHitRateAtK:
    def test_perfect_ranking(self):
        scores = torch.tensor([[10.0, 1.0, 0.0, -1.0]])
        targets = torch.tensor([0])
        assert hit_rate_at_k(scores, targets, k=1) == 1.0

    def test_miss(self):
        scores = torch.tensor([[1.0, 10.0, 0.0, -1.0]])
        targets = torch.tensor([0])
        assert hit_rate_at_k(scores, targets, k=1) == 0.0

    def test_within_topk(self):
        scores = torch.tensor([[5.0, 10.0, 3.0, 1.0]])
        targets = torch.tensor([0])
        assert hit_rate_at_k(scores, targets, k=2) == 1.0

    def test_batch(self):
        scores = torch.tensor([
            [10.0, 1.0, 0.0],
            [0.0, 1.0, 10.0],
        ])
        targets = torch.tensor([0, 2])
        assert hit_rate_at_k(scores, targets, k=1) == 1.0

    def test_half_hit(self):
        scores = torch.tensor([
            [10.0, 1.0, 0.0],
            [0.0, 10.0, 1.0],
        ])
        targets = torch.tensor([0, 2])
        hr = hit_rate_at_k(scores, targets, k=1)
        assert abs(hr - 0.5) < 1e-6


class TestNDCGAtK:
    def test_perfect_ranking(self):
        scores = torch.tensor([[10.0, 1.0, 0.0, -1.0]])
        targets = torch.tensor([0])
        expected = 1.0 / math.log2(0 + 2)
        assert abs(ndcg_at_k(scores, targets, k=1) - expected) < 1e-6

    def test_second_position(self):
        scores = torch.tensor([[5.0, 10.0, 1.0, 0.0]])
        targets = torch.tensor([0])
        expected = 1.0 / math.log2(1 + 2)
        assert abs(ndcg_at_k(scores, targets, k=5) - expected) < 1e-6

    def test_miss(self):
        scores = torch.tensor([[1.0, 10.0, 5.0, 3.0]])
        targets = torch.tensor([0])
        assert ndcg_at_k(scores, targets, k=1) == 0.0

    def test_batch_consistency(self):
        scores = torch.tensor([
            [10.0, 1.0],
            [1.0, 10.0],
        ])
        targets = torch.tensor([0, 1])
        expected = 1.0 / math.log2(2)
        assert abs(ndcg_at_k(scores, targets, k=1) - expected) < 1e-6
