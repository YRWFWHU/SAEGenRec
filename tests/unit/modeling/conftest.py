"""Shared fixtures for modeling unit tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import Dataset

from saegenrec.data.schemas import SID_MAP_FEATURES


@pytest.fixture
def synthetic_embeddings() -> torch.Tensor:
    """Synthetic embeddings: 20 items × 64 dimensions."""
    rng = np.random.default_rng(42)
    return torch.tensor(rng.standard_normal((20, 64)), dtype=torch.float32)


@pytest.fixture
def synthetic_embeddings_small() -> torch.Tensor:
    """Small synthetic embeddings: 8 items × 32 dimensions."""
    rng = np.random.default_rng(123)
    return torch.tensor(rng.standard_normal((8, 32)), dtype=torch.float32)


@pytest.fixture
def synthetic_item_ids() -> list[int]:
    """Item IDs matching synthetic_embeddings (20 items)."""
    return list(range(20))


@pytest.fixture
def mock_sid_map() -> Dataset:
    """Mock SID map dataset: 5 items with 4-level codes."""
    return Dataset.from_dict(
        {
            "item_id": [0, 1, 2, 3, 4],
            "codes": [
                [0, 1, 2, 3],
                [0, 1, 2, 4],
                [0, 2, 3, 1],
                [1, 0, 2, 3],
                [1, 1, 0, 2],
            ],
            "sid_tokens": [
                "<s_a_0><s_b_1><s_c_2><s_d_3>",
                "<s_a_0><s_b_1><s_c_2><s_d_4>",
                "<s_a_0><s_b_2><s_c_3><s_d_1>",
                "<s_a_1><s_b_0><s_c_2><s_d_3>",
                "<s_a_1><s_b_1><s_c_0><s_d_2>",
            ],
        },
        features=SID_MAP_FEATURES,
    )


@pytest.fixture
def mock_item_metadata_dataset() -> Dataset:
    """Mock item metadata for 5 items (mapped IDs)."""
    from saegenrec.data.schemas import ITEM_METADATA_FEATURES

    return Dataset.from_dict(
        {
            "item_id": ["orig_0", "orig_1", "orig_2", "orig_3", "orig_4"],
            "title": ["Item Zero", "Item One", "Item Two", "Item Three", "Item Four"],
            "brand": ["BrandA", "BrandB", "BrandA", "BrandC", "BrandB"],
            "categories": [["cat1"], ["cat1"], ["cat2"], ["cat1"], ["cat3"]],
            "description": [f"Desc {i}" for i in range(5)],
            "price": [9.99, 19.99, None, 29.99, 14.99],
            "image_url": [f"http://img/{i}.jpg" for i in range(5)],
        },
        features=ITEM_METADATA_FEATURES,
    )


@pytest.fixture
def mock_item_id_map_dataset() -> Dataset:
    """Mock item ID map: original → mapped."""
    from saegenrec.data.schemas import ID_MAP_FEATURES

    return Dataset.from_dict(
        {
            "original_id": ["orig_0", "orig_1", "orig_2", "orig_3", "orig_4"],
            "mapped_id": [0, 1, 2, 3, 4],
        },
        features=ID_MAP_FEATURES,
    )


@pytest.fixture
def mock_train_sequences_dataset() -> Dataset:
    """Mock train sequences: 3 users with sequences of mapped item IDs."""
    from saegenrec.data.schemas import USER_SEQUENCES_FEATURES

    return Dataset.from_dict(
        {
            "user_id": [0, 1, 2],
            "item_ids": [
                [0, 1, 2, 3, 4],
                [1, 2, 3],
                [0, 3, 4],
            ],
            "timestamps": [
                [100, 200, 300, 400, 500],
                [150, 250, 350],
                [110, 410, 510],
            ],
            "ratings": [
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [3.0, 5.0, 4.0],
                [4.0, 3.0, 5.0],
            ],
            "review_texts": [
                ["r0", "r1", "r2", "r3", "r4"],
                ["r5", "r6", "r7"],
                ["r8", "r9", "r10"],
            ],
            "review_summaries": [
                ["s0", "s1", "s2", "s3", "s4"],
                ["s5", "s6", "s7"],
                ["s8", "s9", "s10"],
            ],
        },
        features=USER_SEQUENCES_FEATURES,
    )
