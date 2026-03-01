"""Synthetic test data fixtures for data pipeline unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from saegenrec.data.schemas import (
    INTERACTIONS_FEATURES,
    ITEM_METADATA_FEATURES,
    USER_SEQUENCES_FEATURES,
)


@pytest.fixture
def synthetic_interactions_data() -> dict:
    """Synthetic interaction records — 15 interactions across 3 users and 5 items."""
    return {
        "user_id": [
            "u1", "u1", "u1", "u1", "u1",
            "u2", "u2", "u2", "u2", "u2",
            "u3", "u3", "u3", "u3", "u3",
        ],
        "item_id": [
            "i1", "i2", "i3", "i4", "i5",
            "i1", "i2", "i3", "i4", "i5",
            "i1", "i2", "i3", "i4", "i5",
        ],
        "timestamp": [
            100, 200, 300, 400, 500,
            150, 250, 350, 450, 550,
            110, 210, 310, 410, 510,
        ],
        "rating": [
            5.0, 4.0, 3.0, 2.0, 1.0,
            4.0, 3.0, 5.0, 4.0, 2.0,
            3.0, 5.0, 4.0, 3.0, 5.0,
        ],
        "review_text": [f"review text {i}" for i in range(15)],
        "review_summary": [f"summary {i}" for i in range(15)],
    }


@pytest.fixture
def synthetic_interactions_dataset(synthetic_interactions_data) -> Dataset:
    """HuggingFace Dataset of synthetic interactions."""
    return Dataset.from_dict(synthetic_interactions_data, features=INTERACTIONS_FEATURES)


@pytest.fixture
def synthetic_item_metadata_data() -> dict:
    """Synthetic item metadata for 5 items."""
    return {
        "item_id": ["i1", "i2", "i3", "i4", "i5"],
        "title": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"],
        "brand": ["BrandA", "BrandB", "BrandA", "BrandC", "BrandB"],
        "categories": [
            ["cat1", "cat2"],
            ["cat1"],
            ["cat2", "cat3"],
            ["cat1", "cat2"],
            ["cat3"],
        ],
        "description": [f"Description for item {i}" for i in range(1, 6)],
        "price": [9.99, 19.99, None, 29.99, 14.99],
        "image_url": [
            "http://img.example.com/1.jpg",
            "http://img.example.com/2.jpg",
            "",
            "http://img.example.com/4.jpg",
            "http://img.example.com/5.jpg",
        ],
    }


@pytest.fixture
def synthetic_item_metadata_dataset(synthetic_item_metadata_data) -> Dataset:
    """HuggingFace Dataset of synthetic item metadata."""
    return Dataset.from_dict(synthetic_item_metadata_data, features=ITEM_METADATA_FEATURES)


@pytest.fixture
def synthetic_user_sequences_data() -> dict:
    """Synthetic user sequences — 3 users each with 5 items."""
    return {
        "user_id": [0, 1, 2],
        "item_ids": [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ],
        "timestamps": [
            [100, 200, 300, 400, 500],
            [150, 250, 350, 450, 550],
            [110, 210, 310, 410, 510],
        ],
        "ratings": [
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 5.0, 4.0, 2.0],
            [3.0, 5.0, 4.0, 3.0, 5.0],
        ],
        "review_texts": [
            ["r0", "r1", "r2", "r3", "r4"],
            ["r5", "r6", "r7", "r8", "r9"],
            ["r10", "r11", "r12", "r13", "r14"],
        ],
        "review_summaries": [
            ["s0", "s1", "s2", "s3", "s4"],
            ["s5", "s6", "s7", "s8", "s9"],
            ["s10", "s11", "s12", "s13", "s14"],
        ],
    }


@pytest.fixture
def synthetic_user_sequences_dataset(synthetic_user_sequences_data) -> Dataset:
    """HuggingFace Dataset of synthetic user sequences."""
    return Dataset.from_dict(synthetic_user_sequences_data, features=USER_SEQUENCES_FEATURES)


@pytest.fixture
def amazon2015_raw_dir(tmp_path) -> Path:
    """Create a temporary directory with synthetic Amazon2015 raw data files."""
    data_dir = tmp_path / "raw" / "Amazon2015" / "TestCat"
    data_dir.mkdir(parents=True)

    reviews = [
        {
            "reviewerID": "u1", "asin": "i1", "reviewText": "Great product",
            "overall": 5.0, "summary": "Love it", "unixReviewTime": 1000,
        },
        {
            "reviewerID": "u1", "asin": "i2", "reviewText": "Good quality",
            "overall": 4.0, "summary": "Nice", "unixReviewTime": 2000,
        },
        {
            "reviewerID": "u1", "asin": "i3", "reviewText": "Decent",
            "overall": 3.0, "summary": "OK", "unixReviewTime": 3000,
        },
        {
            "reviewerID": "u1", "asin": "i4", "reviewText": "Not bad",
            "overall": 4.0, "summary": "Fine", "unixReviewTime": 4000,
        },
        {
            "reviewerID": "u1", "asin": "i5", "reviewText": "Average",
            "overall": 3.0, "summary": "Meh", "unixReviewTime": 5000,
        },
        {
            "reviewerID": "u2", "asin": "i1", "reviewText": "Works well",
            "overall": 4.0, "summary": "Good", "unixReviewTime": 1500,
        },
        {
            "reviewerID": "u2", "asin": "i2", "reviewText": "Solid",
            "overall": 5.0, "summary": "Great", "unixReviewTime": 2500,
        },
        {
            "reviewerID": "u2", "asin": "i3", "reviewText": "Fair",
            "overall": 3.0, "summary": "Okay", "unixReviewTime": 3500,
        },
        {
            "reviewerID": "u2", "asin": "i4", "reviewText": "Helpful",
            "overall": 4.0, "summary": "Useful", "unixReviewTime": 4500,
        },
        {
            "reviewerID": "u2", "asin": "i5", "reviewText": "Eh",
            "overall": 2.0, "summary": "Pass", "unixReviewTime": 5500,
        },
        {
            "reviewerID": "u3", "asin": "i1", "reviewText": "Cool",
            "overall": 4.0, "summary": "Nice one", "unixReviewTime": 1100,
        },
        {
            "reviewerID": "u3", "asin": "i2", "reviewText": "Fine",
            "overall": 3.0, "summary": "Alright", "unixReviewTime": 2100,
        },
        {
            "reviewerID": "u3", "asin": "i3", "reviewText": "Yes",
            "overall": 5.0, "summary": "Perfect", "unixReviewTime": 3100,
        },
        {
            "reviewerID": "u3", "asin": "i4", "reviewText": "Like it",
            "overall": 4.0, "summary": "Recommend", "unixReviewTime": 4100,
        },
        {
            "reviewerID": "u3", "asin": "i5", "reviewText": "Neat",
            "overall": 5.0, "summary": "Top", "unixReviewTime": 5100,
        },
    ]
    review_file = data_dir / "TestCat.json"
    with open(review_file, "w") as f:
        for r in reviews:
            f.write(json.dumps(r) + "\n")

    meta_records = [
        {
            "asin": "i1", "title": "Item One", "brand": "BrandA",
            "categories": [["Cat1", "SubCat1"]], "description": "Desc 1",
            "price": 9.99, "imUrl": "http://img/1.jpg",
        },
        {
            "asin": "i2", "title": "Item Two", "brand": "BrandB",
            "categories": [["Cat1"]], "description": "Desc 2",
            "price": 19.99, "imUrl": "http://img/2.jpg",
        },
        {
            "asin": "i3", "title": "Item Three", "brand": "",
            "categories": [[]], "description": "Desc 3",
            "imUrl": "",
        },
        {
            "asin": "i4", "title": "Item Four", "brand": "BrandC",
            "categories": [["Cat2", "SubCat2"]], "description": "Desc 4",
            "price": 29.99, "imUrl": "http://img/4.jpg",
        },
        {
            "asin": "i5", "title": "Item Five", "brand": "BrandB",
            "categories": [["Cat3"]], "description": "Desc 5",
            "price": 14.99, "imUrl": "http://img/5.jpg",
        },
    ]
    meta_file = data_dir / "meta_TestCat.json"
    with open(meta_file, "w") as f:
        for m in meta_records:
            f.write(json.dumps(m) + "\n")

    return data_dir


@pytest.fixture
def amazon2023_raw_dir(tmp_path) -> Path:
    """Create a temporary directory with synthetic Amazon2023 raw data files."""
    data_dir = tmp_path / "raw" / "Amazon2023" / "TestCat"
    data_dir.mkdir(parents=True)

    reviews = [
        {
            "user_id": "u1", "parent_asin": "i1", "text": "Great product",
            "rating": 5.0, "title": "Love it", "timestamp": 1000000,
        },
        {
            "user_id": "u1", "parent_asin": "i2", "text": "Good quality",
            "rating": 4.0, "title": "Nice", "timestamp": 2000000,
        },
        {
            "user_id": "u1", "parent_asin": "i3", "text": "Decent",
            "rating": 3.0, "title": "OK", "timestamp": 3000000,
        },
        {
            "user_id": "u1", "parent_asin": "i4", "text": "Not bad",
            "rating": 4.0, "title": "Fine", "timestamp": 4000000,
        },
        {
            "user_id": "u1", "parent_asin": "i5", "text": "Average",
            "rating": 3.0, "title": "Meh", "timestamp": 5000000,
        },
        {
            "user_id": "u2", "parent_asin": "i1", "text": "Works well",
            "rating": 4.0, "title": "Good", "timestamp": 1500000,
        },
        {
            "user_id": "u2", "parent_asin": "i2", "text": "Solid",
            "rating": 5.0, "title": "Great", "timestamp": 2500000,
        },
        {
            "user_id": "u2", "parent_asin": "i3", "text": "Fair",
            "rating": 3.0, "title": "Okay", "timestamp": 3500000,
        },
        {
            "user_id": "u2", "parent_asin": "i4", "text": "Helpful",
            "rating": 4.0, "title": "Useful", "timestamp": 4500000,
        },
        {
            "user_id": "u2", "parent_asin": "i5", "text": "Eh",
            "rating": 2.0, "title": "Pass", "timestamp": 5500000,
        },
        {
            "user_id": "u3", "parent_asin": "i1", "text": "Cool",
            "rating": 4.0, "title": "Nice one", "timestamp": 1100000,
        },
        {
            "user_id": "u3", "parent_asin": "i2", "text": "Fine",
            "rating": 3.0, "title": "Alright", "timestamp": 2100000,
        },
        {
            "user_id": "u3", "parent_asin": "i3", "text": "Yes",
            "rating": 5.0, "title": "Perfect", "timestamp": 3100000,
        },
        {
            "user_id": "u3", "parent_asin": "i4", "text": "Like it",
            "rating": 4.0, "title": "Recommend", "timestamp": 4100000,
        },
        {
            "user_id": "u3", "parent_asin": "i5", "text": "Neat",
            "rating": 5.0, "title": "Top", "timestamp": 5100000,
        },
    ]
    review_file = data_dir / "TestCat.jsonl"
    with open(review_file, "w") as f:
        for r in reviews:
            f.write(json.dumps(r) + "\n")

    meta_records = [
        {
            "parent_asin": "i1", "title": "Item One", "store": "BrandA",
            "categories": ["Cat1", "SubCat1"], "description": ["Desc", "1"],
            "price": "9.99", "images": [{"large": "http://img/1.jpg"}],
        },
        {
            "parent_asin": "i2", "title": "Item Two", "store": "BrandB",
            "categories": ["Cat1"], "description": ["Desc 2"],
            "price": "19.99", "images": [{"large": "http://img/2.jpg"}],
        },
        {
            "parent_asin": "i3", "title": "Item Three", "store": "",
            "categories": [], "description": ["Desc 3"],
            "price": None, "images": [],
        },
        {
            "parent_asin": "i4", "title": "Item Four", "store": "BrandC",
            "categories": ["Cat2", "SubCat2"], "description": ["Desc", "4"],
            "price": "$29.99", "images": [{"large": "http://img/4.jpg"}],
        },
        {
            "parent_asin": "i5", "title": "Item Five", "store": "BrandB",
            "categories": ["Cat3"], "description": ["Desc 5"],
            "price": "14.99", "images": [{"large": "http://img/5.jpg"}],
        },
    ]
    meta_file = data_dir / "meta_TestCat.jsonl"
    with open(meta_file, "w") as f:
        for m in meta_records:
            f.write(json.dumps(m) + "\n")

    return data_dir


@pytest.fixture
def sample_config_path(tmp_path) -> Path:
    """Create a sample YAML config file."""
    config = {
        "dataset": {"name": "amazon2015", "category": "TestCat", "raw_dir": "data/raw"},
        "processing": {"kcore_threshold": 3, "split_strategy": "loo", "max_seq_len": 10},
        "tokenizer": {"name": "passthrough", "params": {}},
        "embedding": {"enabled": False},
        "output": {"interim_dir": "data/interim", "processed_dir": "data/processed"},
    }
    import yaml

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file
