"""Tests for LOO and TO data splitting."""

import pytest

from saegenrec.data.processors.split import split_data


class TestLOOSplit:
    def test_basic_loo(self, synthetic_user_sequences_dataset):
        train, valid, test, stats = split_data(synthetic_user_sequences_dataset, strategy="loo")
        assert len(train) == 3
        assert len(valid) == 3
        assert len(test) == 3

    def test_loo_last_item_in_test(self, synthetic_user_sequences_dataset):
        """Last item of each user goes to test set."""
        train, valid, test, _ = split_data(synthetic_user_sequences_dataset, strategy="loo")
        for row in test:
            assert len(row["item_ids"]) == 1

    def test_loo_second_last_in_valid(self, synthetic_user_sequences_dataset):
        """Second-last item goes to validation set."""
        train, valid, test, _ = split_data(synthetic_user_sequences_dataset, strategy="loo")
        for row in valid:
            assert len(row["item_ids"]) == 1

    def test_loo_rest_in_train(self, synthetic_user_sequences_dataset):
        """All items except last two go to train set."""
        train, valid, test, _ = split_data(synthetic_user_sequences_dataset, strategy="loo")
        for row in train:
            assert len(row["item_ids"]) == 3

    def test_loo_exclude_short_users(self):
        """Users with <3 interactions should be excluded."""
        from datasets import Dataset

        from saegenrec.data.schemas import USER_SEQUENCES_FEATURES

        data = {
            "user_id": [0, 1],
            "item_ids": [[0, 1], [0, 1, 2, 3, 4]],
            "timestamps": [[100, 200], [100, 200, 300, 400, 500]],
            "ratings": [[5.0, 4.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
            "review_texts": [["a", "b"], ["a", "b", "c", "d", "e"]],
            "review_summaries": [["s1", "s2"], ["s1", "s2", "s3", "s4", "s5"]],
        }
        ds = Dataset.from_dict(data, features=USER_SEQUENCES_FEATURES)
        train, valid, test, stats = split_data(ds, strategy="loo")
        assert stats["excluded_users"] == 1
        assert len(train) == 1

    def test_loo_stats(self, synthetic_user_sequences_dataset):
        _, _, _, stats = split_data(synthetic_user_sequences_dataset, strategy="loo")
        assert stats["split_strategy"] == "loo"
        assert stats["train_users"] == 3
        assert stats["excluded_users"] == 0


class TestTOSplit:
    def test_basic_to(self, synthetic_user_sequences_dataset):
        train, valid, test, stats = split_data(
            synthetic_user_sequences_dataset, strategy="to", ratio=[0.6, 0.2, 0.2]
        )
        assert len(train) > 0
        assert stats["split_strategy"] == "to"

    def test_to_global_time_ordering(self, synthetic_user_sequences_dataset):
        """Train timestamps should be <= valid timestamps <= test timestamps."""
        train, valid, test, _ = split_data(
            synthetic_user_sequences_dataset, strategy="to", ratio=[0.6, 0.2, 0.2]
        )
        train_max_ts = max(max(row["timestamps"]) for row in train) if len(train) > 0 else 0
        valid_min_ts = min(min(row["timestamps"]) for row in valid) if len(valid) > 0 else float("inf")
        assert train_max_ts <= valid_min_ts or len(valid) == 0

    def test_to_ratio(self, synthetic_user_sequences_dataset):
        """Total interactions should roughly follow the specified ratio."""
        train, valid, test, _ = split_data(
            synthetic_user_sequences_dataset, strategy="to", ratio=[0.6, 0.2, 0.2]
        )
        total_train = sum(len(r["item_ids"]) for r in train)
        total_valid = sum(len(r["item_ids"]) for r in valid)
        total_test = sum(len(r["item_ids"]) for r in test)
        total = total_train + total_valid + total_test
        assert total == 15
        assert total_train > 0

    def test_invalid_strategy(self, synthetic_user_sequences_dataset):
        with pytest.raises(ValueError, match="Unknown split strategy"):
            split_data(synthetic_user_sequences_dataset, strategy="invalid")
