"""Amazon 2015 dataset loader."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from datasets import Dataset
from loguru import logger

from saegenrec.data.loaders.base import DatasetLoader, register_loader
from saegenrec.data.schemas import INTERACTIONS_FEATURES, ITEM_METADATA_FEATURES


@register_loader("amazon2015")
class Amazon2015Loader(DatasetLoader):
    """Amazon 2015 dataset loader.

    File format: one JSON object per line (.json extension but actually JSON Lines).
    Review file: {Category}.json
    Metadata file: meta_{Category}.json
    """

    def load_interactions(self, data_dir: Path) -> Dataset:
        data_dir = Path(data_dir)
        category = data_dir.name
        review_file = data_dir / f"{category}.json"

        if not review_file.exists():
            raise FileNotFoundError(f"Review file not found: {review_file}")

        records = []
        seen = set()
        with open(review_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                user_id = raw.get("reviewerID", "")
                item_id = raw.get("asin", "")
                if not user_id or not item_id:
                    continue

                timestamp = int(raw.get("unixReviewTime", 0))
                dedup_key = (user_id, item_id, timestamp)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                records.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "rating": float(raw.get("overall", 0.0)),
                        "review_text": raw.get("reviewText", ""),
                        "review_summary": raw.get("summary", ""),
                    }
                )

        logger.info(f"Loaded {len(records)} interactions from {review_file}")
        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in INTERACTIONS_FEATURES},
            features=INTERACTIONS_FEATURES,
        )

    def load_item_metadata(self, data_dir: Path) -> Dataset:
        data_dir = Path(data_dir)
        category = data_dir.name
        meta_file = data_dir / f"meta_{category}.json"

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        records = []
        seen_ids = set()
        with open(meta_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    try:
                        raw = ast.literal_eval(line)
                    except (ValueError, SyntaxError):
                        continue

                item_id = raw.get("asin", "")
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                categories_raw = raw.get("categories", [[]])
                categories = categories_raw[0] if categories_raw else []
                categories = [c for c in categories if c]

                price_raw = raw.get("price")
                price = None
                if price_raw is not None:
                    try:
                        price = float(price_raw)
                    except (ValueError, TypeError):
                        price = None

                records.append(
                    {
                        "item_id": item_id,
                        "title": raw.get("title", ""),
                        "brand": raw.get("brand", ""),
                        "categories": categories,
                        "description": raw.get("description", ""),
                        "price": price,
                        "image_url": raw.get("imUrl", ""),
                    }
                )

        logger.info(f"Loaded {len(records)} item metadata from {meta_file}")
        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in ITEM_METADATA_FEATURES},
            features=ITEM_METADATA_FEATURES,
        )
