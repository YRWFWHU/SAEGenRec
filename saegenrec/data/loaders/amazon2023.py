"""Amazon 2023 dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
import re

from datasets import Dataset
from loguru import logger

from saegenrec.data.loaders.base import DatasetLoader, register_loader
from saegenrec.data.schemas import INTERACTIONS_FEATURES, ITEM_METADATA_FEATURES


@register_loader("amazon2023")
class Amazon2023Loader(DatasetLoader):
    """Amazon 2023 dataset loader.

    File format: JSON Lines (.jsonl extension).
    Review file: {Category}.jsonl
    Metadata file: meta_{Category}.jsonl
    """

    def load_interactions(self, data_dir: Path) -> Dataset:
        data_dir = Path(data_dir)
        category = data_dir.name
        review_file = data_dir / f"{category}.jsonl"

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

                user_id = raw.get("user_id", "")
                item_id = raw.get("parent_asin", "")
                if not user_id or not item_id:
                    continue

                timestamp_ms = int(raw.get("timestamp", 0))
                timestamp = timestamp_ms // 1000

                dedup_key = (user_id, item_id, timestamp)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                records.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "rating": float(raw.get("rating", 0.0)),
                        "review_text": raw.get("text", ""),
                        "review_summary": raw.get("title", ""),
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
        meta_file = data_dir / f"meta_{category}.jsonl"

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
                    continue

                item_id = raw.get("parent_asin", "")
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                categories = raw.get("categories", [])
                if not isinstance(categories, list):
                    categories = []

                desc_raw = raw.get("description", [])
                description = " ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw)

                price = self._parse_price(raw.get("price"))

                images = raw.get("images", [])
                image_url = ""
                if images and isinstance(images, list) and len(images) > 0:
                    first_img = images[0]
                    if isinstance(first_img, dict):
                        image_url = first_img.get("large", "")

                records.append(
                    {
                        "item_id": item_id,
                        "title": raw.get("title", ""),
                        "brand": raw.get("store", ""),
                        "categories": categories,
                        "description": description,
                        "price": price,
                        "image_url": image_url,
                    }
                )

        logger.info(f"Loaded {len(records)} item metadata from {meta_file}")
        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in ITEM_METADATA_FEATURES},
            features=ITEM_METADATA_FEATURES,
        )

    @staticmethod
    def _parse_price(price_raw) -> float | None:
        if price_raw is None:
            return None
        if isinstance(price_raw, (int, float)):
            return float(price_raw)
        if isinstance(price_raw, str):
            cleaned = re.sub(r"[^\d.]", "", price_raw)
            if cleaned:
                try:
                    return float(cleaned)
                except ValueError:
                    return None
        return None
