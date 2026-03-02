"""Index2Item SFT task builder."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from datasets import Dataset, load_from_disk

from saegenrec.data.schemas import SFT_FEATURES

from .base import SFTTaskBuilder, register_sft_task


@register_sft_task("index2item")
class Index2ItemTaskBuilder(SFTTaskBuilder):
    @property
    def task_type(self) -> str:
        return "index2item"

    def build(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        sid_map: Dataset,
        config: dict[str, Any],
    ) -> Dataset:
        templates = self.load_templates(config["template_file"])
        seed = config.get("seed", 42)
        rng = random.Random(seed)

        item_metadata = load_from_disk(str(stage1_dir / "item_metadata"))
        item_id_map = load_from_disk(str(stage1_dir / "item_id_map"))

        orig_to_mapped: dict[str, int] = {
            row["original_id"]: row["mapped_id"] for row in item_id_map
        }
        sid_lookup: dict[int, str] = {row["item_id"]: row["sid_tokens"] for row in sid_map}

        records: list[dict[str, str]] = []
        for row in item_metadata:
            original_id = row["item_id"]
            mapped_id = orig_to_mapped.get(original_id)
            if mapped_id is None:
                continue
            sid_tokens = sid_lookup.get(mapped_id)
            if sid_tokens is None:
                continue

            tpl = rng.choice(templates)
            records.append(
                {
                    "task_type": self.task_type,
                    "instruction": tpl["instruction"],
                    "input": tpl["input_template"].format(sid_tokens=sid_tokens),
                    "output": tpl["output_template"].format(title=row["title"]),
                }
            )

        return Dataset.from_dict(
            {k: [r[k] for r in records] for k in SFT_FEATURES},
            features=SFT_FEATURES,
        )
