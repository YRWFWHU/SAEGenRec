"""SentenceTransformer-based semantic embedder (default implementation)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from loguru import logger

from saegenrec.data.embeddings.semantic.base import SemanticEmbedder, register_semantic_embedder
from saegenrec.data.schemas import SEMANTIC_EMBEDDING_FEATURES


@register_semantic_embedder("sentence-transformer")
class SentenceTransformerEmbedder(SemanticEmbedder):
    """Encode item metadata text fields into semantic embeddings
    using a sentence-transformers model.
    """

    def generate(self, data_dir: Path, output_dir: Path, config: dict) -> Dataset:
        from sentence_transformers import SentenceTransformer

        model_name: str = config.get("model_name", "all-MiniLM-L6-v2")
        text_fields: list[str] = config.get(
            "text_fields", ["title", "brand", "description", "price"]
        )
        normalize: bool = config.get("normalize", False)
        batch_size: int = config.get("batch_size", 256)
        device: str = config.get("device", "cpu")
        force: bool = config.get("force", False)

        output_path = Path(output_dir) / "item_semantic_embeddings"

        if output_path.exists() and not force:
            logger.info(
                f"Semantic embeddings already exist at {output_path}, skipping. "
                "Use --force to regenerate."
            )
            return load_from_disk(str(output_path))

        if output_path.exists() and force:
            shutil.rmtree(output_path)

        item_metadata_dir = Path(data_dir) / "item_metadata"
        item_id_map_dir = Path(data_dir) / "item_id_map"

        if not item_metadata_dir.exists():
            raise FileNotFoundError(
                f"item_metadata not found at {item_metadata_dir}. Run Stage 1 first."
            )
        if not item_id_map_dir.exists():
            raise FileNotFoundError(
                f"item_id_map not found at {item_id_map_dir}. Run Stage 1 first."
            )

        t0 = time.time()

        item_metadata = load_from_disk(str(item_metadata_dir))
        item_id_map = load_from_disk(str(item_id_map_dir))

        orig_to_mapped = dict(
            zip(item_id_map["original_id"], item_id_map["mapped_id"])
        )

        metadata_by_orig = {}
        for row in item_metadata:
            metadata_by_orig[row["item_id"]] = row

        texts: list[str] = []
        mapped_ids: list[int] = []
        empty_count = 0

        for orig_id, mapped_id in sorted(orig_to_mapped.items(), key=lambda x: x[1]):
            if orig_id not in metadata_by_orig:
                logger.warning(
                    f"Item {orig_id} (mapped_id={mapped_id}) exists in item_id_map "
                    "but not in item_metadata — skipping."
                )
                continue

            row = metadata_by_orig[orig_id]
            parts: list[str] = []
            for field_name in text_fields:
                val = row.get(field_name, "")
                if val is None:
                    val = ""
                if isinstance(val, (int, float)):
                    val = str(val)
                if isinstance(val, list):
                    val = " ".join(str(v) for v in val)
                val = str(val).strip()
                if val:
                    parts.append(val)

            text = " ".join(parts)
            if not text.strip():
                empty_count += 1
            texts.append(text)
            mapped_ids.append(mapped_id)

        if empty_count == len(texts) and len(texts) > 0:
            logger.warning(
                "All items have empty text fields — all embeddings will be zero vectors."
            )

        logger.info(f"Encoding {len(texts)} items with {model_name} on {device}")
        model = SentenceTransformer(model_name, device=device)
        embed_dim = model.get_sentence_embedding_dimension()

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            has_content = [bool(t.strip()) for t in batch_texts]
            non_empty_texts = [t for t, h in zip(batch_texts, has_content) if h]

            if non_empty_texts:
                batch_embeds = model.encode(
                    non_empty_texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
            else:
                batch_embeds = np.empty((0, embed_dim), dtype=np.float32)

            result = np.zeros((len(batch_texts), embed_dim), dtype=np.float32)
            non_empty_idx = 0
            for j, h in enumerate(has_content):
                if h:
                    result[j] = batch_embeds[non_empty_idx]
                    non_empty_idx += 1

            all_embeddings.append(result)

        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)
        else:
            embeddings_array = np.empty((0, embed_dim), dtype=np.float32)

        ds = Dataset.from_dict(
            {
                "item_id": mapped_ids,
                "embedding": embeddings_array.tolist(),
            },
            features=SEMANTIC_EMBEDDING_FEATURES,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_path))

        elapsed = time.time() - t0
        logger.info(
            f"Saved {len(ds)} semantic embeddings to {output_path}"
        )
        logger.info(
            f"Stats: items={len(ds)}, dim={embed_dim}, elapsed={elapsed:.1f}s"
        )

        return ds
