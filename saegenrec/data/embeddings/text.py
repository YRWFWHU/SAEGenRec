"""Text embedding generation using sentence-transformers."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_from_disk
from loguru import logger
import numpy as np
from tqdm import tqdm

from saegenrec.data.schemas import TEXT_EMBEDDING_FEATURES


def generate_text_embeddings(
    item_metadata_dir: Path,
    item_id_map_dir: Path,
    output_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    text_fields: list[str] | None = None,
    batch_size: int = 256,
    device: str = "cpu",
) -> Dataset:
    """Generate text embeddings for item metadata.

    Concatenates specified text fields from item metadata, encodes using
    a sentence-transformers model, and saves L2-normalized embeddings.
    """
    from sentence_transformers import SentenceTransformer

    text_fields = text_fields or ["title", "brand", "categories"]

    item_metadata = load_from_disk(str(item_metadata_dir))
    item_id_map = load_from_disk(str(item_id_map_dir))

    orig_to_mapped = dict(zip(item_id_map["original_id"], item_id_map["mapped_id"]))

    # Build texts and collect mapped IDs
    texts = []
    mapped_ids = []
    for row in item_metadata:
        mapped_id = orig_to_mapped.get(row["item_id"])
        if mapped_id is None:
            continue

        parts = []
        for field_name in text_fields:
            val = row.get(field_name, "")
            if isinstance(val, list):
                val = " ".join(val)
            if val:
                parts.append(str(val))
        texts.append(" ".join(parts))
        mapped_ids.append(mapped_id)

    logger.info(f"Encoding {len(texts)} items with {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)

    if all_embeddings:
        embeddings_array = np.vstack(all_embeddings)
    else:
        embeddings_array = np.empty((0, 0), dtype=np.float32)

    ds = Dataset.from_dict(
        {
            "item_id": mapped_ids,
            "embedding": embeddings_array.tolist(),
        },
        features=TEXT_EMBEDDING_FEATURES,
    )

    output_path = Path(output_dir) / "text_embeddings"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))

    logger.info(f"Saved {len(ds)} text embeddings to {output_path}")
    return ds
