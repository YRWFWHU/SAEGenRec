"""Image downloader for item product images."""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from loguru import logger
import requests
from tqdm import tqdm


def download_images(
    item_metadata_dir: Path,
    output_dir: Path,
    timeout: int = 30,
    max_retries: int = 3,
) -> dict:
    """Download item images from URLs in metadata.

    Supports resume (skips existing files) and error skipping.

    Returns:
        Statistics dict with download counts.
    """
    item_metadata = load_from_disk(str(item_metadata_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    downloaded = 0
    skipped = 0
    failed = 0
    errors = []

    for row in tqdm(item_metadata, desc="Downloading images"):
        item_id = row["item_id"]
        image_url = row.get("image_url", "")
        if not image_url:
            continue

        total += 1
        img_path = output_dir / f"{item_id}.jpg"

        if img_path.exists():
            skipped += 1
            continue

        for attempt in range(max_retries):
            try:
                resp = requests.get(image_url, timeout=timeout, stream=True)
                resp.raise_for_status()
                with open(img_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded += 1
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    errors.append({"item_id": item_id, "url": image_url, "error": str(e)})
                    failed += 1

    stats = {
        "total_urls": total,
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "errors": errors[:100],
    }

    logger.info(
        f"Image download: {downloaded} downloaded, {skipped} skipped, {failed} failed "
        f"out of {total} total"
    )

    return stats
