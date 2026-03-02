"""CLI entry point for the data processing pipeline."""

from pathlib import Path
from typing import Optional

from loguru import logger
import typer

app = typer.Typer()


@app.command()
def process(
    config: Path = typer.Argument(..., help="YAML config file path"),
    steps: list[str] = typer.Option(
        None,
        "--step",
        "-s",
        help="Steps to run: load, filter, sequence, split, augment, negative_sampling, generate, embed",
    ),
    dataset: Optional[str] = typer.Option(None, help="Override dataset.name"),
    category: Optional[str] = typer.Option(None, help="Override dataset.category"),
    kcore: Optional[int] = typer.Option(None, help="Override processing.kcore_threshold"),
    max_seq_len: Optional[int] = typer.Option(
        None, "--max-seq-len", help="Override processing.max_seq_len"
    ),
    num_negatives: Optional[int] = typer.Option(
        None, "--num-negatives", help="Override processing.num_negatives"
    ),
    split_strategy: Optional[str] = typer.Option(
        None, "--split-strategy", help="Override processing.split_strategy"
    ),
    split_ratio: Optional[str] = typer.Option(
        None,
        "--split-ratio",
        help="Override processing.split_ratio (comma-separated, e.g. 0.8,0.1,0.1)",
    ),
    seed: Optional[int] = typer.Option(None, help="Override processing.seed"),
):
    """Run the data processing pipeline."""
    from saegenrec.data.config import load_config
    from saegenrec.data.pipeline import run_pipeline

    logger.info(f"Loading config from {config}")
    cfg = load_config(config)

    if dataset is not None:
        cfg.dataset.name = dataset
    if category is not None:
        cfg.dataset.category = category
    if kcore is not None:
        cfg.processing.kcore_threshold = kcore
    if max_seq_len is not None:
        cfg.processing.max_seq_len = max_seq_len
    if num_negatives is not None:
        cfg.processing.num_negatives = num_negatives
    if split_strategy is not None:
        cfg.processing.split_strategy = split_strategy
    if split_ratio is not None:
        cfg.processing.split_ratio = [float(x) for x in split_ratio.split(",")]
    if seed is not None:
        cfg.processing.seed = seed

    stats = run_pipeline(cfg, steps=steps or None)
    logger.info(f"Pipeline stats: {stats}")


@app.command()
def download_images(
    config: Path = typer.Argument(..., help="YAML config file path"),
):
    """Download item images from metadata URLs."""
    from saegenrec.data.config import load_config
    from saegenrec.data.processors.images import download_images as _download

    cfg = load_config(config)
    interim_dir = cfg.output.interim_path(cfg.dataset.name, cfg.dataset.category)
    output_dir = (
        Path(cfg.output.processed_dir).parent
        / "external"
        / "images"
        / cfg.dataset.name
        / cfg.dataset.category
    )
    stats = _download(
        item_metadata_dir=interim_dir / "item_metadata",
        output_dir=output_dir,
    )
    logger.info(f"Download stats: {stats}")


if __name__ == "__main__":
    app()
