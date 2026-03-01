"""CLI entry point for the data processing pipeline."""

from pathlib import Path

import typer
from loguru import logger

app = typer.Typer()


@app.command()
def process(
    config: Path = typer.Argument(..., help="YAML config file path"),
    steps: list[str] = typer.Option(
        None,
        "--step",
        "-s",
        help="Steps to run (load, filter, sequence, split, augment, generate, embed)",
    ),
):
    """Run the data processing pipeline."""
    from saegenrec.data.config import load_config
    from saegenrec.data.pipeline import run_pipeline

    logger.info(f"Loading config from {config}")
    cfg = load_config(config)
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
