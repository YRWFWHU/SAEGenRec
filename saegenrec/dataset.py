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
    force: bool = typer.Option(False, "--force", help="Force overwrite existing outputs"),
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

    stats = run_pipeline(cfg, steps=steps or None, force=force)
    logger.info(f"Pipeline stats: {stats}")


@app.command()
def embed_semantic(
    config: Path = typer.Argument(..., help="YAML config file path"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing output"),
    model_name: Optional[str] = typer.Option(
        None, "--model-name", help="Override semantic_embedding.model_name"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Override semantic_embedding.device"
    ),
):
    """Generate semantic embeddings for item metadata."""
    from dataclasses import asdict

    from saegenrec.data.config import load_config
    from saegenrec.data.embeddings.semantic.base import get_semantic_embedder
    import saegenrec.data.embeddings.semantic.sentence_transformer  # noqa: F401

    cfg = load_config(config)
    stage1_dir = cfg.output.interim_path(cfg.dataset.name, cfg.dataset.category)

    if not (stage1_dir / "item_metadata").exists():
        logger.error(
            f"Stage 1 data not found at {stage1_dir}. "
            "Run 'process --step load --step filter --step sequence' first."
        )
        raise typer.Exit(code=1)

    embed_cfg = cfg.semantic_embedding
    if model_name is not None:
        embed_cfg.model_name = model_name
    if device is not None:
        embed_cfg.device = device

    config_dict = asdict(embed_cfg)
    config_dict["force"] = force

    embedder = get_semantic_embedder(embed_cfg.name)
    embedder.generate(stage1_dir, stage1_dir, config_dict)


@app.command()
def embed_collaborative(
    config: Path = typer.Argument(..., help="YAML config file path"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing output"),
    device: Optional[str] = typer.Option(
        None, "--device", help="Override collaborative_embedding.device"
    ),
    num_epochs: Optional[int] = typer.Option(
        None, "--num-epochs", help="Override collaborative_embedding.num_epochs"
    ),
):
    """Generate collaborative embeddings by training a sequential recommendation model."""
    from dataclasses import asdict

    from saegenrec.data.config import load_config
    from saegenrec.data.embeddings.collaborative.base import get_collaborative_embedder
    import saegenrec.data.embeddings.collaborative.sasrec  # noqa: F401

    cfg = load_config(config)
    stage1_dir = cfg.output.interim_path(cfg.dataset.name, cfg.dataset.category)
    stage2_dir = stage1_dir / cfg.processing.split_strategy

    if not (stage2_dir / "train_sequences").exists():
        logger.error(f"Stage 2 data not found at {stage2_dir}. Run 'process --step split' first.")
        raise typer.Exit(code=1)

    embed_cfg = cfg.collaborative_embedding
    if device is not None:
        embed_cfg.device = device
    if num_epochs is not None:
        embed_cfg.num_epochs = num_epochs

    config_dict = asdict(embed_cfg)
    config_dict["force"] = force

    embedder = get_collaborative_embedder(embed_cfg.name)
    embedder.generate(stage2_dir, stage2_dir, config_dict)


@app.command()
def tokenize(
    config: Path = typer.Argument(..., help="YAML config file path"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing output"),
):
    """Run item tokenization (embedding → SID map)."""
    from saegenrec.data.config import load_config
    from saegenrec.data.pipeline import run_pipeline

    cfg = load_config(config)
    if not cfg.item_tokenizer.enabled:
        cfg.item_tokenizer.enabled = True

    stats = run_pipeline(cfg, steps=["tokenize"], force=force)
    logger.info(f"Tokenize stats: {stats}")


@app.command(name="build-sft")
def build_sft(
    config: Path = typer.Argument(..., help="YAML config file path"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing output"),
):
    """Build SFT instruction-tuning dataset from SID map and sequences."""
    from saegenrec.data.config import load_config
    from saegenrec.data.pipeline import run_pipeline

    cfg = load_config(config)
    if not cfg.sft_builder.enabled:
        cfg.sft_builder.enabled = True

    stats = run_pipeline(cfg, steps=["build-sft"], force=force)
    logger.info(f"Build SFT stats: {stats}")


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
