"""RQ-KMeans ItemTokenizer implementation (T015)."""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
import faiss
from loguru import logger
import numpy as np
import torch

from saegenrec.modeling.tokenizers.base import ItemTokenizer, register_item_tokenizer


@register_item_tokenizer("rqkmeans")
class RQKMeansTokenizer(ItemTokenizer):
    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        use_constrained: bool = False,
        kmeans_niter: int = 20,
        use_gpu: bool = False,
        **kwargs,
    ):
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._use_constrained = use_constrained
        self._kmeans_niter = kmeans_niter
        self._use_gpu = use_gpu
        self._centroids: list[np.ndarray] = []

    def train(
        self,
        semantic_embeddings_dir: Path | str,
        collaborative_embeddings_dir: Path | str | None,
        config: dict,
    ) -> dict:
        ds = load_from_disk(str(semantic_embeddings_dir))
        embeddings = np.array(ds["embedding"], dtype=np.float32)

        self._centroids = []
        residual = embeddings.copy()

        for k in range(self._num_codebooks):
            n_items = len(residual)
            actual_k = min(self._codebook_size, n_items)

            if self._use_constrained:
                from k_means_constrained import KMeansConstrained

                size_min = max(1, n_items // (actual_k * 2))
                size_max = max(size_min + 1, (n_items + actual_k - 1) // actual_k * 2)
                km = KMeansConstrained(
                    n_clusters=actual_k,
                    size_min=size_min,
                    size_max=size_max,
                    random_state=42,
                    max_iter=self._kmeans_niter,
                )
                km.fit(residual)
                centroids = km.cluster_centers_.astype(np.float32)
                codes = km.predict(residual)
            else:
                kmeans = faiss.Kmeans(
                    d=residual.shape[1],
                    k=actual_k,
                    niter=self._kmeans_niter,
                    gpu=self._use_gpu,
                )
                kmeans.train(residual)
                _, indices = kmeans.index.search(residual, 1)
                codes = indices.flatten()
                centroids = kmeans.centroids.copy()

            self._centroids.append(centroids)
            quantized = centroids[codes]
            residual = residual - quantized

            unique_codes = len(np.unique(codes))
            logger.info(f"Codebook {k}: {unique_codes}/{actual_k} entries used")

        return {
            "num_codebooks": self._num_codebooks,
            "codebook_size": self._codebook_size,
        }

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
        else:
            embeddings_np = np.asarray(embeddings, dtype=np.float32)

        all_codes = []
        residual = embeddings_np.copy()

        for centroids in self._centroids:
            dists = np.linalg.norm(residual[:, None, :] - centroids[None, :, :], axis=2)
            codes = dists.argmin(axis=1)
            quantized = centroids[codes]
            residual = residual - quantized
            all_codes.append(codes)

        return torch.tensor(np.stack(all_codes, axis=1), dtype=torch.long)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {f"codebook_{i}": c for i, c in enumerate(self._centroids)}
        np.savez(path / "centroids.npz", **data)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        data = np.load(path / "centroids.npz")
        self._centroids = []
        i = 0
        while f"codebook_{i}" in data:
            self._centroids.append(data[f"codebook_{i}"])
            i += 1
        self._num_codebooks = len(self._centroids)

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_size(self) -> int:
        return self._codebook_size
