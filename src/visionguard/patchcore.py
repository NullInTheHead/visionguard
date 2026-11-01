"""PatchCore (Roth et al., CVPR 2022): memory bank of normal patch features, no training.

Steps (protocol section 4):
1. patch features from two backbone layers, 3x3 neighbourhood averaging, coarser layer
   upsampled to the finer grid, channels pooled to 1,024 per layer, layers averaged;
2. memory bank = patch features of the reference images, reduced by greedy coreset;
3. patch score = distance to the nearest bank vector;
4. image score = max patch score, reweighted by how isolated its nearest bank vector is;
5. anomaly map = patch scores upsampled to the input size + Gaussian blur (sigma 4 px).

Requires the ``models`` extra (torch).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import MVTecDataset, Sample
from .features import FeatureExtractor

# ------------------------------------------------------------------ patch embedding

_POOL_CACHE: dict[tuple, torch.Tensor] = {}


def _pool_matrix(length: int, out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(length, out) matrix M such that x @ M == F.adaptive_avg_pool1d(x, out).

    Same bins as PyTorch: output i averages inputs floor(i*L/out) .. ceil((i+1)*L/out) - 1.
    A matrix product runs on every device (Apple MPS cannot adaptive-pool when the
    sizes are not divisible, e.g. WideResNet's 512 channels -> 1024).
    """
    key = (length, out, str(device), dtype)
    if key not in _POOL_CACHE:
        m = torch.zeros(length, out, dtype=torch.float64)
        for i in range(out):
            start = (i * length) // out
            end = -((-(i + 1) * length) // out)  # ceil
            m[start:end, i] = 1.0 / (end - start)
        # Cast on the CPU first: MPS has no float64.
        _POOL_CACHE[key] = m.to(dtype).to(device)
    return _POOL_CACHE[key]


def adaptive_avg_pool_last(x: torch.Tensor, out: int) -> torch.Tensor:
    """Adaptive average pooling over the last dimension, (..., L) -> (..., out)."""
    return x @ _pool_matrix(x.shape[-1], out, x.device, x.dtype)


def embed_patches(
    features: Sequence[torch.Tensor], pretrain_dim: int = 1024, target_dim: int = 1024
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Feature maps -> (B, h*w, target_dim) patch vectors on the grid of the first map."""
    grid = tuple(features[0].shape[-2:])
    per_layer = []
    for f in features:
        # Local neighbourhood; zero padding at the borders, as in the reference code.
        f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
        if tuple(f.shape[-2:]) != grid:
            f = F.interpolate(f, size=grid, mode="bilinear", align_corners=False)
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(b * h * w, c)
        per_layer.append(adaptive_avg_pool_last(f, pretrain_dim))
    stacked = torch.stack(per_layer, dim=1)  # (N, L, pretrain_dim)
    n = stacked.shape[0]
    pooled = adaptive_avg_pool_last(stacked.reshape(n, -1), target_dim)
    b = features[0].shape[0]
    return pooled.reshape(b, grid[0] * grid[1], target_dim), grid


# ------------------------------------------------------------------------- coreset


def greedy_coreset(
    x: torch.Tensor,
    n: int,
    proj_dim: int | None = 128,
    seed: int = 0,
) -> torch.Tensor:
    """Greedy k-centre selection of ``n`` rows of ``x`` (N, D); returns their indices.

    Each step adds the point farthest from everything selected so far, so the bank
    covers the feature space evenly. Distances use a random Gaussian projection to
    ``proj_dim`` dimensions (Johnson-Lindenstrauss), as in the paper; ``None`` uses
    exact distances.
    """
    total = x.shape[0]
    if n >= total:
        return torch.arange(total, device=x.device)
    if n < 1:
        raise ValueError("Coreset must keep at least one point.")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = x.float()
    if proj_dim is not None and x.shape[1] > proj_dim:
        w = torch.randn(x.shape[1], proj_dim, generator=gen) / math.sqrt(proj_dim)
        z = z @ w.to(x.device)
    start = int(torch.randint(total, (1,), generator=gen))
    selected = torch.empty(n, dtype=torch.long, device=x.device)
    selected[0] = start
    min_dist = torch.linalg.vector_norm(z - z[start], dim=1)
    # The index stays on the device: indexing with z[idx] would copy idx to the CPU
    # and stall the GPU on every one of the n iterations.
    for i in range(1, n):
        idx = torch.argmax(min_dist, dim=0, keepdim=True)  # shape (1,)
        selected[i : i + 1] = idx
        dist = torch.linalg.vector_norm(z - torch.index_select(z, 0, idx), dim=1)
        torch.minimum(min_dist, dist, out=min_dist)
    return selected


# -------------------------------------------------------------- nearest neighbours


def nearest_neighbours(
    query: torch.Tensor, bank: torch.Tensor, k: int = 1, max_elements: int = 2**26
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact k-NN by Euclidean distance, chunked to bound memory: (M, k) distances, indices."""
    k = min(k, bank.shape[0])
    chunk = max(1, max_elements // max(1, bank.shape[0]))
    dists, idxs = [], []
    for start in range(0, query.shape[0], chunk):
        d = torch.cdist(query[start : start + chunk], bank)
        val, idx = torch.topk(d, k, dim=1, largest=False)
        dists.append(val)
        idxs.append(idx)
    return torch.cat(dists), torch.cat(idxs)


def reweighted_image_scores(
    patch_scores: torch.Tensor,
    nn_index: torch.Tensor,
    embeddings: torch.Tensor,
    bank: torch.Tensor,
    num_neighbors: int = 9,
) -> torch.Tensor:
    """Paper eq. 7 (as implemented in anomalib).

    The image score is the max patch score, scaled down when that patch's nearest bank
    vector sits in a dense region of the bank (a common pattern) and kept when it is
    isolated. patch_scores, nn_index: (B, P); embeddings: (B, P, D); bank: (N, D).
    """
    b = patch_scores.shape[0]
    rows = torch.arange(b, device=patch_scores.device)
    max_patch = torch.argmax(patch_scores, dim=1)
    score = patch_scores[rows, max_patch]
    if num_neighbors <= 1:
        return score
    feat = embeddings[rows, max_patch]  # (B, D)
    nn_vec = bank[nn_index[rows, max_patch]]  # (B, D)
    _, support = nearest_neighbours(nn_vec, bank, k=num_neighbors)  # (B, k); first = itself
    dist = torch.linalg.vector_norm(feat[:, None, :] - bank[support], dim=2)  # (B, k)
    weights = 1.0 - torch.softmax(dist, dim=1)[:, 0]
    return weights * score


# -------------------------------------------------------------------------- maps


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, H, W) maps with reflect padding (as anomalib)."""
    if sigma <= 0:
        return maps
    radius = int(4.0 * sigma + 0.5)
    coords = torch.arange(-radius, radius + 1, dtype=maps.dtype, device=maps.device)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    x = maps[:, None]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))
    return x[:, 0]


# ---------------------------------------------------------------------- detector


@dataclass
class Predictions:
    ids: list[str]
    labels: np.ndarray  # (N,) 1 = defective
    defect_types: list[str]
    scores: np.ndarray  # (N,) reweighted image score (the reported one)
    scores_max: np.ndarray  # (N,) plain max patch score
    maps: np.ndarray  # (N, S, S) float32 anomaly maps
    masks: np.ndarray  # (N, S, S) bool ground truth
    seconds_per_image: float = field(default=float("nan"))


class PatchCore:
    def __init__(
        self,
        extractor: FeatureExtractor,
        device: torch.device | str = "cpu",
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        sigma: float = 4.0,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
        proj_dim: int | None = 128,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.extractor = extractor.to(self.device)
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.sigma = sigma
        self.pretrain_dim = pretrain_dim
        self.target_dim = target_dim
        self.proj_dim = proj_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.memory_bank: torch.Tensor | None = None
        self.fit_seconds = float("nan")

    # ----------------------------------------------------------------- internals

    def _loader(self, samples: Sequence[Sample]):
        from torch.utils.data import DataLoader

        dataset = MVTecDataset(samples, self.extractor.preprocess)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def _embed(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feats = self.extractor(images.to(self.device, non_blocking=True))
        return embed_patches(feats, self.pretrain_dim, self.target_dim)

    def embed_samples(self, samples: Sequence[Sample]) -> torch.Tensor:
        """All patch vectors of ``samples`` as (N * P, D), kept on the CPU."""
        out = [
            self._embed(batch["image"])[0].flatten(0, 1).cpu() for batch in self._loader(samples)
        ]
        return torch.cat(out)

    def subsample(self, patches: torch.Tensor, ratio: float | None) -> torch.Tensor:
        """Greedy coreset of ``patches`` (CPU tensor) to ``ratio``; ``None`` or 1 keeps all."""
        if ratio is None or ratio >= 1.0:
            return patches
        n = max(1, round(ratio * patches.shape[0]))
        idx = greedy_coreset(patches.to(self.device), n, self.proj_dim, self.seed)
        return patches[idx.cpu()]

    # ----------------------------------------------------------------- public API

    def fit(
        self, reference: Sequence[Sample], coreset_ratio: float | None | str = "default"
    ) -> PatchCore:
        """Build the memory bank from reference images (defect-free training images only).

        ``coreset_ratio="default"`` uses the ratio given at construction; ``None`` keeps
        every patch (used for few-shot banks).
        """
        if any(s.split != "train" or s.label != 0 for s in reference):
            raise ValueError("Memory bank images must come from train/good only.")
        ratio = self.coreset_ratio if coreset_ratio == "default" else coreset_ratio
        start = time.perf_counter()
        bank = self.subsample(self.embed_samples(reference), ratio)
        self.set_memory_bank(bank)
        self.fit_seconds = time.perf_counter() - start
        return self

    def set_memory_bank(self, bank: torch.Tensor) -> None:
        """Use a bank built elsewhere (e.g. the union of several categories' coresets)."""
        self.memory_bank = bank.to(self.device, dtype=torch.float32).contiguous()

    @property
    def memory_bank_mb(self) -> float:
        if self.memory_bank is None:
            return 0.0
        return self.memory_bank.numel() * self.memory_bank.element_size() / 2**20

    @torch.no_grad()
    def predict(self, samples: Sequence[Sample]) -> Predictions:
        if self.memory_bank is None:
            raise RuntimeError("Call fit() first.")
        size = self.extractor.preprocess.output_size
        ids, labels, types, scores, scores_max, maps, masks = [], [], [], [], [], [], []
        n_images, start = 0, time.perf_counter()
        for batch in self._loader(samples):
            emb, grid = self._embed(batch["image"])
            b, p, d = emb.shape
            dist, idx = nearest_neighbours(emb.reshape(b * p, d), self.memory_bank, k=1)
            patch_scores = dist[:, 0].reshape(b, p)
            nn_index = idx[:, 0].reshape(b, p)
            img = reweighted_image_scores(
                patch_scores, nn_index, emb, self.memory_bank, self.num_neighbors
            )
            amap = F.interpolate(
                patch_scores.reshape(b, 1, *grid),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            amap = gaussian_blur(amap, self.sigma)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            scores.append(img.cpu())
            scores_max.append(patch_scores.amax(dim=1).cpu())
            maps.append(amap.cpu().float())
            masks.append(torch.as_tensor(batch["mask"]))
            labels.append(torch.as_tensor(batch["label"]))
            ids.extend(batch["id"])
            types.extend(batch["defect_type"])
            n_images += b
        elapsed = time.perf_counter() - start
        return Predictions(
            ids=ids,
            labels=torch.cat(labels).numpy().astype(int),
            defect_types=types,
            scores=torch.cat(scores).numpy(),
            scores_max=torch.cat(scores_max).numpy(),
            maps=torch.cat(maps).numpy(),
            masks=torch.cat(masks).numpy().astype(bool),
            seconds_per_image=elapsed / max(1, n_images),
        )
