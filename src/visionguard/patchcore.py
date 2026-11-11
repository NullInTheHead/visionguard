"""PatchCore (Roth et al., CVPR 2022): memory bank of normal patch features, no training.

Steps (protocol section 4):
1. patch features from two backbone layers, 3x3 neighbourhood averaging, coarser layer
   upsampled to the finer grid, each layer's channels pooled to 1,024, the two vectors
   concatenated (2,048) and pooled to 1,024 (see ``embed_patches``);
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
    """Feature maps -> (B, h*w, target_dim) patch vectors on the grid of the first map.

    Per layer: 3x3 average pooling, upsampling to the first map's grid, then adaptive
    average pooling of the channels to ``pretrain_dim``. The per-layer vectors are then
    concatenated (L x pretrain_dim) and adaptive-average-pooled to ``target_dim``, as in
    the original PatchCore code (its ``MeanMapper`` and ``Aggregator``). With two layers
    and target_dim == pretrain_dim, that last step averages *adjacent dimensions within
    each layer's vector*: the first half of the output comes only from the first layer,
    the second half only from the second. It is not an average across layers.

    anomalib differs: it concatenates the raw channels (512 + 1,024 = 1,536 for WRN-50)
    with no reduction. Our reproduction check agrees with it within 0.5 AUROC points.
    """
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
    """Greedy k-centre selection of ``n`` rows of ``x`` (N, D); returns CPU indices.

    Each step adds the point farthest from everything selected so far, so the bank
    covers the feature space evenly. Distances use a random Gaussian projection to
    ``proj_dim`` dimensions (Johnson-Lindenstrauss), as in the paper; ``None`` uses
    exact distances.

    Runs on the CPU whatever the device of ``x``: the loop is thousands of small,
    sequential steps, which is faster on the CPU than on an Apple GPU, and it makes the
    selection independent of the GPU type. Squared distances come from one
    matrix-vector product per step (||a||^2 - 2 a.b + ||b||^2), about 10x faster than
    forming the differences.
    """
    total = x.shape[0]
    if n >= total:
        return torch.arange(total)
    if n < 1:
        raise ValueError("Coreset must keep at least one point.")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = x.detach().to("cpu", torch.float32)
    if proj_dim is not None and x.shape[1] > proj_dim:
        w = torch.randn(x.shape[1], proj_dim, generator=gen) / math.sqrt(proj_dim)
        z = z @ w
    sq_norm = (z * z).sum(dim=1)
    start = int(torch.randint(total, (1,), generator=gen))
    selected = torch.empty(n, dtype=torch.long)
    selected[0] = start
    min_dist = (sq_norm - 2.0 * (z @ z[start]) + sq_norm[start]).clamp_min_(0.0)
    for i in range(1, n):
        idx = int(torch.argmax(min_dist))
        selected[i] = idx
        dist = sq_norm - 2.0 * torch.mv(z, z[idx]) + sq_norm[idx]
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


# ------------------------------------------------------------------ measurement


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so a timer measures the work and not just its launch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (never decreases)."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes vs KiB


def device_memory_mb(device: torch.device) -> dict[str, float]:
    """Device memory: peak for CUDA; MPS only reports what is allocated right now."""
    if device.type == "cuda":
        return {"device_peak_mb": torch.cuda.max_memory_allocated(device) / 2**20}
    if device.type == "mps":
        return {"device_current_mb": torch.mps.current_allocated_memory() / 2**20}
    return {}


class StageTimer:
    """Accumulates wall time per named stage, synchronising the device around each one."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds: dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds

    def stage(self, name: str):
        timer = self

        class _Stage:
            def __enter__(self):
                synchronize(timer.device)
                self.start = time.perf_counter()

            def __exit__(self, *exc):
                synchronize(timer.device)
                timer.add(name, time.perf_counter() - self.start)

        return _Stage()


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
    timing: dict[str, float] = field(default_factory=dict)  # seconds per image, per stage


@dataclass
class EmbeddedSet:
    """Patch embeddings of a set of images, kept on the CPU so they can be scored against
    many memory banks (E2b) without running the backbone again."""

    ids: list[str]
    labels: np.ndarray
    defect_types: list[str]
    masks: np.ndarray
    embeddings: torch.Tensor  # (N, P, D) float32, CPU
    grid: tuple[int, int]
    timing: dict[str, float] = field(default_factory=dict)  # total seconds per stage

    def __len__(self) -> int:
        return len(self.ids)


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
        self.fit_timing: dict[str, float] = {}

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

    def _batches(self, samples: Sequence[Sample], timer: StageTimer):
        """Yield (batch, embeddings on device, grid), timing loading and feature extraction."""
        it = iter(self._loader(samples))
        while True:
            start = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            timer.add("load", time.perf_counter() - start)
            with timer.stage("features"):
                emb, grid = self._embed(batch["image"])
            yield batch, emb, grid

    def embed_samples(self, samples: Sequence[Sample]) -> torch.Tensor:
        """All patch vectors of ``samples`` as (N * P, D), kept on the CPU."""
        timer = StageTimer(self.device)
        out = [emb.flatten(0, 1).cpu() for _, emb, _ in self._batches(samples, timer)]
        self._last_embed_timing = timer.seconds
        return torch.cat(out)

    def subsample(self, patches: torch.Tensor, ratio: float | None) -> torch.Tensor:
        """Greedy coreset of ``patches`` (CPU tensor) to ``ratio``; ``None`` or 1 keeps all."""
        if ratio is None or ratio >= 1.0:
            return patches
        n = max(1, round(ratio * patches.shape[0]))
        idx = greedy_coreset(patches, n, self.proj_dim, self.seed)
        return patches[idx]

    def _score_batch(self, emb: torch.Tensor, grid: tuple[int, int], timer: StageTimer):
        """Image scores, plain max scores and smoothed maps for one batch of embeddings."""
        b, p, d = emb.shape
        size = self.extractor.preprocess.output_size
        with timer.stage("nn_search"):
            dist, idx = nearest_neighbours(emb.reshape(b * p, d), self.memory_bank, k=1)
            patch_scores = dist[:, 0].reshape(b, p)
            img = reweighted_image_scores(
                patch_scores, idx[:, 0].reshape(b, p), emb, self.memory_bank, self.num_neighbors
            )
        with timer.stage("postprocess"):
            amap = F.interpolate(
                patch_scores.reshape(b, 1, *grid),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            amap = gaussian_blur(amap, self.sigma).cpu().float()
        return img.cpu(), patch_scores.amax(dim=1).cpu(), amap

    @staticmethod
    def _per_image(seconds: dict[str, float], n: int) -> dict[str, float]:
        return {k: v / max(1, n) for k, v in seconds.items()}

    # ----------------------------------------------------------------- public API

    def fit(
        self, reference: Sequence[Sample], coreset_ratio: float | None | str = "default"
    ) -> PatchCore:
        """Build the memory bank from reference images (defect-free training images only).

        ``coreset_ratio="default"`` uses the ratio given at construction; ``None`` keeps
        every patch (used for few-shot banks). ``fit_timing`` records load, feature and
        coreset seconds separately.
        """
        if any(s.split != "train" or s.label != 0 for s in reference):
            raise ValueError("Memory bank images must come from train/good only.")
        ratio = self.coreset_ratio if coreset_ratio == "default" else coreset_ratio
        start = time.perf_counter()
        patches = self.embed_samples(reference)
        timing = dict(self._last_embed_timing)
        t0 = time.perf_counter()
        bank = self.subsample(patches, ratio)
        timing["coreset"] = time.perf_counter() - t0
        self.set_memory_bank(bank)
        self.fit_seconds = time.perf_counter() - start
        self.fit_timing = timing
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
        """Score images batch by batch (low memory). ``timing`` = seconds per image per stage."""
        if self.memory_bank is None:
            raise RuntimeError("Call fit() first.")
        timer = StageTimer(self.device)
        ids, labels, types, scores, scores_max, maps, masks = [], [], [], [], [], [], []
        start = time.perf_counter()
        for batch, emb, grid in self._batches(samples, timer):
            img, mx, amap = self._score_batch(emb, grid, timer)
            scores.append(img)
            scores_max.append(mx)
            maps.append(amap)
            masks.append(torch.as_tensor(batch["mask"]))
            labels.append(torch.as_tensor(batch["label"]))
            ids.extend(batch["id"])
            types.extend(batch["defect_type"])
        n = len(ids)
        return Predictions(
            ids=ids,
            labels=torch.cat(labels).numpy().astype(int),
            defect_types=types,
            scores=torch.cat(scores).numpy(),
            scores_max=torch.cat(scores_max).numpy(),
            maps=torch.cat(maps).numpy(),
            masks=torch.cat(masks).numpy().astype(bool),
            seconds_per_image=(time.perf_counter() - start) / max(1, n),
            timing=self._per_image(timer.seconds, n),
        )

    @torch.no_grad()
    def embed_for_scoring(self, samples: Sequence[Sample]) -> EmbeddedSet:
        """Run the backbone once and keep every image's patch embeddings on the CPU."""
        timer = StageTimer(self.device)
        ids, labels, types, masks, embs = [], [], [], [], []
        grid = (0, 0)
        for batch, emb, batch_grid in self._batches(samples, timer):
            grid = batch_grid
            embs.append(emb.cpu())
            masks.append(torch.as_tensor(batch["mask"]))
            labels.append(torch.as_tensor(batch["label"]))
            ids.extend(batch["id"])
            types.extend(batch["defect_type"])
        return EmbeddedSet(
            ids=ids,
            labels=torch.cat(labels).numpy().astype(int),
            defect_types=types,
            masks=torch.cat(masks).numpy().astype(bool),
            embeddings=torch.cat(embs),
            grid=grid,
            timing=timer.seconds,
        )

    @torch.no_grad()
    def score_embedded(self, data: EmbeddedSet) -> Predictions:
        """Score cached embeddings against the current memory bank. ``timing`` per image
        includes the cached load and feature time, so it is comparable with ``predict``."""
        if self.memory_bank is None:
            raise RuntimeError("Call fit() or set_memory_bank() first.")
        timer = StageTimer(self.device)
        scores, scores_max, maps = [], [], []
        start = time.perf_counter()
        for i in range(0, len(data), self.batch_size):
            emb = data.embeddings[i : i + self.batch_size].to(self.device)
            img, mx, amap = self._score_batch(emb, data.grid, timer)
            scores.append(img)
            scores_max.append(mx)
            maps.append(amap)
        n = len(data)
        seconds = {**data.timing, **timer.seconds}
        return Predictions(
            ids=list(data.ids),
            labels=data.labels,
            defect_types=list(data.defect_types),
            scores=torch.cat(scores).numpy(),
            scores_max=torch.cat(scores_max).numpy(),
            maps=torch.cat(maps).numpy(),
            masks=data.masks,
            seconds_per_image=(time.perf_counter() - start + sum(data.timing.values())) / max(1, n),
            timing=self._per_image(seconds, n),
        )
