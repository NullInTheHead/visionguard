"""Memory banks for E2: cached per-category coresets, the shared zero-shot bank, and
few-shot banks built from cached reference embeddings.

Coresets are slow to build and are reused across E2 runs (and later E3), so each one is
saved under ``cache/banks/`` (git-ignored). The file name carries every setting that
changes the bank, plus ``CACHE_VERSION``, which is bumped whenever the embedding or
coreset code changes, so a stale bank is never reused.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .data import Sample, by_split, calibration_split, index_category, sample_k_shot
from .patchcore import PatchCore

CACHE_VERSION = 1


def reference_pool(root: str | Path, category: str, cal_fraction: float, cal_seed: int):
    """(reference images, calibration images, test images) of one category, as in E1."""
    samples = index_category(root, category)
    reference, calibration = calibration_split(by_split(samples, "train"), cal_fraction, cal_seed)
    return reference, calibration, by_split(samples, "test")


def _bank_path(cache_dir: Path, backbone: str, category: str, model: PatchCore,
               ratio: float, cal_fraction: float, cal_seed: int) -> Path:  # fmt: skip
    name = (
        f"v{CACHE_VERSION}_{backbone}_{category}_r{ratio:g}_cal{cal_fraction:g}s{cal_seed}"
        f"_core{model.seed}_d{model.pretrain_dim}-{model.target_dim}_p{model.proj_dim}.pt"
    )
    return cache_dir / name


def category_coreset(
    model: PatchCore,
    backbone: str,
    root: str | Path,
    category: str,
    ratio: float,
    cal_fraction: float,
    cal_seed: int,
    cache_dir: str | Path | None = "cache/banks",
) -> torch.Tensor:
    """The E1 coreset of one category (CPU tensor), loaded from the cache when present."""
    path = None
    if cache_dir is not None:
        path = _bank_path(Path(cache_dir), backbone, category, model, ratio, cal_fraction,
                          cal_seed)  # fmt: skip
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True)
    reference, _, _ = reference_pool(root, category, cal_fraction, cal_seed)
    bank = model.subsample(model.embed_samples(reference), ratio)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, path)
    return bank


def shared_bank(
    model: PatchCore,
    backbone: str,
    root: str | Path,
    source_categories: Sequence[str],
    target: str,
    ratio: float,
    cal_fraction: float,
    cal_seed: int,
    cache_dir: str | Path | None = "cache/banks",
) -> tuple[torch.Tensor, list[str]]:
    """E2a bank: union of the source categories' coresets, never including ``target``."""
    sources = [c for c in source_categories if c != target]
    if not sources:
        raise ValueError("The zero-shot bank needs at least one category other than the target.")
    banks = [
        category_coreset(model, backbone, root, c, ratio, cal_fraction, cal_seed, cache_dir)
        for c in sources
    ]
    return torch.cat(banks), sources


def few_shot_draws(
    reference: Sequence[Sample], ks: Sequence[int], seeds: Sequence[int]
) -> dict[tuple[int, int], list[Sample]]:
    """{(k, seed): reference images}; nested in k for each seed (protocol section 7)."""
    k_max = max(ks)
    draws = {}
    for seed in seeds:
        largest = sample_k_shot(reference, k_max, seed)
        for k in ks:
            draws[(k, seed)] = largest[:k]
            assert draws[(k, seed)] == sample_k_shot(reference, k, seed)  # nesting holds
    return draws


def embed_references(model: PatchCore, images: Sequence[Sample]) -> dict[str, torch.Tensor]:
    """Patch embeddings (P, D) of each distinct reference image, keyed by image id."""
    unique = sorted({s.id: s for s in images}.values())
    data = model.embed_for_scoring(unique)
    return {i: data.embeddings[n] for n, i in enumerate(data.ids)}


# ------------------------------------------------------------------ E2c rotations

ROTATIONS: dict[str, tuple[float, ...]] = {
    "none": (0.0,),
    "rot90": (0.0, 90.0, 180.0, 270.0),
    "rot45": tuple(45.0 * i for i in range(8)),
}


def valid_patch_mask(
    size: int, grid: tuple[int, int], angle: float, erode: int = 2
) -> torch.Tensor:
    """(h*w,) bool: patches that lie well inside the image after rotating by ``angle``.

    A patch is kept when its own cell, and every cell within ``erode`` cells of it, holds
    no empty pixel from the rotation's corners (the backbone sees well beyond one cell).
    Multiples of 90 degrees rotate a square exactly, so every patch is kept.
    """
    import torch.nn.functional as F
    from torchvision.transforms.functional import InterpolationMode, rotate

    h, w = grid
    if angle % 90 == 0:
        return torch.ones(h * w, dtype=torch.bool)
    ones = torch.ones(1, 1, size, size)
    pixels = rotate(ones, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)
    empty = (F.adaptive_avg_pool2d(pixels, (h, w)) < 1.0 - 1e-6).float()
    if erode > 0:  # grow the empty region by `erode` cells (image borders stay valid)
        empty = F.max_pool2d(empty, 2 * erode + 1, stride=1, padding=erode)
    return (empty[0, 0] == 0).flatten()


@torch.no_grad()
def embed_references_rotated(
    model: PatchCore, images: Sequence[Sample], angles: Sequence[float]
) -> dict[str, torch.Tensor]:
    """Like ``embed_references``, but each image is also embedded rotated by every angle;
    returns {image id: (valid patches over all angles, D)} on the CPU."""
    from torchvision.transforms.functional import InterpolationMode, rotate

    from .data import MVTecDataset

    unique = sorted({s.id: s for s in images}.values())
    dataset = MVTecDataset(unique, model.extractor.preprocess)
    size = model.extractor.preprocess.output_size
    out: dict[str, list[torch.Tensor]] = {s.id: [] for s in unique}
    for start in range(0, len(unique), model.batch_size):
        items = [dataset[i] for i in range(start, min(start + model.batch_size, len(unique)))]
        batch = torch.as_tensor(np.stack([it["image"] for it in items]))
        for angle in angles:
            x = (
                batch
                if angle == 0
                else rotate(batch, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            )
            emb, grid = model._embed(x)
            keep = valid_patch_mask(size, grid, angle).to(emb.device)
            for it, e in zip(items, emb, strict=True):
                out[it["id"]].append(e[keep].cpu())
    return {i: torch.cat(parts) for i, parts in out.items()}
