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
