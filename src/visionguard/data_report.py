"""Phase 1 data inventory: counts, defect sizes, centre-crop losses and sample figures.

The numbers produced here feed three later decisions:
* ``n_tiny_defects``: defects smaller than one cell of a 28x28 feature grid,
  which a PatchCore-style detector can only see blurred. Input for the failure taxonomy.
* ``crop_retention``: share of each defect that survives the standard
  resize-256 / centre-crop-224 preprocessing. Defects cut away by the crop are
  still labelled defective, which silently penalises every method.
* ``counts_match_published``: catches partial or corrupted downloads.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .data import CATEGORIES, GOOD, Sample, by_split, category_type, index_category, load_mask_full

# (train images, test images) per category as published with MVTec AD.
PUBLISHED_COUNTS: dict[str, tuple[int, int]] = {
    "bottle": (209, 83),
    "cable": (224, 150),
    "capsule": (219, 132),
    "carpet": (280, 117),
    "grid": (264, 78),
    "hazelnut": (391, 110),
    "leather": (245, 124),
    "metal_nut": (220, 115),
    "pill": (267, 167),
    "screw": (320, 160),
    "tile": (230, 117),
    "toothbrush": (60, 42),
    "transistor": (213, 100),
    "wood": (247, 79),
    "zipper": (240, 151),
}

# One cell of a 28x28 feature grid (WideResNet-50 layer 2 at 224 px) as an image fraction.
TINY_AREA_FRAC = 1.0 / (28 * 28)
DEFAULT_CROP_KEEP = 224 / 256


def centre_box(h: int, w: int, keep: float) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the centre region that survives a crop keeping ``keep``."""
    kh, kw = round(h * keep), round(w * keep)
    top, left = (h - kh) // 2, (w - kw) // 2
    return top, top + kh, left, left + kw


def defect_stats(sample: Sample, crop_keep: float = DEFAULT_CROP_KEEP) -> dict:
    """Area, region count and centre-crop retention of one defective image's mask."""
    mask = load_mask_full(sample)
    h, w = mask.shape
    area = int(mask.sum())
    _, n_regions = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    top, bottom, left, right = centre_box(h, w, crop_keep)
    kept = int(mask[top:bottom, left:right].sum())
    return {
        "id": sample.id,
        "category": sample.category,
        "defect_type": sample.defect_type,
        "height": h,
        "width": w,
        "area_frac": area / mask.size,
        "n_regions": int(n_regions),
        "is_tiny": area / mask.size < TINY_AREA_FRAC,
        "crop_retention": kept / area if area else float("nan"),
    }


def _image_info(path: Path) -> tuple[str, str]:
    with Image.open(path) as im:  # header only; pixels are not decoded
        return f"{im.size[0]}x{im.size[1]}", im.mode


def dataset_report(
    root: str | Path,
    categories: Sequence[str] = CATEGORIES,
    crop_keep: float = DEFAULT_CROP_KEEP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (per-category table, per-defect-type table, per-defect-image table)."""
    cat_rows, image_rows = [], []
    for category in categories:
        samples = index_category(root, category)
        train, test = by_split(samples, "train"), by_split(samples, "test")
        defects = [s for s in test if s.label == 1]
        per_image = [defect_stats(s, crop_keep) for s in defects]
        image_rows.extend(per_image)

        info = [_image_info(s.path) for s in samples]
        area = np.array([r["area_frac"] for r in per_image]) if per_image else np.array([np.nan])
        retention = (
            np.array([r["crop_retention"] for r in per_image]) if per_image else np.array([])
        )
        published = PUBLISHED_COUNTS.get(category)
        cat_rows.append(
            {
                "category": category,
                "type": category_type(category),
                "n_train": len(train),
                "n_test_good": sum(s.defect_type == GOOD for s in test),
                "n_test_defect": len(defects),
                "n_defect_types": len({s.defect_type for s in defects}),
                "defect_types": ";".join(sorted({s.defect_type for s in defects})),
                "image_sizes": ";".join(sorted({size for size, _ in info})),
                "modes": ";".join(sorted({mode for _, mode in info})),
                "median_defect_area_pct": 100 * float(np.nanmedian(area)),
                "min_defect_area_pct": 100 * float(np.nanmin(area)),
                "n_tiny_defects": int(sum(r["is_tiny"] for r in per_image)),
                "n_crop_retention_lt50": int(np.sum(retention < 0.5)),
                "n_crop_retention_0": int(np.sum(retention == 0)),
                "counts_match_published": (
                    None if published is None else (len(train), len(test)) == published
                ),
            }
        )

    per_category = pd.DataFrame(cat_rows)
    per_image_df = pd.DataFrame(image_rows)
    if per_image_df.empty:
        per_defect = pd.DataFrame()
    else:
        per_defect = (
            per_image_df.groupby(["category", "defect_type"])
            .agg(
                n_images=("id", "size"),
                median_area_pct=("area_frac", lambda a: 100 * a.median()),
                min_area_pct=("area_frac", lambda a: 100 * a.min()),
                n_tiny=("is_tiny", "sum"),
                median_regions=("n_regions", "median"),
                min_crop_retention=("crop_retention", "min"),
            )
            .reset_index()
        )
    return per_category, per_defect, per_image_df


# --------------------------------------------------------------------- figures


def _thumb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))


def _thumb_mask(sample: Sample, size: int) -> np.ndarray:
    mask = Image.fromarray(load_mask_full(sample).astype(np.uint8) * 255)
    return np.asarray(mask.resize((size, size), Image.Resampling.NEAREST)) > 127


def _draw(ax, sample: Sample, size: int, crop_keep: float, title: str) -> None:
    from matplotlib.patches import Rectangle

    ax.imshow(_thumb(sample.path, size))
    if sample.label == 1:
        mask = _thumb_mask(sample, size)
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="red", linewidths=1.0)
    top, bottom, left, right = centre_box(size, size, crop_keep)
    ax.add_patch(
        Rectangle(
            (left - 0.5, top - 0.5),
            right - left,
            bottom - top,
            fill=False,
            edgecolor="yellow",
            linestyle="--",
            linewidth=0.8,
        )
    )
    ax.set_title(title, fontsize=7)
    ax.set_axis_off()


def plot_category(
    samples: Sequence[Sample],
    out_path: str | Path,
    n_normal: int = 3,
    seed: int = 0,
    crop_keep: float = DEFAULT_CROP_KEEP,
    thumb: int = 256,
    ncols: int = 6,
) -> Path:
    """Grid of normal training images plus one example per defect type (mask in red,
    centre-crop region dashed in yellow)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    train = sorted(by_split(samples, "train"))
    test = sorted(by_split(samples, "test"))
    picks: list[tuple[Sample, str]] = []
    for i in rng.choice(len(train), size=min(n_normal, len(train)), replace=False):
        picks.append((train[i], "train / good"))
    for defect in sorted({s.defect_type for s in test if s.label == 1}):
        group = [s for s in test if s.defect_type == defect]
        s = group[int(rng.integers(len(group)))]
        area = 100 * load_mask_full(s).mean()
        picks.append((s, f"{defect} ({area:.2f}% area)"))

    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.35 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (s, title) in zip(axes.ravel(), picks, strict=False):
        _draw(ax, s, thumb, crop_keep, title)
    fig.suptitle(samples[0].category, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_overview(
    samples_by_category: dict[str, Sequence[Sample]],
    out_path: str | Path,
    seed: int = 0,
    crop_keep: float = DEFAULT_CROP_KEEP,
    thumb: int = 192,
) -> Path:
    """One normal and one defective example per category, side by side."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    cats = list(samples_by_category)
    ncols = 6  # three categories per row, two panels each
    nrows = int(np.ceil(2 * len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    flat = axes.ravel()
    for i, cat in enumerate(cats):
        samples = samples_by_category[cat]
        train = sorted(by_split(samples, "train"))
        defects = sorted(s for s in samples if s.label == 1)
        _draw(flat[2 * i], train[int(rng.integers(len(train)))], thumb, crop_keep, f"{cat}: good")
        if defects:
            d = defects[int(rng.integers(len(defects)))]
            _draw(flat[2 * i + 1], d, thumb, crop_keep, f"{cat}: {d.defect_type}")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
