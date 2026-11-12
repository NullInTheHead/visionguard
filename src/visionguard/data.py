"""MVTec AD indexing, preprocessing, calibration split and k-shot sampling.

Everything here is torch-free so it can be unit-tested in CI without a GPU stack.
``MVTecDataset`` is a plain map-style dataset (``__len__`` + ``__getitem__``),
which ``torch.utils.data.DataLoader`` accepts as is.

Leakage guards that live in this module:
* reference images (calibration split, k-shot sampler) can only come from
  ``train/good``; passing any test sample raises;
* the calibration split and the k-shot sampler use separate random streams,
  derived from (seed, category), so results do not depend on Python's hash
  randomisation or on the order files were listed in.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

TEXTURES: tuple[str, ...] = ("carpet", "grid", "leather", "tile", "wood")
OBJECTS: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
CATEGORIES: tuple[str, ...] = tuple(sorted(TEXTURES + OBJECTS))

# VisA (Zou et al., ECCV 2022), the untouched clean test set (protocol decision D5), after
# scripts/prepare_visa.py has put it in the MVTec folder layout. Types as in the VisA paper.
VISA_TYPES: dict[str, str] = {
    **{c: "complex_structure" for c in ("pcb1", "pcb2", "pcb3", "pcb4")},
    **{c: "multiple_instances" for c in ("candle", "capsules", "macaroni1", "macaroni2")},
    **{c: "single_instance" for c in ("cashew", "chewinggum", "fryum", "pipe_fryum")},
}
VISA_CATEGORIES: tuple[str, ...] = tuple(sorted(VISA_TYPES))
DATASET_CATEGORIES: dict[str, tuple[str, ...]] = {"mvtec": CATEGORIES, "visa": VISA_CATEGORIES}
GOOD = "good"
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})

# Separate random streams so the calibration split and the k-shot draws never
# share random numbers, even with the same user-facing seed.
_STREAM_CALIBRATION = 1
_STREAM_KSHOT = 2


def category_type(category: str) -> str:
    """``"texture"`` or ``"object"`` for an MVTec AD category; the VisA type for VisA."""
    if category in TEXTURES:
        return "texture"
    if category in OBJECTS:
        return "object"
    if category in VISA_TYPES:
        return VISA_TYPES[category]
    raise ValueError(f"Unknown category: {category!r}")


@dataclass(frozen=True, order=True)
class Sample:
    """One image of the dataset. Ordering is canonical (category, split, defect, path)."""

    category: str
    split: str  # "train" or "test"
    defect_type: str  # "good" or the defect folder name
    path: Path
    mask_path: Path | None = None

    @property
    def label(self) -> int:
        """1 for a defective image, 0 for a defect-free one."""
        return int(self.defect_type != GOOD)

    @property
    def id(self) -> str:
        """Stable identifier, e.g. ``screw/test/scratch_head/003``."""
        return f"{self.category}/{self.split}/{self.defect_type}/{self.path.stem}"


# --------------------------------------------------------------------------- indexing


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def index_category(root: str | Path, category: str) -> list[Sample]:
    """List every train and test image of one category, with mask paths for defects."""
    cat_dir = Path(root) / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"Category folder not found: {cat_dir}")

    samples: list[Sample] = []
    for split in ("train", "test"):
        split_dir = cat_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect = defect_dir.name
            if split == "train" and defect != GOOD:
                raise ValueError(f"Unexpected non-normal training folder: {defect_dir}")
            for img in _image_files(defect_dir):
                mask = None
                if defect != GOOD:
                    mask = cat_dir / "ground_truth" / defect / f"{img.stem}_mask.png"
                    if not mask.is_file():
                        raise FileNotFoundError(f"Missing mask for {img}: expected {mask}")
                samples.append(Sample(category, split, defect, img, mask))
    if not any(s.split == "train" for s in samples):
        raise ValueError(f"No training images found for {category} under {cat_dir}")
    return samples


def index_dataset(root: str | Path, categories: Sequence[str] | None = None) -> list[Sample]:
    """Index several categories (all 15 by default)."""
    out: list[Sample] = []
    for category in categories or CATEGORIES:
        out.extend(index_category(root, category))
    return out


def by_split(samples: Sequence[Sample], split: str) -> list[Sample]:
    """Keep only ``"train"`` or ``"test"`` samples."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    return [s for s in samples if s.split == split]


# --------------------------------------------------------- reference pools and k-shot


def _stable_hash(text: str) -> int:
    """Deterministic across processes (unlike ``hash``)."""
    return zlib.crc32(text.encode("utf-8"))


def _rng(seed: int, category: str, stream: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), _stable_hash(category), stream])


def _check_reference_pool(pool: Sequence[Sample]) -> str:
    """Validate a pool of reference candidates and return its (single) category."""
    if not pool:
        raise ValueError("Reference pool is empty.")
    bad = [s.id for s in pool if s.split != "train" or s.label != 0]
    if bad:
        raise ValueError(
            "Reference images must come from train/good only; got e.g. " + ", ".join(bad[:3])
        )
    categories = {s.category for s in pool}
    if len(categories) != 1:
        raise ValueError(f"Reference pool mixes categories: {sorted(categories)}")
    return categories.pop()


def calibration_split(
    train: Sequence[Sample], fraction: float, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Split one category's defect-free training images into (reference, calibration).

    Reference images may enter a memory bank; calibration images never do. They
    are only scored to set a decision threshold without looking at any defect.
    ``fraction=0`` returns every image as reference (the published protocol).
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1).")
    category = _check_reference_pool(train)
    ordered = sorted(train)
    if fraction == 0.0:
        return ordered, []
    n_cal = max(1, round(fraction * len(ordered)))
    if n_cal >= len(ordered):
        raise ValueError("Calibration split would leave no reference images.")
    perm = _rng(seed, category, _STREAM_CALIBRATION).permutation(len(ordered))
    cal_idx = set(perm[:n_cal].tolist())
    reference = [s for i, s in enumerate(ordered) if i not in cal_idx]
    calibration = [s for i, s in enumerate(ordered) if i in cal_idx]
    return reference, calibration


def sample_k_shot(pool: Sequence[Sample], k: int | None, seed: int) -> list[Sample]:
    """Draw ``k`` reference images from one category's reference pool.

    * ``k=None`` returns the whole pool (the full-data end of the curve).
    * ``k=0`` returns nothing (the zero-shot end).
    * Draws are nested: for a fixed (category, seed), the k=1 draw is the first
      element of the k=2 draw, which is the start of the k=4 draw, and so on.
      Differences between points on the recovery curve therefore come from the
      added images, not from a different random draw.
    * The result does not depend on the order of ``pool``.
    """
    category = _check_reference_pool(pool)
    ordered = sorted(pool)
    if k is None:
        return ordered
    if k < 0:
        raise ValueError("k must be >= 0.")
    if k > len(ordered):
        raise ValueError(f"k={k} exceeds the {len(ordered)} reference images of {category}.")
    perm = _rng(seed, category, _STREAM_KSHOT).permutation(len(ordered))
    return [ordered[i] for i in perm[:k]]


# ------------------------------------------------------------------- preprocessing


@dataclass(frozen=True)
class Preprocess:
    """Square resize, then optional centre crop, applied identically to image and mask.

    Images use bilinear resampling (anti-aliased by PIL when shrinking); masks
    use nearest-neighbour so they stay binary. Normalisation is left to the
    backbone wrapper because ImageNet and CLIP use different statistics.
    """

    resize: int = 256
    crop: int | None = 224

    def __post_init__(self) -> None:
        if self.crop is not None and self.crop > self.resize:
            raise ValueError("crop must not exceed resize.")

    @property
    def output_size(self) -> int:
        return self.crop if self.crop is not None else self.resize

    def _crop_box(self) -> tuple[int, int, int, int]:
        size = self.output_size
        off = (self.resize - size) // 2
        return off, off, off + size, off + size

    def image(self, img: Image.Image) -> np.ndarray:
        """PIL image -> float32 array (3, S, S) with values in [0, 1]."""
        img = img.convert("RGB").resize((self.resize, self.resize), Image.Resampling.BILINEAR)
        if self.crop is not None:
            img = img.crop(self._crop_box())
        return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0

    def mask(self, mask: Image.Image) -> np.ndarray:
        """PIL mask -> bool array (S, S)."""
        mask = mask.convert("L").resize((self.resize, self.resize), Image.Resampling.NEAREST)
        if self.crop is not None:
            mask = mask.crop(self._crop_box())
        return np.asarray(mask) > 127


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def load_mask_full(sample: Sample) -> np.ndarray:
    """Ground-truth mask at the original image resolution (all False for good images)."""
    if sample.mask_path is None:
        with Image.open(sample.path) as im:
            w, h = im.size
        return np.zeros((h, w), dtype=bool)
    with Image.open(sample.mask_path) as m:
        return np.asarray(m.convert("L")) > 127


class MVTecDataset:
    """Map-style dataset over a list of samples; usable directly with a torch DataLoader."""

    def __init__(self, samples: Sequence[Sample], preprocess: Preprocess | None = None) -> None:
        self.samples = list(samples)
        self.preprocess = preprocess or Preprocess()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        with Image.open(s.path) as im:
            image = self.preprocess.image(im)
        if s.mask_path is not None:
            with Image.open(s.mask_path) as m:
                mask = self.preprocess.mask(m)
        else:
            size = self.preprocess.output_size
            mask = np.zeros((size, size), dtype=bool)
        return {
            "image": image,
            "mask": mask,
            "label": s.label,
            "category": s.category,
            "defect_type": s.defect_type,
            "id": s.id,
        }
