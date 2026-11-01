from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from visionguard.data import (
    CATEGORIES,
    MVTecDataset,
    Preprocess,
    by_split,
    calibration_split,
    category_type,
    index_category,
    index_dataset,
    sample_k_shot,
)


def test_categories_are_the_fifteen_mvtec_ones():
    assert len(CATEGORIES) == 15
    assert sum(category_type(c) == "texture" for c in CATEGORIES) == 5
    with pytest.raises(ValueError):
        category_type("banana")


# ---------------------------------------------------------------- indexing


def test_index_counts_labels_and_masks(fake_mvtec):
    samples = index_category(fake_mvtec, "carpet")
    train, test = by_split(samples, "train"), by_split(samples, "test")
    assert len(train) == 6  # .DS_Store and license.txt ignored
    assert len(test) == 5
    assert all(s.label == 0 and s.mask_path is None for s in train)
    defects = [s for s in test if s.label == 1]
    assert {s.defect_type for s in defects} == {"hole", "cut"}
    assert all(s.mask_path is not None and s.mask_path.is_file() for s in defects)
    assert defects[0].id.startswith("carpet/test/")


def test_index_missing_category_raises(fake_mvtec):
    with pytest.raises(FileNotFoundError):
        index_category(fake_mvtec, "bottle")


def test_index_missing_mask_raises(tmp_path):
    (tmp_path / "wood" / "train" / "good").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "wood" / "train" / "good" / "000.png")
    (tmp_path / "wood" / "test" / "hole").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "wood" / "test" / "hole" / "000.png")
    with pytest.raises(FileNotFoundError, match="Missing mask"):
        index_category(tmp_path, "wood")


# --------------------------------------------------------- calibration split


def test_calibration_split_is_disjoint_complete_and_deterministic(fake_mvtec):
    train = by_split(index_category(fake_mvtec, "carpet"), "train")
    ref, cal = calibration_split(train, fraction=0.34, seed=0)
    assert len(cal) == 2 and len(ref) == 4
    assert set(ref).isdisjoint(cal)
    assert set(ref) | set(cal) == set(train)
    assert calibration_split(list(reversed(train)), 0.34, seed=0) == (ref, cal)


def test_calibration_fraction_zero_keeps_everything(fake_mvtec):
    train = by_split(index_category(fake_mvtec, "carpet"), "train")
    ref, cal = calibration_split(train, fraction=0.0)
    assert cal == [] and ref == sorted(train)


def test_calibration_rejects_test_images(fake_mvtec):
    samples = index_category(fake_mvtec, "carpet")
    with pytest.raises(ValueError, match="train/good"):
        calibration_split(samples, fraction=0.2)


# ---------------------------------------------------------------- k-shot


@pytest.fixture()
def carpet_pools(fake_mvtec):
    samples = index_category(fake_mvtec, "carpet")
    ref, cal = calibration_split(by_split(samples, "train"), fraction=0.2, seed=0)
    return ref, cal, by_split(samples, "test")


def test_k_shot_never_touches_test_or_calibration(carpet_pools):
    ref, cal, test = carpet_pools
    for seed in range(5):
        for k in range(len(ref) + 1):
            refs = sample_k_shot(ref, k, seed)
            assert len(refs) == k == len(set(refs))
            assert set(refs) <= set(ref)
            assert set(refs).isdisjoint(test)
            assert set(refs).isdisjoint(cal)
            assert {r.path for r in refs}.isdisjoint({t.path for t in test})


def test_k_shot_is_nested_across_k(carpet_pools):
    ref, _, _ = carpet_pools
    for seed in range(5):
        draws = [sample_k_shot(ref, k, seed) for k in range(1, len(ref) + 1)]
        for small, big in zip(draws, draws[1:], strict=False):
            assert big[: len(small)] == small


def test_k_shot_deterministic_order_free_and_seed_dependent(carpet_pools):
    ref, _, _ = carpet_pools
    shuffled = ref[:]
    random.Random(123).shuffle(shuffled)
    assert sample_k_shot(ref, 2, seed=3) == sample_k_shot(shuffled, 2, seed=3)
    draws = {tuple(s.id for s in sample_k_shot(ref, 2, seed)) for seed in range(10)}
    assert len(draws) > 1


def test_k_shot_edge_cases(carpet_pools):
    ref, _, test = carpet_pools
    assert sample_k_shot(ref, 0, seed=0) == []
    assert sample_k_shot(ref, None, seed=0) == sorted(ref)
    with pytest.raises(ValueError, match="exceeds"):
        sample_k_shot(ref, len(ref) + 1, seed=0)
    with pytest.raises(ValueError, match="train/good"):
        sample_k_shot(ref + test, 1, seed=0)


def test_k_shot_rejects_mixed_categories(fake_mvtec):
    train = by_split(index_dataset(fake_mvtec, ["carpet", "screw"]), "train")
    with pytest.raises(ValueError, match="mixes categories"):
        sample_k_shot(train, 1, seed=0)


# ------------------------------------------------------------ preprocessing


def test_dataset_item_shapes_and_grayscale_to_rgb(fake_mvtec):
    samples = index_category(fake_mvtec, "screw")  # grayscale category
    ds = MVTecDataset(samples, Preprocess(resize=32, crop=28))
    item = ds[0]
    assert item["image"].shape == (3, 28, 28)
    assert item["image"].dtype == np.float32
    assert item["image"].min() >= 0.0 and item["image"].max() <= 1.0
    assert item["mask"].shape == (28, 28) and item["mask"].dtype == bool
    good = next(ds[i] for i in range(len(ds)) if ds[i]["label"] == 0)
    assert not good["mask"].any()
    bad = next(ds[i] for i in range(len(ds)) if ds[i]["label"] == 1)
    assert bad["mask"].any()


def test_image_and_mask_stay_aligned(tmp_path):
    """A white square in the image must land exactly where its mask lands."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    img[40:88, 56:104] = 255
    mask[40:88, 56:104] = 255
    pre = Preprocess(resize=64, crop=48)
    x = pre.image(Image.fromarray(img))
    m = pre.mask(Image.fromarray(mask))
    bright = x.mean(axis=0) > 0.5
    assert m.sum() > 0
    assert (bright == m).mean() > 0.98


def test_preprocess_rejects_crop_larger_than_resize():
    with pytest.raises(ValueError):
        Preprocess(resize=224, crop=256)
