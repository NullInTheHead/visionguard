from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from visionguard.banks import (  # noqa: E402
    category_coreset,
    few_shot_draws,
    reference_pool,
    shared_bank,
)
from visionguard.features import FeatureExtractor  # noqa: E402
from visionguard.patchcore import PatchCore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def two_products(fake_mvtec, tmp_path):
    """screw (held-out target) plus a copy of carpet named tile (a development category)."""
    root = tmp_path / "mvtec"
    shutil.copytree(fake_mvtec / "screw", root / "screw")
    shutil.copytree(fake_mvtec / "carpet", root / "tile")
    return root


def _model():
    return PatchCore(FeatureExtractor("toy"), coreset_ratio=0.5, batch_size=4)


def test_few_shot_draws_are_nested_and_from_the_pool(two_products):
    reference, calibration, test = reference_pool(two_products, "screw", 0.2, 0)
    draws = few_shot_draws(reference, ks=[1, 2, 4], seeds=[0, 1])
    for seed in (0, 1):
        assert draws[(2, seed)][:1] == draws[(1, seed)]
        assert draws[(4, seed)][:2] == draws[(2, seed)]
    used = {s for d in draws.values() for s in d}
    assert used <= set(reference)
    assert used.isdisjoint(calibration) and used.isdisjoint(test)


def test_shared_bank_never_contains_the_target(two_products, tmp_path):
    model = _model()
    bank, sources = shared_bank(model, "toy", two_products, ["tile"], "screw", 0.5, 0.2, 0,
                                cache_dir=tmp_path / "cache")  # fmt: skip
    assert sources == ["tile"]
    tile_only = category_coreset(model, "toy", two_products, "tile", 0.5, 0.2, 0,
                                 cache_dir=None)  # fmt: skip
    assert torch.equal(bank, tile_only)
    with pytest.raises(ValueError, match="other than the target"):
        shared_bank(model, "toy", two_products, ["tile"], "tile", 0.5, 0.2, 0, cache_dir=None)


def test_coreset_cache_is_reused_and_keyed_by_settings(two_products, tmp_path):
    model, cache = _model(), tmp_path / "cache"
    first = category_coreset(model, "toy", two_products, "tile", 0.5, 0.2, 0, cache)
    files = list(cache.iterdir())
    assert len(files) == 1 and "r0.5" in files[0].name and "cal0.2" in files[0].name
    torch.save(torch.zeros_like(first), files[0])  # a cached bank must be read, not rebuilt
    assert torch.equal(category_coreset(model, "toy", two_products, "tile", 0.5, 0.2, 0, cache),
                       torch.zeros_like(first))  # fmt: skip
    category_coreset(model, "toy", two_products, "tile", 0.25, 0.2, 0, cache)
    assert len(list(cache.iterdir())) == 2  # a different ratio is a different file


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_e2_end_to_end(two_products, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    out = tmp_path / "e2"
    args = ["--root", str(two_products), "--categories", "screw", "--bank-categories", "tile",
            "--backbone", "toy", "--no-pretrained", "--out", str(out), "--ks", "1", "2",
            "--seeds", "0", "1", "--no-cache"]  # fmt: skip
    assert _load_script("run_e2").main(args) == 0
    df = pd.read_csv(out / "metrics.csv")
    assert len(df) == 1 + 2 * 2
    zero = df[df.experiment == "e2a"].iloc[0]
    assert zero.k == 0 and zero.bank_categories == "tile" and zero.role == "held_out"
    few = df[df.experiment == "e2b"]
    assert set(few.k) == {1, 2} and set(few.seed) == {0, 1}
    assert (few[few.k == 1].bank_patches == 49).all()  # one 7x7 image, every patch kept
    assert {"ms_features", "ms_nn_search", "peak_rss_mb"} <= set(df.columns)
    scores = pd.read_csv(out / "scores" / "toy_screw.csv")
    assert len(scores) == 5 * 4  # 5 banks x 4 test images
    assert (out / "recovery_toy_image_auroc.png").is_file()
    assert not (REPO / "cache").exists()  # --no-cache writes nothing


def test_run_e2_refuses_a_held_out_bank_source(two_products, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    args = ["--root", str(two_products), "--categories", "screw", "--bank-categories",
            "carpet", "--backbone", "toy", "--no-pretrained", "--out", str(tmp_path)]  # fmt: skip
    assert _load_script("run_e2").main(args) == 2


# ------------------------------------------------------------------ E2c rotations


def test_valid_patch_mask():
    from visionguard.banks import valid_patch_mask

    assert valid_patch_mask(224, (28, 28), 0).all()
    assert valid_patch_mask(224, (28, 28), 90).all()
    m = valid_patch_mask(224, (28, 28), 45).reshape(28, 28)
    assert not m[0, 0] and not m[27, 27] and m[14, 14]
    assert 0.3 < m.float().mean() < 0.9


def test_rotated_references_contain_the_unrotated_ones(two_products):
    from visionguard.banks import ROTATIONS, embed_references, embed_references_rotated

    model = _model()
    reference, _, _ = reference_pool(two_products, "screw", 0.2, 0)
    plain = embed_references(model, reference[:2])
    rot90 = embed_references_rotated(model, reference[:2], ROTATIONS["rot90"])
    rot45 = embed_references_rotated(model, reference[:2], ROTATIONS["rot45"])
    for i, e in plain.items():
        assert rot90[i].shape == (4 * 49, 1024)
        torch.testing.assert_close(rot90[i][:49], e)  # angle 0 comes first, unchanged
        assert 49 < rot45[i].shape[0] < 8 * 49  # corner patches of 45-degree turns dropped


@pytest.fixture()
def bars(tmp_path):
    """Good parts: a dark bar, horizontal or vertical. Defects: a faint mark on the bar.
    Training images are all horizontal, so few-shot references never show a vertical bar."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    root, size = tmp_path / "bars", 64

    def part(vertical: bool, defect: bool) -> np.ndarray:
        img = 150 + rng.normal(0, 3, (size, size))
        if vertical:
            img[:, 28:36] = 40
        else:
            img[28:36, :] = 40
        mask = np.zeros((size, size), bool)
        if defect:
            r, c = (rng.integers(12, 44), 32) if vertical else (32, rng.integers(12, 44))
            mask[r - 4 : r + 4, c - 4 : c + 4] = True
            img[mask] = 75  # faint: subtler than a change of pose, as on real screws
        rgb = np.clip(np.repeat(img[..., None], 3, axis=2), 0, 255).astype(np.uint8)
        return rgb, mask

    def save(arr, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)

    cat = root / "screw"
    for i in range(10):
        save(part(False, False)[0], cat / "train" / "good" / f"{i:03d}.png")
    for i in range(8):
        save(part(i % 2 == 1, False)[0], cat / "test" / "good" / f"{i:03d}.png")
        img, mask = part(i % 2 == 1, True)
        save(img, cat / "test" / "blob" / f"{i:03d}.png")
        save((mask * 255).astype("uint8"), cat / "ground_truth" / "blob" / f"{i:03d}_mask.png")
    return root


def test_rotation_bank_fixes_pose_on_a_synthetic_part(bars, tmp_path, monkeypatch):
    """The mechanism the diagnostic tests: unseen pose looks anomalous; rotations fix it."""
    monkeypatch.chdir(REPO)
    out = tmp_path / "e2"
    common = ["--root", str(bars), "--categories", "screw", "--backbone", "toy",
              "--no-pretrained", "--out", str(out), "--ks", "1", "--seeds", "0", "1",
              "--skip-zero-shot", "--no-cache"]  # fmt: skip
    run_e2 = _load_script("run_e2")
    assert run_e2.main(common) == 0
    assert run_e2.main(common + ["--rotations", "rot90"]) == 0
    df = pd.read_csv(out / "metrics.csv")
    plain = df[df.experiment == "e2b"].image_auroc.mean()
    rotated = df[df.experiment == "e2c-rot90"].image_auroc.mean()
    assert rotated > plain + 0.2, (plain, rotated)
    assert (df[df.experiment == "e2c-rot90"].bank_patches == 4 * 49).all()
    assert (df[df.experiment == "e2c-rot90"].rotations == "rot90").all()
    # E2c writes its own scores file and leaves the E2b one in place.
    assert (out / "scores" / "toy_screw.csv").is_file()
    assert (out / "scores" / "toy_screw_rot90.csv").is_file()
    assert (out / "rotation_toy_image_auroc.png").is_file()
