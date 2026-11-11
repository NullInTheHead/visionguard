from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exchangeability_row_detects_a_shift():
    analysis = _load_script("threshold_analysis")
    rng = np.random.default_rng(0)
    same = analysis.exchangeability_row(rng.normal(1, 0.1, 25), rng.normal(1, 0.1, 400))
    shifted = analysis.exchangeability_row(rng.normal(1, 0.1, 25), rng.normal(1.25, 0.1, 400))
    assert same["exceed_expected"] == pytest.approx(1 / 26)
    assert same["exceed_observed"] < 0.15 and same["mw_p_test_higher"] > 0.01
    assert shifted["exceed_observed"] > 0.3 and shifted["mw_p_test_higher"] < 1e-6
    assert shifted["median_shift_pct"] > 15


def test_threshold_analysis_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    scores = tmp_path / "scores"
    scores.mkdir()
    rng = np.random.default_rng(1)
    for cat, shift in [("leather", 0.2), ("tile", 0.0), ("carpet", 0.1)]:
        rows = [("calibration", 0, s) for s in rng.normal(1, 0.05, 20)]
        rows += [("test", 0, s) for s in rng.normal(1 + shift, 0.05, 30)]
        rows += [("test", 1, s) for s in rng.normal(2, 0.1, 30)]
        pd.DataFrame(rows, columns=["split", "label", "score"]).to_csv(
            scores / f"wrn50_{cat}.csv", index=False
        )
    analysis = _load_script("threshold_analysis")
    assert analysis.main(["--scores", str(scores), "--out", str(tmp_path / "out")]) == 0
    df = pd.read_csv(tmp_path / "out" / "exchangeability_wrn50.csv").set_index("category")
    assert df.loc["leather", "shift_significant"] and not df.loc["tile", "shift_significant"]
    assert df.loc["carpet", "role"] == "held_out (post-hoc)"
    assert (tmp_path / "out" / "exchangeability_wrn50.png").is_file()


@pytest.fixture()
def twins(tmp_path):
    """A development product (tile) whose training images come in near-identical pairs
    (000/001, 002/003, ...); test goods are new, independent images."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    root, size = tmp_path / "mvtec", 64
    yy, xx = np.mgrid[:size, :size]

    def scene():
        f, ph = rng.uniform(3, 8), rng.uniform(0, 6.3)
        base = 128 + 60 * np.sin(xx / f + ph) * np.cos(yy / (f + 2))
        return base

    def save(arr, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.clip(np.repeat(arr[..., None], 3, axis=2), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    cat = root / "tile"
    for i in range(0, 40, 2):
        base = scene()
        save(base + rng.normal(0, 1, base.shape), cat / "train" / "good" / f"{i:03d}.png")
        save(base + rng.normal(0, 1, base.shape), cat / "train" / "good" / f"{i + 1:03d}.png")
    for i in range(6):
        save(scene(), cat / "test" / "good" / f"{i:03d}.png")
        img = scene()
        img[20:30, 20:30] = 250
        save(img, cat / "test" / "crack" / f"{i:03d}.png")
        mask = np.zeros((size, size), np.uint8)
        mask[20:30, 20:30] = 255
        (cat / "ground_truth" / "crack").mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(cat / "ground_truth" / "crack" / f"{i:03d}_mask.png")
    return root


def test_calibration_diagnostic_sees_twins(twins, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    diag = _load_script("calibration_diagnostic")
    args = ["--root", str(twins), "--categories", "tile", "--backbone", "toy",
            "--no-pretrained", "--out", str(tmp_path)]  # fmt: skip
    assert diag.main(args) == 0
    df = pd.read_csv(tmp_path / "calibration_diagnostic_toy.csv").set_index("split")
    assert set(df.index) == {"random", "blocked"}
    assert (df.n_calibration == 4).all()
    # Random split: calibration images keep their twin in the bank, so they sit much closer
    # to it than test goods do, and score lower.
    r = df.loc["random"]
    assert r.nn_dist_calibration_median < 0.5 * r.nn_dist_test_good_median
    assert r.calibration_median < r.test_good_median
    # Blocked split holds back whole pairs, so the twin advantage mostly disappears.
    b = df.loc["blocked"]
    assert b.calibration_median > r.calibration_median
