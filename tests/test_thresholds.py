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


# ------------------------------------------------------------------ deployment calibration


def test_conformal_rank_and_validity_limits():
    dc = _load_script("deployment_calibration")
    assert dc.conformal_rank(9, 0.10) == 9 and dc.conformal_rank(8, 0.10) is None
    assert dc.conformal_rank(19, 0.05) == 19 and dc.conformal_rank(18, 0.05) is None
    assert dc.conformal_rank(99, 0.01) == 99
    assert dc.expected_conformal_fpr(9, 0.10) == pytest.approx(0.10)
    assert dc.expected_conformal_fpr(29, 0.10) == pytest.approx(3 / 30)
    assert dc.conformal_threshold(np.arange(1.0, 11.0), 0.10) == 10.0  # k = ceil(11 * 0.9)
    assert dc.conformal_rank(100 - 1, 1 - 0.99) == 99  # no float off-by-one


def test_deployment_conformal_hits_its_exact_expectation_and_training_fails_under_drift():
    """Exchangeable deployment calibration lands on (n+1-k)/(n+1); calibration on a
    drifted training session misses the target."""
    dc = _load_script("deployment_calibration")
    rng = np.random.default_rng(0)
    goods = rng.normal(1.10, 0.05, 40)  # test session
    defects = rng.normal(1.40, 0.10, 40)
    train_cal = rng.normal(1.00, 0.05, 25)  # training session scores lower: drift
    rows = dc.simulate(goods, defects, train_cal, [0.10], margin=0.0, repeats=4000, seed=1)
    by = {r["rule"]: r for r in rows}
    dep = by["deployment conformal"]
    assert dep["available"] and dep["n_calibration"] == 20
    assert dep["expected_fpr_if_exchangeable"] == pytest.approx(2 / 21)
    assert abs(dep["mean_fpr"] - 2 / 21) < 0.01
    assert not dep["exceeds_alpha"]
    assert by["training conformal"]["exceeds_alpha"]  # drift breaks the guarantee
    assert by["training conformal"]["mean_fpr"] > 0.5


def test_single_draw_p_value_separates_luck_from_drift():
    dc = _load_script("deployment_calibration")
    rng = np.random.default_rng(4)
    cal = rng.normal(1.0, 0.05, 25)
    same = [dc.single_draw_p_value(rng.normal(1.0, 0.05, 200), cal, 0.10) for _ in range(1)]
    drifted = dc.single_draw_p_value(rng.normal(1.1, 0.05, 200), cal, 0.10)
    assert same[0] > 0.01 and drifted < 1e-6
    assert dc.single_draw_p_value(rng.normal(1, 0.1, 20), cal[:6], 0.10) is None
    # Calibrated under exchangeability, p-values are roughly uniform (not too many small).
    ps = [dc.single_draw_p_value(rng.normal(1, 0.05, 60), rng.normal(1, 0.05, 25), 0.10)
          for _ in range(300)]  # fmt: skip
    assert np.mean(np.array(ps) < 0.05) < 0.10


def test_rules_report_not_available_when_calibration_is_too_small():
    dc = _load_script("deployment_calibration")
    rng = np.random.default_rng(2)
    rows = dc.simulate(rng.normal(1, 0.1, 12), rng.normal(2, 0.1, 5), rng.normal(1, 0.1, 6),
                       [0.10, 0.05], margin=0.13, repeats=50, seed=0)  # fmt: skip
    by = {(r["rule"], r["alpha"]): r for r in rows}
    assert not by[("deployment conformal", 0.10)]["available"]  # 6 < 9
    assert not by[("training conformal", 0.05)]["available"]  # 6 < 19
    assert by[("deployment max + margin", 0.10)]["available"]


def test_deployment_calibration_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    scores = tmp_path / "scores"
    scores.mkdir()
    rng = np.random.default_rng(3)
    for cat, drift in [("leather", 0.15), ("tile", 0.0), ("screw", 0.05)]:
        rows = [("calibration", 0, v) for v in rng.normal(1, 0.05, 24)]
        rows += [("test", 0, v) for v in rng.normal(1 + drift, 0.05, 32)]
        rows += [("test", 1, v) for v in rng.normal(1.6, 0.1, 40)]
        pd.DataFrame(rows, columns=["split", "label", "score"]).to_csv(
            scores / f"wrn50_{cat}.csv", index=False
        )
    dc = _load_script("deployment_calibration")
    out = tmp_path / "out"
    assert dc.main(["--scores", str(scores), "--out", str(out), "--repeats", "300"]) == 0
    df = pd.read_csv(out / "deployment_calibration_wrn50.csv")
    assert set(df.rule) == set(dc.RULES) and set(df.alpha) == {0.10, 0.05}
    lea = df[(df.category == "leather") & (df.alpha == 0.10)].set_index("rule")
    assert lea.loc["training conformal", "exceeds_alpha"]
    assert not lea.loc["deployment conformal", "exceeds_alpha"]
    assert df[df.category == "screw"].role.iloc[0] == "held_out (post-hoc)"
    assert (out / "deployment_calibration_wrn50.png").is_file()
