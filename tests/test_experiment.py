from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visionguard.config import load_config, load_split, resolve_categories
from visionguard.experiment import evaluate, upsert_rows

REPO = Path(__file__).resolve().parents[1]


def test_load_config_follows_extends():
    cfg = load_config(REPO / "configs" / "e1.yaml")
    assert cfg["experiment"] == "e1"
    assert cfg["calibration"]["fraction"] == 0.1  # inherited from base.yaml
    assert cfg["patchcore"]["coreset_ratio"] == 0.1
    assert "extends" not in cfg


def test_resolve_categories():
    split = load_split(REPO / "configs" / "split.yaml")
    assert resolve_categories("dev", split) == list(split.dev)
    assert len(resolve_categories("all", split)) == 15
    assert resolve_categories(["screw"], split) == ["screw"]
    assert resolve_categories("bottle", split) == ["bottle"]
    assert resolve_categories(["held_out"], split) == list(split.held_out)  # from argparse
    assert resolve_categories(["dev"], split) == list(split.dev)
    with pytest.raises(ValueError):
        resolve_categories(["banana"], split)


def test_upsert_rows_replaces_same_key(tmp_path):
    path = tmp_path / "m.csv"
    upsert_rows(path, [{"b": "wrn50", "c": "bottle", "v": 1}], key=["b", "c"])
    upsert_rows(path, [{"b": "wrn50", "c": "cable", "v": 2}], key=["b", "c"])
    df = upsert_rows(path, [{"b": "wrn50", "c": "bottle", "v": 3}], key=["b", "c"])
    assert len(df) == 2
    assert pd.read_csv(path).set_index("c").loc["bottle", "v"] == 3


def test_evaluate_returns_all_protocol_metrics():
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    masks = np.zeros((6, 8, 8), bool)
    masks[3:, 2:5, 2:5] = True
    maps = rng.random((6, 8, 8)) + masks
    m = evaluate(labels, scores, scores, masks, maps)
    assert m["image_auroc"] == 1.0 and m["fpr_at_95_recall"] == 0.0
    assert set(m) >= {"pixel_auroc", "au_pro", "rejects_per_1000_at_95_recall"}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_e1_refuses_held_out_categories(fake_mvtec, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    run_e1 = _load_script("run_e1")
    rc = run_e1.main(["--root", str(fake_mvtec), "--categories", "screw", "--out", str(tmp_path)])
    assert rc == 2
    assert not (tmp_path / "metrics.csv").exists()


def test_run_e1_end_to_end_with_toy_backbone(fake_mvtec, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.chdir(REPO)
    run_e1 = _load_script("run_e1")
    args = ["--root", str(fake_mvtec), "--categories", "carpet", "--allow-held-out",
            "--backbone", "toy", "--no-pretrained", "--out", str(tmp_path),
            "--calibration-fraction", "0.34"]  # fmt: skip
    assert run_e1.main(args) == 0
    df = pd.read_csv(tmp_path / "metrics.csv")
    row = df.iloc[0]
    assert (row.category, row.role, row.n_reference, row.n_calibration) == (
        "carpet", "held_out", 4, 2)  # fmt: skip
    assert 0.0 <= row.image_auroc <= 1.0 and np.isfinite(row.calibration_max_score)
    scores = pd.read_csv(tmp_path / "scores" / "toy_carpet.csv")
    assert (scores.split == "calibration").sum() == 2 and (scores.split == "test").sum() == 5
    assert (tmp_path / "heatmaps" / "toy_carpet.png").is_file()
    assert (tmp_path / "examples" / "toy_carpet.png").is_file()


def test_choose_margin_picks_smallest_margin_meeting_target(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    split = load_split(REPO / "configs" / "split.yaml")
    scores = tmp_path / "scores"
    scores.mkdir()
    for category in split.dev:
        # Calibration max = 1.0; good test parts up to 1.08; defects 1.2 to 2.0.
        rows = [("calibration", 0, 1.0)] + [("calibration", 0, 0.5)]
        rows += [("test", 0, s) for s in np.linspace(0.6, 1.08, 10)]
        rows += [("test", 1, s) for s in np.linspace(1.2, 2.0, 10)]
        pd.DataFrame(rows, columns=["split", "label", "score"]).to_csv(
            scores / f"wrn50_{category}.csv", index=False
        )
    choose = _load_script("choose_margin")
    assert choose.main(["--scores", str(scores), "--out", str(tmp_path)]) == 0
    table = pd.read_csv(tmp_path / "threshold_wrn50.csv")
    # Smallest grid margin with threshold above 1.08 is 0.09 -> FPR 0, recall 1.
    assert table.margin.iloc[0] == pytest.approx(0.09)
    assert (table.fpr == 0).all() and (table.recall == 1).all()


def test_apply_threshold_uses_frozen_margin(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    scores = tmp_path / "scores"
    scores.mkdir()
    rows = [("calibration", 0, 1.0), ("test", 0, 1.10), ("test", 0, 1.20)]
    rows += [("test", 1, 1.05), ("test", 1, 2.0)]
    pd.DataFrame(rows, columns=["split", "label", "score"]).to_csv(
        scores / "wrn50_screw.csv", index=False
    )
    apply = _load_script("apply_threshold")
    args = ["--categories", "screw", "--scores", str(scores), "--out", str(tmp_path)]
    assert apply.main(args) == 0
    row = pd.read_csv(tmp_path / "threshold_wrn50_screw.csv").iloc[0]
    margin = load_config(REPO / "configs" / "base.yaml")["calibration"]["margin"]
    assert row.margin == margin == 0.13  # read from the config, not re-tuned
    # threshold 1.13: good 1.20 is flagged (FPR 0.5), defect 1.05 is missed (recall 0.5)
    assert (row.fpr, row.recall, row.role) == (0.5, 0.5, "held_out")
