"""VisA support: preparation into the MVTec layout, the clean-test split, an E1 run."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from visionguard.config import load_config, load_split, resolve_categories
from visionguard.data import VISA_CATEGORIES, by_split, category_type, index_category

REPO = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def raw_visa(tmp_path):
    """A tiny raw VisA tree (JPEG images, multi-class masks) and its 1cls-style split file."""
    rng = np.random.default_rng(0)
    raw, rows = tmp_path / "VisA", []
    for cat in ("pcb1", "fryum"):
        img_dir = raw / cat / "Data" / "Images"
        for i in range(14):
            split = "train" if i < 10 else "test"
            path = img_dir / "Normal" / f"{i:04d}.JPG"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(90, 110, (48, 64, 3), dtype=np.uint8)).save(path)
            rows.append([cat, split, "normal", str(path.relative_to(raw)), ""])
        for i in range(4):
            path = img_dir / "Anomaly" / f"{i:03d}.JPG"
            path.parent.mkdir(parents=True, exist_ok=True)
            img = rng.integers(90, 110, (48, 64, 3), dtype=np.uint8)
            img[10:20, 10:20] = 250
            Image.fromarray(img).save(path)
            mask = np.zeros((48, 64), np.uint8)
            mask[10:20, 10:20] = 3  # VisA masks hold class ids, not 255
            mpath = raw / cat / "Data" / "Masks" / "Anomaly" / f"{i:03d}.png"
            mpath.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(mpath)
            rows.append([cat, "test", "anomaly", str(path.relative_to(raw)),
                         str(mpath.relative_to(raw))])  # fmt: skip
    split_file = tmp_path / "split.csv"
    with open(split_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "split", "label", "image", "mask"])
        w.writerows(rows)
    return raw, split_file


def test_prepare_visa_gives_the_mvtec_layout(raw_visa, tmp_path):
    raw, split_file = raw_visa
    prep = _load_script("prepare_visa")
    out = tmp_path / "visa"
    assert prep.main(["--raw", str(raw), "--out", str(out), "--split-file", str(split_file)]) == 0
    samples = index_category(out, "pcb1")
    assert len(by_split(samples, "train")) == 10
    test = by_split(samples, "test")
    assert sum(s.label for s in test) == 4 and len(test) == 8
    mask = np.array(Image.open(next(s.mask_path for s in test if s.label)))
    assert set(np.unique(mask)) == {0, 255}
    # Re-running is a no-op, not an error.
    assert prep.main(["--raw", str(raw), "--out", str(out), "--split-file", str(split_file)]) == 0


def test_official_split_file_matches_the_visa_paper():
    with open(REPO / "configs" / "visa_split_1cls.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10821
    assert {r["object"] for r in rows} == set(VISA_CATEGORIES)
    assert not [r for r in rows if r["split"] == "train" and r["label"] != "normal"]


def test_visa_split_is_all_clean_test():
    split = load_split(REPO / "configs" / "split_visa.yaml")
    assert split.dataset == "visa" and split.dev == ()
    assert resolve_categories("held_out", split) == list(split.held_out)
    assert len(split.held_out) == 12
    assert split.role("pcb1") == "clean test"
    assert category_type("pcb1") == "complex_structure"
    with pytest.raises(ValueError):
        resolve_categories(["screw"], split)  # an MVTec name is not a VisA category
    mvtec = load_split(REPO / "configs" / "split.yaml")
    assert mvtec.role("screw") == "held_out (post-hoc)" and mvtec.role("tile") == "dev"


def test_visa_config_keeps_every_frozen_setting():
    e1, visa = (
        load_config(REPO / "configs" / "e1.yaml"),
        load_config(REPO / "configs" / "e1_visa.yaml"),
    )
    for key in ("patchcore", "calibration", "backbone", "preprocess"):
        assert visa.get(key) == e1.get(key)
    assert visa["data"]["split"] == "configs/split_visa.yaml"


def test_run_e1_on_prepared_visa(raw_visa, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    raw, split_file = raw_visa
    out = tmp_path / "visa"
    _load_script("prepare_visa").main(["--raw", str(raw), "--out", str(out),
                                       "--split-file", str(split_file)])  # fmt: skip
    monkeypatch.chdir(REPO)
    run_e1 = _load_script("run_e1")
    args = ["--config", "configs/e1_visa.yaml", "--root", str(out), "--categories", "pcb1",
            "fryum", "--out", str(tmp_path / "e1"), "--backbone", "toy", "--no-pretrained",
            "--no-heatmaps"]  # fmt: skip
    assert run_e1.main(args) == 2  # every VisA category is a test category
    assert run_e1.main(args + ["--allow-held-out"]) == 0
    m = pd.read_csv(tmp_path / "e1" / "metrics.csv")
    assert set(m.category) == {"pcb1", "fryum"} and (m.type == "complex_structure").any()
    assert (m.n_calibration == 1).all()  # 10% of 10 training images
