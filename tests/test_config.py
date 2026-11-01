from __future__ import annotations

from pathlib import Path

import pytest

from visionguard.config import CategorySplit, load_split, load_yaml

REPO = Path(__file__).resolve().parents[1]


def test_committed_split_is_valid():
    split = load_split(REPO / "configs" / "split.yaml")
    assert len(split.dev) == 10 and len(split.held_out) == 5


def test_base_config_loads():
    cfg = load_yaml(REPO / "configs" / "base.yaml")
    assert 0.0 <= cfg["calibration"]["fraction"] < 1.0
    assert len(cfg["few_shot"]["seeds"]) >= 5


def test_split_rejects_overlap_and_missing():
    with pytest.raises(ValueError, match="both dev and held-out"):
        CategorySplit(dev=("bottle",), held_out=("bottle",)).validate()
    with pytest.raises(ValueError, match="all 15"):
        CategorySplit(dev=("bottle",), held_out=("carpet",)).validate()


def test_split_rejects_wrong_type_listing(tmp_path):
    bad = tmp_path / "split.yaml"
    bad.write_text("held_out:\n  textures: [screw]\n  objects: []\ndev: []\n")
    with pytest.raises(ValueError, match="has type 'object'"):
        load_split(bad)
