from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from visionguard.data_report import centre_box, dataset_report

REPO = Path(__file__).resolve().parents[1]


def test_centre_box_matches_256_to_224_crop():
    assert centre_box(256, 256, 224 / 256) == (16, 240, 16, 240)


def test_report_counts_and_crop_losses(fake_mvtec):
    per_cat, per_defect, per_image = dataset_report(fake_mvtec, ["carpet", "screw"])
    carpet = per_cat.set_index("category").loc["carpet"]
    assert carpet.n_train == 6 and carpet.n_test_good == 2 and carpet.n_test_defect == 3
    assert carpet.defect_types == "cut;hole"
    # Synthetic sizes differ from the real dataset, so the download check must fail.
    assert not carpet.counts_match_published
    # carpet/cut/000 lies entirely outside the centre crop.
    corner = per_image.set_index("id").loc["carpet/test/cut/000"]
    assert corner.crop_retention == 0.0
    assert carpet.n_crop_retention_0 == 1
    assert per_cat.set_index("category").loc["screw"].modes == "L"
    assert set(per_defect.columns) >= {"category", "defect_type", "n_images", "n_tiny"}


def test_inspect_script_end_to_end(fake_mvtec, tmp_path):
    spec = importlib.util.spec_from_file_location("inspect_data", REPO / "scripts/inspect_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "report"
    rc = module.main(
        ["--root", str(fake_mvtec), "--out", str(out), "--categories", "carpet", "screw"]
    )
    assert rc == 0
    for name in [
        "category_stats.csv",
        "defect_type_stats.csv",
        "defect_images.csv",
        "overview.png",
    ]:
        assert (out / name).is_file(), name
    assert (out / "samples" / "screw.png").is_file()
    assert len(pd.read_csv(out / "category_stats.csv")) == 2


@pytest.mark.parametrize("keep", [1.0, 0.5])
def test_full_keep_retains_everything(fake_mvtec, keep):
    _, _, per_image = dataset_report(fake_mvtec, ["screw"], crop_keep=keep)
    if keep == 1.0:
        assert (per_image.crop_retention == 1.0).all()
    else:
        assert (per_image.crop_retention <= 1.0).all()
