"""RQ2, step 2: why do defect-free test images score higher than calibration images?

Two explanations predict different things (protocol section 11.1):

* **Near-twins.** Training images shot in sequence can have near-identical neighbours. A
  randomly held-back calibration image then keeps its twin in the memory bank and scores
  artificially low; test images have no twin. Prediction: holding back a *contiguous
  block* of training images (by file number) raises calibration scores towards the test
  level, and calibration images sit much closer to their nearest bank image than test
  images do.
* **Session drift.** Test images were captured under slightly different conditions.
  Prediction: the blocked split changes little, and test images are farther from the
  bank than *any* training image, whichever way the split is made.

For each category, both splits hold back the same number of images (10%) and build a
10% coreset bank from the rest, as in E1. Per split it reports the share of defect-free
test images above the calibration maximum, the median score shift, and the median
distance from each calibration / test image to its nearest bank *image* (cosine distance
between mean-pooled patch embeddings).

    python scripts/calibration_diagnostic.py                 # 10 development categories
    python scripts/calibration_diagnostic.py --categories leather zipper

Writes results/thresholds/calibration_diagnostic_<backbone>.csv.
Assumes file numbers follow capture order; if that is false, the blocked split is just
another random split, and the nearest-image distances carry the evidence.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy import stats

from visionguard.config import load_config, load_split, resolve_categories
from visionguard.data import by_split, calibration_split, index_category
from visionguard.experiment import run_metadata, upsert_rows


def blocked_split(train, n_cal: int):
    """Hold back the last ``n_cal`` images by file number; the rest are the reference set."""
    ordered = sorted(train, key=lambda s: s.path.stem)
    return ordered[:-n_cal], ordered[-n_cal:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/e1.yaml")
    p.add_argument("--backbone")
    p.add_argument("--categories", nargs="*", default=["dev"])
    p.add_argument("--root")
    p.add_argument("--out", default="results/thresholds")
    p.add_argument("--device")
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # tests
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    pc = cfg["patchcore"]
    backbone = args.backbone or cfg["backbone"]
    root = args.root or cfg["data"]["root"]
    fraction, seed = cfg["calibration"]["fraction"], cfg["calibration"]["seed"]
    split = load_split(cfg["data"]["split"])
    categories = resolve_categories(args.categories, split)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    import torch.nn.functional as F

    from visionguard.features import FeatureExtractor, pick_device
    from visionguard.patchcore import PatchCore

    device = pick_device(args.device or cfg.get("device", "auto"))
    model = PatchCore(
        FeatureExtractor(backbone, pretrained=not args.no_pretrained),
        device=device,
        coreset_ratio=pc["coreset_ratio"],
        num_neighbors=pc["num_neighbors"],
        sigma=pc["sigma"],
        batch_size=pc["batch_size"],
        num_workers=pc["num_workers"],
    )
    meta = run_metadata(str(device))
    out = Path(args.out)
    print(f"Calibration diagnostic | backbone={backbone} device={meta['device']}")

    for category in categories:
        torch.manual_seed(0)
        samples = index_category(root, category)
        train = sorted(by_split(samples, "train"))
        goods = [s for s in by_split(samples, "test") if s.label == 0]
        train_set = model.embed_for_scoring(train)  # every training image, embedded once
        good_set = model.embed_for_scoring(goods)
        position = {sid: i for i, sid in enumerate(train_set.ids)}
        global_train = F.normalize(train_set.embeddings.mean(dim=1), dim=1)
        global_good = F.normalize(good_set.embeddings.mean(dim=1), dim=1)

        random_ref, random_cal = calibration_split(train, fraction, seed)
        splits = {
            "random": (random_ref, random_cal),
            "blocked": blocked_split(train, len(random_cal)),
        }
        rows = []
        for name, (ref, cal) in splits.items():
            ref_idx = [position[s.id] for s in ref]
            cal_idx = [position[s.id] for s in cal]
            patches = train_set.embeddings[ref_idx].flatten(0, 1)
            model.set_memory_bank(model.subsample(patches, pc["coreset_ratio"]))
            cal_scores = model.score_embedded(train_set.subset(cal_idx)).scores
            good_scores = model.score_embedded(good_set).scores

            bank_global = global_train[ref_idx]
            nn_cal = 1 - (global_train[cal_idx] @ bank_global.T).max(dim=1).values
            nn_good = 1 - (global_good @ bank_global.T).max(dim=1).values
            rows.append(
                {
                    "backbone": backbone,
                    "category": category,
                    "role": split.role(category),
                    "split": name,
                    "n_calibration": len(cal),
                    "exceed_observed": round(float((good_scores > cal_scores.max()).mean()), 4),
                    "exceed_expected": round(1 / (len(cal) + 1), 4),
                    "calibration_median": round(float(np.median(cal_scores)), 4),
                    "test_good_median": round(float(np.median(good_scores)), 4),
                    "median_shift_pct": round(
                        float(100 * (np.median(good_scores) / np.median(cal_scores) - 1)), 2
                    ),
                    "mw_p_test_higher": round(
                        float(
                            stats.mannwhitneyu(
                                good_scores, cal_scores, alternative="greater"
                            ).pvalue
                        ),
                        5,
                    ),  # fmt: skip
                    "nn_dist_calibration_median": round(float(nn_cal.median()), 6),
                    "nn_dist_test_good_median": round(float(nn_good.median()), 6),
                    **meta,
                }
            )
        upsert_rows(out / f"calibration_diagnostic_{backbone}.csv", rows,
                    key=["backbone", "category", "split"])  # fmt: skip
        r, b = rows
        print(
            f"  {category:<11} shift random {r['median_shift_pct']:+5.1f}%  blocked "
            f"{b['median_shift_pct']:+5.1f}% | above cal max: random {r['exceed_observed']:.0%} "
            f"blocked {b['exceed_observed']:.0%} | nearest-image distance cal/test "
            f"{r['nn_dist_calibration_median']:.4f}/{r['nn_dist_test_good_median']:.4f}"
        )
        del train_set, good_set
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
