"""E1: in-category anomaly detection with PatchCore (protocol section 5).

For each category: memory bank from its reference images, scored on its test set.

    python scripts/run_e1.py                                   # WRN-50, 10 dev categories
    python scripts/run_e1.py --categories bottle cable         # a subset
    python scripts/run_e1.py --calibration-fraction 0          # published protocol (anomalib check)
    python scripts/run_e1.py --backbone dinov2                 # Phase 3

Outputs (under --out, default results/e1):
    metrics.csv                         one row per (backbone, category, calibration fraction)
    scores/<backbone>_<category>.csv    per-image scores, test and calibration images
    heatmaps/<backbone>_<category>.png  top true positives, false positives and misses
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from visionguard.config import load_config, load_split, resolve_categories  # noqa: E402
from visionguard.data import by_split, calibration_split, category_type, index_category  # noqa: E402
from visionguard.experiment import (  # noqa: E402
    evaluate,
    heatmap_grid,
    run_metadata,
    scores_frame,
    upsert_rows,
)


def parse_args(argv):
    p = argparse.ArgumentParser(description="E1: in-category detection with PatchCore.")
    p.add_argument("--config", default="configs/e1.yaml")
    p.add_argument("--backbone")
    p.add_argument("--categories", nargs="*", help="names, or one of: dev, held_out, all")
    p.add_argument("--root", help="MVTec AD folder (overrides the config)")
    p.add_argument("--out")
    p.add_argument("--device")
    p.add_argument("--calibration-fraction", type=float)
    p.add_argument("--coreset-ratio", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument(
        "--allow-held-out",
        action="store_true",
        help="required to run on held-out categories (only with frozen settings)",
    )
    p.add_argument("--no-heatmaps", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # tests
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    pc = cfg["patchcore"]
    backbone = args.backbone or cfg["backbone"]
    root = args.root or cfg["data"]["root"]
    out = Path(args.out or cfg["output"])
    device_name = args.device or cfg.get("device", "auto")
    cal_fraction = (
        args.calibration_fraction
        if args.calibration_fraction is not None
        else cfg["calibration"]["fraction"]
    )
    coreset_ratio = args.coreset_ratio if args.coreset_ratio is not None else pc["coreset_ratio"]

    split = load_split(cfg["data"]["split"])
    categories = resolve_categories(args.categories or cfg["categories"], split)
    held = [c for c in categories if c in split.held_out]
    if held and not args.allow_held_out:
        print(
            f"Refusing to run on held-out categories {held} without --allow-held-out. "
            "Held-out categories are only evaluated once settings are frozen (protocol 2.1).",
            file=sys.stderr,
        )
        return 2

    # Apple GPUs (MPS) lack a few operations; run those on the CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch  # heavy imports after argument checks

    from visionguard.features import FeatureExtractor, pick_device
    from visionguard.patchcore import PatchCore

    device = pick_device(device_name)
    extractor = FeatureExtractor(backbone, pretrained=not args.no_pretrained)
    model = PatchCore(
        extractor,
        device=device,
        coreset_ratio=coreset_ratio,
        num_neighbors=pc["num_neighbors"],
        sigma=pc["sigma"],
        batch_size=args.batch_size or pc["batch_size"],
        num_workers=args.num_workers if args.num_workers is not None else pc["num_workers"],
    )
    meta = run_metadata(str(device))
    print(f"E1 | backbone={backbone} device={meta['device']} categories={len(categories)}")

    for category in categories:
        torch.manual_seed(0)
        samples = index_category(root, category)
        reference, calibration = calibration_split(
            by_split(samples, "train"), cal_fraction, cfg["calibration"]["seed"]
        )
        test = by_split(samples, "test")

        model.fit(reference)
        pred = model.predict(test)
        metrics = evaluate(pred.labels, pred.scores, pred.scores_max, pred.masks, pred.maps)

        frames = [scores_frame(pred, "test")]
        cal_max = float("nan")
        if calibration:
            cal_pred = model.predict(calibration)
            cal_max = float(cal_pred.scores.max())
            frames.append(scores_frame(cal_pred, "calibration"))
        tag = f"{backbone}_{category}" + ("" if cal_fraction else "_fullref")
        (out / "scores").mkdir(parents=True, exist_ok=True)
        pd.concat(frames).to_csv(out / "scores" / f"{tag}.csv", index=False)

        row = {
            "experiment": "e1",
            "backbone": backbone,
            "category": category,
            "type": category_type(category),
            "role": "held_out" if category in split.held_out else "dev",
            "calibration_fraction": cal_fraction,
            "n_reference": len(reference),
            "n_calibration": len(calibration),
            "n_test": len(test),
            "coreset_ratio": coreset_ratio,
            "bank_patches": int(model.memory_bank.shape[0]),
            "bank_mb": round(model.memory_bank_mb, 2),
            **{k: round(v, 5) for k, v in metrics.items()},
            "calibration_max_score": round(cal_max, 5),
            "fit_seconds": round(model.fit_seconds, 2),
            "seconds_per_image": round(pred.seconds_per_image, 4),
            **meta,
            "config": args.config,
        }
        upsert_rows(
            out / "metrics.csv", [row], key=["backbone", "category", "calibration_fraction"]
        )
        if not args.no_heatmaps:
            heatmap_grid(
                test,
                pred,
                extractor.preprocess,
                out / "heatmaps" / f"{tag}.png",
                per_row=cfg.get("heatmaps", {}).get("per_row", 4),
                title=f"{category} | {backbone} | image AUROC {metrics['image_auroc']:.3f}",
            )
        print(
            f"  {category:<11} image AUROC {metrics['image_auroc']:.4f}  "
            f"pixel AUROC {metrics['pixel_auroc']:.4f}  AU-PRO {metrics['au_pro']:.4f}  "
            f"FPR@95 {metrics['fpr_at_95_recall']:.3f}  "
            f"({model.fit_seconds:.0f}s fit, {1000 * pred.seconds_per_image:.0f} ms/img)"
        )

    results = pd.read_csv(out / "metrics.csv")
    mine = results[
        (results.backbone == backbone)
        & (results.category.isin(categories))
        & (results.calibration_fraction == cal_fraction)
    ]
    cols = ["image_auroc", "pixel_auroc", "au_pro", "fpr_at_95_recall"]
    print("\nMean over categories:\n" + mine[cols].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
