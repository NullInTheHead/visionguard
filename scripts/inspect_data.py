"""Phase 1: inventory MVTec AD and save tables and sample figures.

Usage:
    python scripts/inspect_data.py --root data/mvtec_anomaly_detection --out results/data

Outputs:
    category_stats.csv     counts, image sizes, defect sizes, crop losses per category
    defect_type_stats.csv  the same broken down by defect type
    defect_images.csv      one row per defective test image (used later for failure analysis)
    samples/<category>.png normal and defective examples, mask in red, crop region in yellow
    overview.png           one normal and one defective example per category
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from visionguard.data import CATEGORIES, index_category  # noqa: E402
from visionguard.data_report import (  # noqa: E402
    DEFAULT_CROP_KEEP,
    dataset_report,
    plot_category,
    plot_overview,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--out", default="results/data")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--crop-keep", type=float, default=DEFAULT_CROP_KEEP)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_category, per_defect, per_image = dataset_report(
        args.root, args.categories, crop_keep=args.crop_keep
    )
    per_category.to_csv(out / "category_stats.csv", index=False)
    per_defect.to_csv(out / "defect_type_stats.csv", index=False)
    per_image.to_csv(out / "defect_images.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = [
            "category",
            "type",
            "n_train",
            "n_test_good",
            "n_test_defect",
            "n_defect_types",
            "median_defect_area_pct",
            "n_tiny_defects",
            "n_crop_retention_lt50",
            "counts_match_published",
        ]
        print(per_category[cols].round(3).to_string(index=False))
        print(
            f"\nTotal: {per_category.n_train.sum()} train, "
            f"{(per_category.n_test_good + per_category.n_test_defect).sum()} test images; "
            f"{per_category.n_tiny_defects.sum()} tiny defects; "
            f"{per_category.n_crop_retention_lt50.sum()} defects lose >50% to the centre crop."
        )

    mismatched = per_category[per_category.counts_match_published == False]  # noqa: E712
    if len(mismatched):
        print(
            "\nWARNING: counts differ from the published dataset for: "
            + ", ".join(mismatched.category)
            + ". Check for a partial download.",
            file=sys.stderr,
        )

    if not args.no_figures:
        by_cat = {c: index_category(args.root, c) for c in args.categories}
        for cat, samples in by_cat.items():
            plot_category(samples, out / "samples" / f"{cat}.png", seed=args.seed)
        plot_overview(by_cat, out / "overview.png", seed=args.seed)
        print(f"\nFigures written to {out / 'samples'} and {out / 'overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
