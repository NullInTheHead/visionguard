"""Choose the deployable-threshold margin m on development categories (protocol section 6).

Threshold per category = (1 + m) x the maximum image score over its calibration images
(defect-free training images that never entered the memory bank). m is the smallest value
on a grid whose mean false-reject rate over the development categories' defect-free test
images is at most the target (1%). The chosen m is then frozen for every later experiment.

    python scripts/choose_margin.py                      # WRN-50 E1 scores
    python scripts/choose_margin.py --backbone dinov2

Writes results/e1/threshold_<backbone>.csv (per category, at the chosen m) and
results/e1/margin_sweep_<backbone>.csv (mean FPR and recall for every m on the grid).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from visionguard.config import load_split
from visionguard.metrics import operating_point_at


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backbone", default="wrn50")
    p.add_argument("--scores", default="results/e1/scores")
    p.add_argument("--split", default="configs/split.yaml")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--out", default="results/e1")
    args = p.parse_args(argv)

    split = load_split(args.split)
    per_cat = {}
    for category in split.dev:
        path = Path(args.scores) / f"{args.backbone}_{category}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/run_e1.py first.")
        df = pd.read_csv(path)
        cal = df[df.split == "calibration"]
        if cal.empty:
            raise ValueError(f"{path} has no calibration scores (run with calibration on).")
        per_cat[category] = (df[df.split == "test"], float(cal.score.max()))

    grid = np.round(np.arange(0.0, 0.501, 0.01), 2)
    sweep = []
    for m in grid:
        ops = [
            operating_point_at(test.label, test.score, (1 + m) * cal_max)
            for test, cal_max in per_cat.values()
        ]
        sweep.append(
            {
                "margin": m,
                "mean_fpr": np.mean([o.fpr for o in ops]),
                "max_fpr": np.max([o.fpr for o in ops]),
                "mean_recall": np.mean([o.recall for o in ops]),
                "min_recall": np.min([o.recall for o in ops]),
            }
        )
    sweep = pd.DataFrame(sweep)
    ok = sweep[sweep.mean_fpr <= args.target_fpr]
    if ok.empty:
        raise SystemExit("No margin on the grid reaches the target false-reject rate.")
    m = float(ok.margin.iloc[0])

    rows = []
    for category, (test, cal_max) in per_cat.items():
        op = operating_point_at(test.label, test.score, (1 + m) * cal_max)
        rows.append(
            {
                "backbone": args.backbone,
                "category": category,
                "margin": m,
                "calibration_max": round(cal_max, 5),
                "threshold": round(op.threshold, 5),
                "recall": round(op.recall, 4),
                "fpr": round(op.fpr, 4),
                "rejects_per_1000": round(op.false_rejects_per_1000, 1),
            }
        )
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"threshold_{args.backbone}.csv", index=False)
    sweep.round(4).to_csv(out / f"margin_sweep_{args.backbone}.csv", index=False)

    print(f"Chosen margin m = {m:.2f} (smallest with mean FPR <= {args.target_fpr:.0%})\n")
    print(table[["category", "recall", "fpr", "rejects_per_1000"]].to_string(index=False))
    print(
        f"\nMean recall {table.recall.mean():.3f}, mean false rejects "
        f"{table.rejects_per_1000.mean():.1f} per 1,000 good parts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
