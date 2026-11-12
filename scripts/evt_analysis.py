"""RQ2, step 5: extreme-value tail fit on training calibration images (protocol 11.4b).

From the calibration-size scores (scripts/calibration_size.py): for each category,
calibration fraction and seed, training EVT (tail fractions 0.3, plus 0.2 and 0.5 as a
sensitivity check) next to training split conformal, at alpha = 10%, 5% and 1%. Reports
the false-reject rate on all defect-free test images and recall.

    python scripts/evt_analysis.py
Writes results/thresholds/evt_training_<backbone>.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from visionguard.config import load_split

_spec = importlib.util.spec_from_file_location(
    "deployment_calibration", Path(__file__).with_name("deployment_calibration.py")
)
_dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dc)

TAILS = [0.3, 0.2, 0.5]  # 0.3 is the rule; the others are the sensitivity check


def rows_for(cal, goods, defects, alphas, tails) -> list[dict]:
    rows = []
    fits = {tail: _dc.evt_fit(cal, tail) for tail in tails}
    for alpha in alphas:
        rules = {"conformal": _dc.conformal_threshold(cal, alpha)}
        for tail in tails:
            rules[f"evt tail {tail}"] = _dc.evt_threshold(fits[tail], alpha)
        for rule, t in rules.items():
            row = {"alpha": alpha, "rule": rule, "available": t is not None}
            if t is not None:
                row |= {"threshold": t, "fpr": float((goods > t).mean()),
                        "recall": float((defects > t).mean())}  # fmt: skip
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backbone", default="wrn50")
    p.add_argument("--scores", default="results/thresholds/calibration_size_scores")
    p.add_argument("--split", default="configs/split.yaml")
    p.add_argument("--out", default="results/thresholds")
    p.add_argument("--alphas", nargs="*", type=float, default=[0.10, 0.05, 0.01])
    args = p.parse_args(argv)

    split = load_split(args.split)
    rows = []
    for path in sorted(Path(args.scores).glob(f"{args.backbone}_*.csv")):
        category = path.stem[len(args.backbone) + 1 :]
        s = pd.read_csv(path)
        for (fraction, seed), d in s.groupby(["calibration_fraction", "seed"]):
            cal = d[d.split == "calibration"].score.to_numpy()
            test = d[d.split == "test"]
            goods = test[test.label == 0].score.to_numpy()
            defects = test[test.label == 1].score.to_numpy()
            for r in rows_for(cal, goods, defects, args.alphas, TAILS):
                rows.append({"backbone": args.backbone, "category": category,
                             "role": "dev" if category in split.dev else "held_out (post-hoc)",
                             "calibration_fraction": fraction, "seed": seed,
                             "n_calibration": len(cal), **r})  # fmt: skip
    if not rows:
        raise SystemExit(f"No calibration-size scores under {args.scores}.")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(out / f"evt_training_{args.backbone}.csv", index=False)

    for alpha in args.alphas:
        d = df[(df.alpha == alpha) & (df.role == "dev")]
        print(f"\nalpha = {alpha:.0%}, development categories: mean false-reject rate "
              "(available cells), by calibration fraction")  # fmt: skip
        t = d.groupby(["rule", "calibration_fraction"]).agg(
            available=("available", "sum"), mean_fpr=("fpr", "mean")
        )
        print(t.unstack().round(3).to_string())
    # Prediction 1: EVT (tail 0.3) vs conformal on the same calibration images.
    d = df[(df.alpha == 0.10) & (df.role == "dev")]
    w = d.pivot_table(index=["category", "calibration_fraction", "seed"], columns="rule",
                      values="fpr")  # fmt: skip
    if {"conformal", "evt tail 0.3"} <= set(w.columns):
        both = w[["conformal", "evt tail 0.3"]].dropna()
        print(f"\nalpha = 10%: mean |EVT - conformal| false-reject rate = "
              f"{np.abs(both['evt tail 0.3'] - both['conformal']).mean():.3f} "
              f"over {len(both)} cells")  # fmt: skip
        print(both.groupby(level="category").mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
