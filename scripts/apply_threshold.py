"""Apply the frozen deployable threshold to any categories (protocol section 6).

Threshold per category = (1 + m) x max calibration score, with m read from
configs/base.yaml (calibration.margin), never re-tuned. On the held-out categories this
is the independent test of the threshold rule; on development categories it reproduces
the in-sample tuning diagnostic.

    python scripts/apply_threshold.py                          # WRN-50, held-out categories
    python scripts/apply_threshold.py --backbone dinov2 --categories dev

Writes results/e1/threshold_<backbone>_<categories>.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from visionguard.config import load_config, load_split, resolve_categories
from visionguard.metrics import operating_point_at


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--backbone", default="wrn50")
    p.add_argument("--categories", nargs="*", default=["held_out"])
    p.add_argument("--scores", default="results/e1/scores")
    p.add_argument("--out", default="results/e1")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    m = float(cfg["calibration"]["margin"])
    split = load_split(cfg["data"]["split"])
    categories = resolve_categories(args.categories, split)

    rows = []
    for category in categories:
        df = pd.read_csv(Path(args.scores) / f"{args.backbone}_{category}.csv")
        test, cal = df[df.split == "test"], df[df.split == "calibration"]
        if cal.empty:
            raise ValueError(f"No calibration scores for {category}.")
        cal_max = float(cal.score.max())
        op = operating_point_at(test.label, test.score, (1 + m) * cal_max)
        rows.append(
            {
                "backbone": args.backbone,
                "category": category,
                "role": "held_out" if category in split.held_out else "dev",
                "margin": m,
                "calibration_max": round(cal_max, 5),
                "threshold": round(op.threshold, 5),
                "recall": round(op.recall, 4),
                "fpr": round(op.fpr, 4),
                "rejects_per_1000": round(op.false_rejects_per_1000, 1),
            }
        )
    table = pd.DataFrame(rows)
    tag = args.categories[0] if len(args.categories) == 1 else "custom"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"threshold_{args.backbone}_{tag}.csv", index=False)

    print(f"Frozen margin m = {m:.2f}\n")
    print(table[["category", "recall", "fpr", "rejects_per_1000"]].to_string(index=False))
    print(
        f"\nMean recall {table.recall.mean():.3f}, mean false rejects "
        f"{table.rejects_per_1000.mean():.1f} per 1,000 good parts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
