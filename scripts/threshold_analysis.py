"""RQ2, step 1: are calibration images exchangeable with defect-free test images?

Every calibration rule (max + margin, split conformal, a fitted tail) assumes that the
held-back defect-free *training* images and the defect-free *test* images come from the
same score distribution. This checks it per category, from the E1 per-image scores:

* exceedance: share of defect-free test images scoring above the calibration maximum,
  against 1/(n + 1), its expected value under exchangeability with n calibration images;
* Kolmogorov-Smirnov and one-sided Mann-Whitney tests (do test goods score higher?);
* median shift of test goods over calibration images, in percent.

Development categories are the evidence; held-out categories are listed separately and
labelled post-hoc (protocol section 11).

    python scripts/threshold_analysis.py
Writes results/thresholds/exchangeability_<backbone>.csv and exchangeability_<backbone>.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from visionguard.config import load_split  # noqa: E402


def exchangeability_row(cal: np.ndarray, good: np.ndarray) -> dict:
    n = len(cal)
    return {
        "n_calibration": n,
        "n_test_good": len(good),
        "exceed_observed": float((good > cal.max()).mean()),
        "exceed_expected": 1.0 / (n + 1),
        "calibration_median": float(np.median(cal)),
        "test_good_median": float(np.median(good)),
        "median_shift_pct": float(100 * (np.median(good) / np.median(cal) - 1)),
        "ks_p": float(stats.ks_2samp(cal, good).pvalue),
        "mw_p_test_higher": float(stats.mannwhitneyu(good, cal, alternative="greater").pvalue),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backbone", default="wrn50")
    p.add_argument("--scores", default="results/e1/scores")
    p.add_argument("--split", default="configs/split.yaml")
    p.add_argument("--out", default="results/thresholds")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level for flags")
    args = p.parse_args(argv)

    split = load_split(args.split)
    rows, dists = [], {}
    for category in list(split.dev) + list(split.held_out):
        path = Path(args.scores) / f"{args.backbone}_{category}.csv"
        if not path.is_file():
            continue
        s = pd.read_csv(path)
        cal = s[s.split == "calibration"].score.to_numpy()
        good = s[(s.split == "test") & (s.label == 0)].score.to_numpy()
        if len(cal) == 0 or len(good) == 0:
            continue
        role = "dev" if category in split.dev else "held_out (post-hoc)"
        rows.append({"backbone": args.backbone, "category": category, "role": role,
                     **exchangeability_row(cal, good)})  # fmt: skip
        dists[category] = (cal, good, role)
    if not rows:
        raise SystemExit(f"No E1 scores found under {args.scores}.")
    df = pd.DataFrame(rows)
    df["shift_significant"] = df.mw_p_test_higher < args.alpha

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / f"exchangeability_{args.backbone}.csv", index=False)
    _figure(dists, out / f"exchangeability_{args.backbone}.png", args.backbone)

    cols = ["category", "role", "n_calibration", "exceed_observed", "exceed_expected",
            "median_shift_pct", "mw_p_test_higher"]  # fmt: skip
    print(df[cols].round(3).to_string(index=False))
    dev = df[df.role == "dev"]
    pooled_obs = (dev.exceed_observed * dev.n_test_good).sum() / dev.n_test_good.sum()
    pooled_exp = (dev.exceed_expected * dev.n_test_good).sum() / dev.n_test_good.sum()
    print(
        f"\nDevelopment categories: {pooled_obs:.1%} of defect-free test images score above "
        f"the calibration maximum (expected {pooled_exp:.1%} if exchangeable); "
        f"test goods score significantly higher in {int(dev.shift_significant.sum())} of "
        f"{len(dev)} (one-sided Mann-Whitney, p < {args.alpha})."
    )
    return 0


def _figure(dists: dict, path: Path, backbone: str) -> None:
    """Empirical CDFs of calibration vs defect-free test scores, one panel per category."""
    import matplotlib.pyplot as plt

    cats = list(dists)
    ncols = 5
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, cat in zip(axes.ravel(), cats, strict=False):
        cal, good, role = dists[cat]
        ax.set_visible(True)
        for values, name, color in ((cal, "calibration (train goods)", "C0"),
                                    (good, "test goods", "C1")):  # fmt: skip
            x = np.sort(values)
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, lw=1.3,
                    label=name)  # fmt: skip
        ax.axvline(cal.max(), color="C0", ls=":", lw=1)
        ax.set_title(f"{cat}" + ("" if role == "dev" else " (held-out)"), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Calibration vs defect-free test scores (E1, {backbone}); dotted = "
                 "calibration maximum", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
