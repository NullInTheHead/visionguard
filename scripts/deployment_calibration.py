"""RQ2, step 3: calibrate where you deploy (protocol section 11.2).

From the E1 per-image scores (the memory bank is not changed). For each category, R times:
half of the defect-free test images, drawn at random, are *deployment calibration*
images; the other half are evaluation goods; every defective test image counts for
recall. Four threshold rules are applied to the same draws:

    training max + margin     (1 + m) x max of the held-back training images (E1 rule)
    training conformal        split conformal on the held-back training images
    deployment max + margin   (1 + m) x max of the deployment calibration images
    deployment conformal      split conformal on the deployment calibration images

    training EVT / deployment EVT   generalized Pareto tail fit (protocol section 11.4)

Split conformal: threshold = the ceil((n + 1)(1 - alpha))-th smallest calibration score,
an image is rejected when its score is strictly above it, and it is valid only for
n >= 1/alpha - 1. Under exchangeability its expected false-reject rate is exactly
(n + 1 - k) / (n + 1) <= alpha, reported as ``expected_fpr_if_exchangeable``.

    python scripts/deployment_calibration.py
Writes results/thresholds/deployment_calibration_<backbone>.csv and .png.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from visionguard.config import load_config, load_split  # noqa: E402

RULES = ["training max + margin", "training conformal", "training EVT",
         "deployment max + margin", "deployment conformal", "deployment EVT"]  # fmt: skip
EVT_TAIL = 0.3  # protocol 11.4: fixed, not tuned
EVT_MIN_TAIL = 5


def conformal_rank(n: int, alpha: float) -> int | None:
    """1-based rank k of the split-conformal threshold, or None when n is too small."""
    # The small tolerance stops float rounding (e.g. 100 * 0.99 = 99.00000000000001) from
    # pushing k up by one.
    k = math.ceil((n + 1) * (1 - alpha) - 1e-9)
    return k if k <= n else None


def conformal_threshold(cal: np.ndarray, alpha: float) -> float | None:
    k = conformal_rank(len(cal), alpha)
    return None if k is None else float(np.sort(cal)[k - 1])


def expected_conformal_fpr(n: int, alpha: float) -> float | None:
    """Exact expected false-reject rate of split conformal under exchangeability."""
    k = conformal_rank(n, alpha)
    return None if k is None else (n + 1 - k) / (n + 1)


def single_draw_p_value(goods: np.ndarray, cal: np.ndarray, alpha: float) -> float | None:
    """Exact test of one fixed conformal threshold: if calibration and test goods were
    exchangeable, the number of goods above the k-th smallest of n calibration scores
    would be beta-binomial(m, n + 1 - k, k). Returns P(at least the observed count).

    A training calibration set is a single draw, so its realised false-reject rate can
    exceed alpha by bad luck alone; this separates luck from drift.
    """
    k = conformal_rank(len(cal), alpha)
    if k is None:
        return None
    threshold = np.sort(cal)[k - 1]
    observed = int((goods > threshold).sum())
    return float(stats.betabinom(len(goods), len(cal) + 1 - k, k).sf(observed - 1))


def evt_fit(cal: np.ndarray, tail: float = EVT_TAIL) -> tuple | None:
    """Peaks over threshold: GPD fitted to the top floor(tail * n) scores above u."""
    n = len(cal)
    n_tail = int(math.floor(tail * n + 1e-9))
    if n_tail < EVT_MIN_TAIL or n_tail >= n:
        return None
    x = np.sort(cal)
    u = x[n - n_tail - 1]
    exc = x[n - n_tail :] - u
    if not np.all(np.isfinite(exc)) or exc.max() <= 0:
        return None
    fit = gpd_mle(exc)
    if fit is None:
        return None
    c, scale = fit
    return float(u), n_tail / n, c, scale


def gpd_mle(x: np.ndarray) -> tuple[float, float] | None:
    """Maximum-likelihood GPD (location 0) by the one-dimensional profile likelihood.

    With theta = shape / scale, the shape that maximises the likelihood for a given theta is
    k(theta) = mean(log(1 + theta x)), and the profile log-likelihood is
    -n (log(k / theta) + k + 1). It is maximised over theta with the shape kept >= -1
    (below that the likelihood is unbounded). Same estimate as scipy.stats.genpareto.fit
    with floc=0, about 100x faster.
    """
    from scipy.optimize import minimize_scalar

    x = np.asarray(x, dtype=float)
    xmax, mean = x.max(), x.mean()
    if xmax <= 0:
        return None

    def negloglik(s: float) -> float:
        t = math.sinh(s)  # theta * xmax, in (-1, inf)
        if abs(t) < 1e-9:  # exponential limit
            return math.log(mean) + 1
        k = float(np.log1p(t * x / xmax).mean())
        if k < -1:
            return 1e6 * (1 + (-1 - k))
        return math.log(k / (t / xmax)) + k + 1

    lo, hi = math.asinh(-1 + 1e-9), math.asinh(1e4)
    res = minimize_scalar(negloglik, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-7})  # fmt: skip
    best = min((res.x, 0.0), key=negloglik)  # the exponential fit can win at the boundary
    t = math.sinh(best)
    if abs(t) < 1e-9:
        return 0.0, float(mean)
    c = float(np.log1p(t * x / xmax).mean())
    scale = c / (t / xmax)
    if not (np.isfinite(c) and np.isfinite(scale) and scale > 0):
        return None
    return c, float(scale)


def evt_threshold(fit: tuple | None, alpha: float) -> float | None:
    """The (1 - alpha) quantile of the fitted tail model; None if unavailable."""
    if fit is None:
        return None
    u, p_tail, c, scale = fit
    if alpha >= p_tail:
        return None
    t = u + float(stats.genpareto.ppf(1 - alpha / p_tail, c, loc=0, scale=scale))
    return t if np.isfinite(t) else None


def simulate(
    goods: np.ndarray,
    defects: np.ndarray,
    train_cal: np.ndarray,
    alphas: list[float],
    margin: float,
    repeats: int,
    seed: int,
) -> list[dict]:
    """One row per (rule, alpha): false-reject and recall statistics over ``repeats`` draws."""
    rng = np.random.default_rng(seed)
    n_dep = len(goods) // 2
    perms = np.stack([rng.permutation(len(goods)) for _ in range(repeats)])
    dep = np.sort(goods[perms[:, :n_dep]], axis=1)  # (R, n_dep), sorted per draw
    evals = goods[perms[:, n_dep:]]  # (R, n_eval)

    train_fit = evt_fit(train_cal)
    dep_fits = [evt_fit(d) for d in dep]
    rows = []
    for alpha in alphas:
        thresholds = {
            "training max + margin": np.full(repeats, (1 + margin) * train_cal.max()),
            "deployment max + margin": (1 + margin) * dep[:, -1],
        }
        n_cal = {"training max + margin": len(train_cal), "deployment max + margin": n_dep}
        t = conformal_threshold(train_cal, alpha)
        thresholds["training conformal"] = None if t is None else np.full(repeats, t)
        n_cal["training conformal"] = len(train_cal)
        k = conformal_rank(n_dep, alpha)
        thresholds["deployment conformal"] = None if k is None else dep[:, k - 1]
        n_cal["deployment conformal"] = n_dep
        t = evt_threshold(train_fit, alpha)
        thresholds["training EVT"] = None if t is None else np.full(repeats, t)
        n_cal["training EVT"] = len(train_cal)
        dep_t = np.array([np.nan if (v := evt_threshold(f, alpha)) is None else v
                          for f in dep_fits])  # fmt: skip
        # Available only if the fit works in every draw (else a biased subset is reported).
        thresholds["deployment EVT"] = None if np.isnan(dep_t).any() else dep_t
        n_cal["deployment EVT"] = n_dep

        for rule in RULES:
            thr = thresholds[rule]
            base = {"rule": rule, "alpha": alpha, "n_calibration": n_cal[rule],
                    "n_eval_goods": evals.shape[1], "n_defects": len(defects),
                    "repeats": repeats}  # fmt: skip
            expected = expected_conformal_fpr(n_cal[rule], alpha) if "conformal" in rule else None
            if thr is None:
                rows.append({**base, "available": False})
                continue
            fpr = (evals > thr[:, None]).mean(axis=1)
            recall = (defects[None, :] > thr[:, None]).mean(axis=1)
            se = fpr.std(ddof=1) / math.sqrt(repeats) if repeats > 1 else 0.0
            rows.append(
                {
                    **base,
                    "available": True,
                    "mean_fpr": float(fpr.mean()),
                    "mean_fpr_ci_low": float(fpr.mean() - 1.96 * se),
                    "mean_fpr_ci_high": float(fpr.mean() + 1.96 * se),
                    "expected_fpr_if_exchangeable": expected,
                    "share_draws_above_alpha": float((fpr > alpha).mean()),
                    "mean_recall": float(recall.mean()),
                    "exceeds_alpha": bool(fpr.mean() - 1.96 * se > alpha),
                    # Training rules are one fixed draw: their FPR on *all* goods, and the
                    # exact test against exchangeability (conformal only).
                    "fpr_all_goods": float((goods > thr[0]).mean())
                    if rule.startswith("training")
                    else None,
                    "p_exchangeable": single_draw_p_value(goods, train_cal, alpha)
                    if rule == "training conformal"
                    else None,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--backbone", default="wrn50")
    p.add_argument("--scores", default="results/e1/scores")
    p.add_argument("--out", default="results/thresholds")
    p.add_argument("--alphas", nargs="*", type=float, default=[0.10, 0.05, 0.01])
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    margin = float(cfg["calibration"]["margin"])
    split = load_split(cfg["data"]["split"])
    rows = []
    for category in list(split.dev) + list(split.held_out):
        path = Path(args.scores) / f"{args.backbone}_{category}.csv"
        if not path.is_file():
            continue
        s = pd.read_csv(path)
        test = s[s.split == "test"]
        goods = test[test.label == 0].score.to_numpy()
        defects = test[test.label == 1].score.to_numpy()
        train_cal = s[s.split == "calibration"].score.to_numpy()
        if len(goods) < 4 or len(train_cal) == 0:
            continue
        role = split.role(category)
        for r in simulate(goods, defects, train_cal, args.alphas, margin, args.repeats,
                          args.seed):  # fmt: skip
            rows.append({"backbone": args.backbone, "category": category, "role": role,
                         "margin": margin, **r})  # fmt: skip
    if not rows:
        raise SystemExit(f"No E1 scores found under {args.scores}.")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(out / f"deployment_calibration_{args.backbone}.csv", index=False)
    _figure(df, out / f"deployment_calibration_{args.backbone}.png", args.backbone,
            max(args.alphas))  # fmt: skip
    _print_summary(df, args.alphas)
    return 0


def _print_summary(df: pd.DataFrame, alphas: list[float]) -> None:
    for alpha in alphas:
        d = df[(df.alpha == alpha) & df.available]
        print(f"\nalpha = {alpha:.0%}: mean false-reject rate (recall) per category")
        table = d.pivot(index="category", columns="rule", values="mean_fpr")
        rec = d.pivot(index="category", columns="rule", values="mean_recall")
        cols = [r for r in RULES if r in table.columns]
        both = table[cols].map(lambda v: f"{v:.3f}")
        for c in cols:
            both[c] = both[c] + rec[c].map(lambda v: f" ({v:.2f})")
        print(both.to_string())
        for role in dict.fromkeys(d.role):
            r = d[d.role == role]
            parts = []
            for rule in cols:
                x = r[r.rule == rule]
                parts.append(f"{rule}: {int(x.exceeds_alpha.sum())}/{len(x)} exceed alpha")
            print(f"  {role}: " + "; ".join(parts))
            tc = r[r.rule == "training conformal"]
            drift = tc[tc.p_exchangeable < 0.05].category.tolist()
            print(f"    training conformal beyond chance (exact test, p < 0.05): "
                  f"{drift or 'none'}")  # fmt: skip


def _figure(df: pd.DataFrame, path: Path, backbone: str, alpha: float) -> None:
    """Mean false-reject rate per category and rule at one alpha, with the alpha line."""
    import matplotlib.pyplot as plt

    d = df[(df.alpha == alpha) & df.available]
    cats = list(dict.fromkeys(df.category))
    y = {c: i for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(cats) + 1.4))
    styles = {"training max + margin": ("C0", "o", False), "training conformal": ("C0", "s", True),
              "training EVT": ("C0", "^", True), "deployment max + margin": ("C1", "o", False),
              "deployment conformal": ("C1", "s", True),
              "deployment EVT": ("C1", "^", True)}  # fmt: skip
    for n, rule in enumerate(RULES):
        r = d[d.rule == rule]
        color, marker, filled = styles[rule]
        ys = [y[c] + (n - 2.5) * 0.13 for c in r.category]
        ax.errorbar(r.mean_fpr, ys, xerr=[r.mean_fpr - r.mean_fpr_ci_low,
                    r.mean_fpr_ci_high - r.mean_fpr], fmt=marker, color=color, ms=5,
                    mfc=color if filled else "white", lw=0.8, label=rule)  # fmt: skip
    ax.axvline(alpha, color="0.3", ls="--", lw=1)
    ax.text(alpha, -0.9, f" target {alpha:.0%}", fontsize=8, color="0.3")
    ax.set_yticks(range(len(cats)), [c + ("" if c in set(df[df.role == "dev"].category)
                                          else " *") for c in cats], fontsize=8)  # fmt: skip
    ax.invert_yaxis()
    ax.set_xlabel("mean false-reject rate on defect-free test images (1,000 draws)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Calibrate where you deploy ({backbone}, alpha = {alpha:.0%}); "
                 "* = held-out (post-hoc) or clean test", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
