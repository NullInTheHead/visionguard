"""RQ2, step 4: does a larger training calibration set help? (protocol section 11.3)

For each category, calibration fraction (10%, 25%, 50% of the training images) and
calibration seed (0, 1, 2): hold back that share of the training images, build the E1
memory bank (10% coreset) from the rest, and score the held-back images and the test set.
Fraction 10% with seed 0 is the E1 split, so it reproduces E1.

Reported per row: calibration and bank sizes, image AUROC, pixel AUROC, AU-PRO; for
training split conformal at each alpha (where n >= 1/alpha - 1) the false-reject rate on
all defect-free test images, recall and the exact beta-binomial p-value; the same two
rates for max + margin (m from the config).

    python scripts/calibration_size.py                        # 10 development categories
    python scripts/calibration_size.py --categories leather wood
    python scripts/calibration_size.py --categories held_out --allow-held-out   # post-hoc

Writes results/thresholds/calibration_size_<backbone>.csv (one row per category,
fraction and seed; reruns replace their rows), per-image scores under
results/thresholds/calibration_size_scores/, and calibration_size_<backbone>.png.
Every training image is embedded once per category; each (fraction, seed) only rebuilds
the coreset, so the run takes about an hour on an Apple M-series GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import load_config, load_split, resolve_categories  # noqa: E402
from visionguard.data import by_split, calibration_split, index_category  # noqa: E402
from visionguard.experiment import evaluate, run_metadata, scores_frame, upsert_rows  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "deployment_calibration", Path(__file__).with_name("deployment_calibration.py")
)
_dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dc)
conformal_threshold = _dc.conformal_threshold
single_draw_p_value = _dc.single_draw_p_value


def threshold_columns(cal: np.ndarray, goods: np.ndarray, defects: np.ndarray,
                      alphas: list[float], margin: float) -> dict:  # fmt: skip
    """False-reject rate, recall and (conformal) exact p-value per rule."""
    row = {}
    t = (1 + margin) * cal.max()
    row["margin_fpr"] = float((goods > t).mean())
    row["margin_recall"] = float((defects > t).mean())
    for alpha in alphas:
        tag = f"conformal{round(alpha * 100)}"
        t = conformal_threshold(cal, alpha)
        row[f"{tag}_available"] = t is not None
        row[f"{tag}_fpr"] = None if t is None else float((goods > t).mean())
        row[f"{tag}_recall"] = None if t is None else float((defects > t).mean())
        row[f"{tag}_p_exchangeable"] = single_draw_p_value(goods, cal, alpha)
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/e1.yaml")
    p.add_argument("--backbone")
    p.add_argument("--categories", nargs="*", default=["dev"])
    p.add_argument("--fractions", nargs="*", type=float, default=[0.10, 0.25, 0.50])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--alphas", nargs="*", type=float, default=[0.10, 0.05, 0.01])
    p.add_argument("--allow-held-out", action="store_true",
                   help="also run held-out categories (post-hoc, protocol 11.3)")  # fmt: skip
    p.add_argument("--root")
    p.add_argument("--out", default="results/thresholds")
    p.add_argument("--device")
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # tests
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    pc = cfg["patchcore"]
    backbone = args.backbone or cfg["backbone"]
    root = args.root or cfg["data"]["root"]
    margin = float(cfg["calibration"]["margin"])
    split = load_split(cfg["data"]["split"])
    categories = resolve_categories(args.categories, split)
    held = [c for c in categories if c in split.held_out]
    if held and not args.allow_held_out:
        print(f"Refusing held-out categories {held} without --allow-held-out (post-hoc only).",
              file=sys.stderr)  # fmt: skip
        return 2

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

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
    (out / "calibration_size_scores").mkdir(parents=True, exist_ok=True)
    print(f"Calibration size | backbone={backbone} device={meta['device']} "
          f"fractions={args.fractions} seeds={args.seeds}")  # fmt: skip

    for category in categories:
        torch.manual_seed(0)
        samples = index_category(root, category)
        train = sorted(by_split(samples, "train"))
        test = by_split(samples, "test")
        train_set = model.embed_for_scoring(train)  # every training image, embedded once
        test_set = model.embed_for_scoring(test)
        position = {sid: i for i, sid in enumerate(train_set.ids)}
        rows, frames = [], []
        for fraction in args.fractions:
            for seed in args.seeds:
                try:
                    ref, cal = calibration_split(train, fraction, seed)
                except ValueError as err:
                    print(f"  {category:<11} fraction {fraction:.0%}: {err}")
                    continue
                ref_idx = [position[s.id] for s in ref]
                patches = train_set.embeddings[ref_idx].flatten(0, 1)
                model.set_memory_bank(model.subsample(patches, pc["coreset_ratio"]))
                cal_pred = model.score_embedded(train_set.subset([position[s.id] for s in cal]))
                pred = model.score_embedded(test_set)
                m = evaluate(pred.labels, pred.scores, pred.scores_max, pred.masks, pred.maps)
                goods = pred.scores[pred.labels == 0]
                defects = pred.scores[pred.labels == 1]
                row = {
                    "backbone": backbone,
                    "category": category,
                    "role": split.role(category),
                    "calibration_fraction": fraction,
                    "seed": seed,
                    "n_reference": len(ref),
                    "n_calibration": len(cal),
                    "n_test_good": len(goods),
                    "n_defects": len(defects),
                    "bank_patches": int(model.memory_bank.shape[0]),
                    "image_auroc": round(m["image_auroc"], 5),
                    "pixel_auroc": round(m["pixel_auroc"], 5),
                    "au_pro": round(m["au_pro"], 5),
                    "margin": margin,
                    **{k: (round(v, 5) if isinstance(v, float) else v)
                       for k, v in threshold_columns(cal_pred.scores, goods, defects,
                                                     args.alphas, margin).items()},
                    **meta,
                }  # fmt: skip
                rows.append(row)
                for name, pr in (("calibration", cal_pred), ("test", pred)):
                    f = scores_frame(pr, name)
                    f.insert(0, "seed", seed)
                    f.insert(0, "calibration_fraction", fraction)
                    frames.append(f)
                c10 = row.get("conformal10_fpr")
                print(f"  {category:<11} fraction {fraction:.0%} seed {seed}: n_cal "
                      f"{len(cal):>3}  AUROC {m['image_auroc']:.4f}  conformal 10% FPR "
                      + ("n/a" if c10 is None else f"{c10:.3f} (p {row['conformal10_p_exchangeable']:.3f})"))  # fmt: skip  # noqa: E501
        upsert_rows(out / f"calibration_size_{backbone}.csv", rows,
                    key=["backbone", "category", "calibration_fraction", "seed"])  # fmt: skip
        if frames:
            pd.concat(frames).to_csv(
                out / "calibration_size_scores" / f"{backbone}_{category}.csv", index=False
            )
        del train_set, test_set

    df = pd.read_csv(out / f"calibration_size_{backbone}.csv")
    df = df[df.backbone == backbone]
    _figure(df, out / f"calibration_size_{backbone}.png", backbone)
    _print_summary(df, args.alphas)
    return 0


def _print_summary(df: pd.DataFrame, alphas: list[float]) -> None:
    for role in dict.fromkeys(df.role):
        d = df[df.role == role]
        if d.empty:
            continue
        print(f"\n{role}: means over categories and seeds, by calibration fraction")
        g = d.groupby("calibration_fraction")
        table = pd.DataFrame({
            "n_cal (median)": g.n_calibration.median(),
            "image_auroc": g.image_auroc.mean().round(4),
            "au_pro": g.au_pro.mean().round(4),
            "margin_fpr": g.margin_fpr.mean().round(3),
            "margin_recall": g.margin_recall.mean().round(3),
        })  # fmt: skip
        for alpha in alphas:
            tag = f"conformal{round(alpha * 100)}"
            if f"{tag}_fpr" not in d:
                continue
            table[f"{tag}: available"] = g[f"{tag}_available"].sum().astype(int)
            table[f"{tag}: mean FPR"] = g[f"{tag}_fpr"].mean().round(3)
            table[f"{tag}: p<0.05"] = g[f"{tag}_p_exchangeable"].apply(
                lambda s: int((s < 0.05).sum())
            )
        print(table.T.to_string())
        if "conformal10_p_exchangeable" in d:
            low = d[d.conformal10_p_exchangeable < 0.05]
            sig = low.groupby(["category", "calibration_fraction"]).size().unstack(fill_value=0)
            if not sig.empty:
                print("\nseeds beyond chance at alpha = 10% (category x fraction):")
                print(sig.to_string())


def _figure(df: pd.DataFrame, path: Path, backbone: str) -> None:
    """Image AUROC and training-conformal false-reject rate (alpha = 10%) vs fraction."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for n, (cat, d) in enumerate(df.groupby("category", sort=False)):
        g = d.groupby("calibration_fraction")
        style = {"color": f"C{n % 10}", "marker": "o", "ms": 4, "lw": 1,
                 "ls": "-" if d.role.iloc[0] == "dev" else "--"}  # fmt: skip
        axes[0].plot(g.image_auroc.mean().index, g.image_auroc.mean(), label=cat, **style)
        if "conformal10_fpr" in d and d.conformal10_fpr.notna().any():
            f = g.conformal10_fpr.mean()
            axes[1].errorbar(f.index, f, yerr=g.conformal10_fpr.std().fillna(0), capsize=2,
                             **style)  # fmt: skip
    axes[1].axhline(0.10, color="0.3", ls=":", lw=1)
    axes[0].set_ylabel("image AUROC (mean of seeds)", fontsize=8)
    axes[1].set_ylabel("false-reject rate, training conformal at 10%", fontsize=8)
    for ax in axes:
        ax.set_xlabel("share of training images held back for calibration", fontsize=8)
        ax.set_xticks(sorted(df.calibration_fraction.unique()))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Calibration size ({backbone}); dashed = held-out, post-hoc; bars = seed "
                 "spread", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
