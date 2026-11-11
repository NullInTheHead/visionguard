"""E2: anomaly detection on a product the detector has never seen (protocol section 5).

For each target category (the five held-out categories by default):
  E2a  k = 0: a shared memory bank made of the development categories' coresets, with
       nothing from the target. Scored on the target's test set.
  E2b  k = 1, 2, 4, 8: a bank of every patch of k reference images of the target, drawn
       with 5 seeds (nested draws). Scored on the same test set.
The target's test set is embedded once and scored against all 21 banks.

    python scripts/run_e2.py                       # WRN-50, held-out targets
    python scripts/run_e2.py --backbone dinov2     # Phase 3, other backbones
    python scripts/run_e2.py --skip-zero-shot      # E2b only
    python scripts/run_e2.py --rotations rot90     # E2c: post-hoc rotation diagnostic
    python scripts/run_e2.py --rotations rot45     #      (protocol section 16.1)

Outputs (under --out, default results/e2):
    metrics.csv                         one row per (backbone, category, experiment, k, seed)
    scores/<backbone>_<category>.csv    per-image scores for every bank
    recovery_<backbone>_image_auroc.png recovery curves, k = 0 to 8, with E1 (k = all)
    recovery_<backbone>_au_pro.png
    rotation_<backbone>_image_auroc.png  E2b vs E2c (written once an E2c run exists)
    E2c scores go to scores/<backbone>_<category>_<rotations>.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import load_config, load_split, resolve_categories  # noqa: E402
from visionguard.data import category_type  # noqa: E402
from visionguard.experiment import (  # noqa: E402
    efficiency_columns,
    evaluate,
    recovery_figure,
    reset_device_peak,
    run_metadata,
    scores_frame,
    upsert_rows,
)


def parse_args(argv):
    p = argparse.ArgumentParser(description="E2: zero-shot and few-shot on new products.")
    p.add_argument("--config", default="configs/e2.yaml")
    p.add_argument("--backbone")
    p.add_argument("--categories", nargs="*", help="targets: names, or dev / held_out / all")
    p.add_argument("--root")
    p.add_argument("--out")
    p.add_argument("--device")
    p.add_argument("--ks", nargs="*", type=int)
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--skip-zero-shot", action="store_true")
    p.add_argument(
        "--rotations",
        choices=["none", "rot90", "rot45"],
        default="none",
        help="E2c post-hoc diagnostic: add rotated copies of the references to the bank",
    )
    p.add_argument("--no-cache", action="store_true", help="rebuild coresets, do not save")
    p.add_argument("--bank-categories", nargs="*", help=argparse.SUPPRESS)  # tests
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # tests
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    pc = cfg["patchcore"]
    backbone = args.backbone or cfg["backbone"]
    root = args.root or cfg["data"]["root"]
    out = Path(args.out or cfg["output"])
    cal_fraction = cfg["calibration"]["fraction"]
    cal_seed = cfg["calibration"]["seed"]
    ks = args.ks or cfg["few_shot"]["ks"]
    seeds = args.seeds if args.seeds is not None else cfg["few_shot"]["seeds"]
    cache_dir = None if args.no_cache else cfg.get("cache_dir", "cache/banks")

    split = load_split(cfg["data"]["split"])
    targets = resolve_categories(args.categories or cfg["categories"], split)
    bank_sources = args.bank_categories or list(split.dev)
    if not set(bank_sources) <= set(split.dev):
        print("The zero-shot bank may only use development categories.", file=sys.stderr)
        return 2
    if len(seeds) < 5:
        print(f"Note: {len(seeds)} seeds; the protocol reports 5 (section 8, rule 4).")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    from visionguard.banks import (
        ROTATIONS,
        embed_references,
        embed_references_rotated,
        few_shot_draws,
        reference_pool,
        shared_bank,
    )
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
    rotated = args.rotations != "none"
    few_label = f"e2c-{args.rotations}" if rotated else "e2b"
    if rotated:
        print("E2c is a post-hoc diagnostic (protocol section 16.1); zero-shot is skipped.")
    print(f"E2 | backbone={backbone} device={meta['device']} targets={targets} ks={ks} "
          f"seeds={seeds}")  # fmt: skip

    for category in targets:
        torch.manual_seed(0)
        reference, _, test = reference_pool(root, category, cal_fraction, cal_seed)
        test_set = model.embed_for_scoring(test)
        base = {
            "backbone": backbone,
            "category": category,
            "type": category_type(category),
            "role": "held_out" if category in split.held_out else "dev",
            "n_test": len(test),
        }
        rows, frames = [], []

        def record(experiment, k, seed, extra, pred, base=base, rows=rows, frames=frames):
            metrics = evaluate(pred.labels, pred.scores, pred.scores_max, pred.masks, pred.maps)
            row = {
                **base,
                "experiment": experiment,
                "k": k,
                "seed": seed,
                "bank_patches": int(model.memory_bank.shape[0]),
                "bank_mb": round(model.memory_bank_mb, 2),
                **extra,
                **{m: round(v, 5) for m, v in metrics.items()},
                **efficiency_columns(model, pred),
                **meta,
                "config": args.config,
            }
            rows.append(row)
            frame = scores_frame(pred, "test")
            frame.insert(0, "seed", seed)
            frame.insert(0, "k", k)
            frame.insert(0, "experiment", experiment)
            frames.append(frame)
            return metrics

        # E2a: zero-shot, bank from other products only.
        if not (args.skip_zero_shot or rotated):
            reset_device_peak(device)
            bank, sources = shared_bank(model, backbone, root, bank_sources, category,
                                        pc["coreset_ratio"], cal_fraction, cal_seed,
                                        cache_dir)  # fmt: skip
            model.set_memory_bank(bank)
            model.fit_timing = {}
            m = record("e2a", 0, -1, {"bank_categories": ";".join(sources)},
                       model.score_embedded(test_set))  # fmt: skip
            print(f"  {category:<11} k=0      image AUROC {m['image_auroc']:.4f}  "
                  f"AU-PRO {m['au_pro']:.4f}  (bank: {len(sources)} other products)")  # fmt: skip

        # E2b: few-shot, k images of the target, full patch bank.
        draws = few_shot_draws(reference, ks, seeds)
        used = [s for d in draws.values() for s in d]
        if rotated:
            ref_emb = embed_references_rotated(model, used, ROTATIONS[args.rotations])
        else:
            ref_emb = embed_references(model, used)
        for k in ks:
            aurocs = []
            for seed in seeds:
                reset_device_peak(device)
                images = draws[(k, seed)]
                model.set_memory_bank(torch.cat([ref_emb[s.id] for s in images]))
                model.fit_timing = {}
                extra = {"reference_ids": ";".join(s.id for s in images),
                         "rotations": args.rotations}  # fmt: skip
                m = record(few_label, k, seed, extra, model.score_embedded(test_set))
                aurocs.append(m["image_auroc"])
            sd = np.std(aurocs, ddof=1) if len(aurocs) > 1 else 0.0
            print(f"  {category:<11} k={k:<6} image AUROC {np.mean(aurocs):.4f} ± {sd:.4f}")

        upsert_rows(out / "metrics.csv", rows,
                    key=["backbone", "category", "experiment", "k", "seed"])  # fmt: skip
        (out / "scores").mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.rotations}" if rotated else ""
        pd.concat(frames).to_csv(out / "scores" / f"{backbone}_{category}{suffix}.csv",
                                 index=False)  # fmt: skip
        del test_set, ref_emb

    results = pd.read_csv(out / "metrics.csv")
    mine = results[(results.backbone == backbone) & results.category.isin(targets)]
    e1_path = Path(cfg["data"].get("e1_metrics", "results/e1/metrics.csv"))
    e1 = pd.read_csv(e1_path) if e1_path.is_file() else None
    if e1 is not None:
        e1 = e1[e1.backbone == backbone]
    base_runs = mine[mine.experiment.isin(["e2a", "e2b"])]
    if (base_runs.experiment == "e2b").any():
        for metric in ["image_auroc", "au_pro"]:
            recovery_figure(base_runs, e1, out / f"recovery_{backbone}_{metric}.png",
                            metric=metric, title=f"E2 recovery curves | {backbone}")  # fmt: skip
    series = [(e, lab) for e, lab in [("e2b", "no rotation"), ("e2c-rot90", "rot90"),
                                      ("e2c-rot45", "rot45")]
              if (mine.experiment == e).any()]  # fmt: skip
    if len(series) > 1:
        recovery_figure(mine, e1, out / f"rotation_{backbone}_image_auroc.png",
                        series=series, include_zero_shot=False,
                        title=f"E2c rotation diagnostic (post-hoc) | {backbone}")  # fmt: skip

    this_run = mine[mine.experiment.isin(["e2a", few_label] if not rotated else [few_label])]
    summary = (
        this_run.assign(
            step=np.where(this_run.experiment == "e2a", "k=0", "k=" + this_run.k.astype(str))
        )
        .groupby(["category", "step"])["image_auroc"]
        .mean()
        .unstack()
    )
    order = [c for c in ["k=0"] + [f"k={k}" for k in ks] if c in summary.columns]
    print(f"\nMean image AUROC by category and k ({few_label}):\n"
          + summary[order].round(4).to_string())  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
