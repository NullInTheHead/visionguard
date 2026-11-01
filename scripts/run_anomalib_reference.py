"""Reference numbers from anomalib's PatchCore, to check our implementation (Phase 2).

Run it in a separate environment, since anomalib pins its own torch/lightning versions:

    python -m venv .venv-anomalib && source .venv-anomalib/bin/activate
    pip install "anomalib>=2.0,<3" pandas
    python scripts/run_anomalib_reference.py --categories bottle capsule

Settings match ours: WideResNet-50 layers 2+3, 10% coreset, 9-neighbour reweighting,
resize 256 -> centre crop 224, every training image in the memory bank. Compare with

    python scripts/run_e1.py --categories bottle capsule --calibration-fraction 0

Our implementation should land within about 1-2 AUROC points (roadmap Phase 2).
Use development categories only: held-out categories are never looked at before the
settings are frozen. Capsule is a useful second check because of its tiny defects.
Writes results/anomalib_reference.csv.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="data/mvtec_anomaly_detection")
    p.add_argument("--categories", nargs="+", default=["bottle", "capsule"])
    p.add_argument("--out", default="results/anomalib_reference.csv")
    p.add_argument("--accelerator", default="auto", help="auto | gpu | cpu | mps")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # tests
    args = p.parse_args(argv)

    import anomalib
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.metrics import AUPRO, AUROC, Evaluator
    from anomalib.models import Patchcore

    rows = []
    for category in args.categories:
        datamodule = MVTecAD(
            root=args.root,
            category=category,
            train_batch_size=16,
            eval_batch_size=16,
            num_workers=args.num_workers,
        )
        evaluator = Evaluator(
            test_metrics=[
                AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
                AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
                AUPRO(
                    fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False, fpr_limit=0.3
                ),
            ]
        )
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            pre_trained=not args.no_pretrained,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            pre_processor=Patchcore.configure_pre_processor(
                image_size=(256, 256), center_crop_size=(224, 224)
            ),
            evaluator=evaluator,
            visualizer=False,
        )
        engine = Engine(
            default_root_dir="results/anomalib_runs",
            logger=False,
            accelerator=args.accelerator,
            devices=1,
        )
        engine.fit(model=model, datamodule=datamodule)
        metrics = engine.test(model=model, datamodule=datamodule, verbose=False)[0]
        row = {
            "category": category,
            "image_auroc": round(float(metrics["image_AUROC"]), 5),
            "pixel_auroc": round(float(metrics["pixel_AUROC"]), 5),
            "au_pro": round(float(metrics["pixel_AUPRO"]), 5),
            "anomalib": anomalib.__version__,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        rows.append(row)
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out.is_file():
        old = pd.read_csv(out)
        new = pd.concat([old[~old.category.isin(new.category)], new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
