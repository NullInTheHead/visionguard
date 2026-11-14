"""ViT layer sweep on the development categories (protocol section 12.1).

For each ViT backbone and each candidate pair of blocks, runs E1 on the 10 development
categories (no heatmaps) and applies the selection rule registered in the protocol:
highest mean of (image AUROC + AU-PRO) / 2; a pair within 0.002 of the best that uses
shallower blocks wins (it is cheaper).

    python scripts/vit_layer_sweep.py                        # 3 backbones x 3 pairs, ~4 h on a Mac
    python scripts/vit_layer_sweep.py --backbones dinov2     # one backbone

Finished (backbone, pair) runs are skipped, so an interrupted sweep resumes where it
stopped. Writes results/vit_sweep/metrics.csv, scores/ and selection.csv.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from visionguard.config import load_config, load_split

PAIRS = ["2,5", "3,7", "5,9"]  # 0-based blocks of 12; 3,7 is the protocol's starting point
TIE = 0.002


def tag(backbone: str, pair: str) -> str:
    a, b = pair.split(",")
    return f"{backbone}_L{a}-{b}"


def select_layers(df: pd.DataFrame, backbones: list[str], pairs: list[str]) -> pd.DataFrame:
    """One row per backbone and pair, with the selection rule applied."""
    rows = []
    for backbone in backbones:
        for pair in pairs:
            d = df[df.backbone == tag(backbone, pair)]
            if d.empty:
                continue
            rows.append({
                "backbone": backbone, "layers": pair, "n_categories": len(d),
                "image_auroc": d.image_auroc.mean(), "au_pro": d.au_pro.mean(),
                "pixel_auroc": d.pixel_auroc.mean(),
                "criterion": ((d.image_auroc + d.au_pro) / 2).mean(),
                "fit_seconds": d.fit_seconds.mean(),
                "ms_per_image": 1000 * d.seconds_per_image.mean(),
            })  # fmt: skip
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["selected"] = False
    for _, g in out.groupby("backbone"):
        best = g.criterion.max()
        near = g[g.criterion >= best - TIE]
        shallowest = min(near.index, key=lambda i: int(near.loc[i, "layers"].split(",")[1]))
        out.loc[shallowest, "selected"] = True
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/e1.yaml")
    p.add_argument("--backbones", nargs="*", default=["vit_sup", "dinov2", "clip"])
    p.add_argument("--pairs", nargs="*", default=PAIRS)
    p.add_argument("--out", default="results/vit_sweep")
    p.add_argument("--device")
    p.add_argument("--root")
    args = p.parse_args(argv)

    split = load_split(load_config(args.config)["data"]["split"])
    dev = set(split.dev)
    out = Path(args.out)
    spec = importlib.util.spec_from_file_location("run_e1", Path(__file__).with_name("run_e1.py"))
    run_e1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_e1)

    for backbone in args.backbones:
        for pair in args.pairs:
            metrics = out / "metrics.csv"
            if metrics.is_file():
                done = pd.read_csv(metrics)
                if dev <= set(done[done.backbone == tag(backbone, pair)].category):
                    print(f"{tag(backbone, pair)}: done, skipped")
                    continue
            extra = (["--device", args.device] if args.device else []) + (
                ["--root", args.root] if args.root else [])  # fmt: skip
            rc = run_e1.main(["--config", args.config, "--backbone", backbone, "--layers", pair,
                              "--categories", "dev", "--out", str(out), "--no-heatmaps",
                              *extra])  # fmt: skip
            if rc != 0:
                return rc

    table = select_layers(pd.read_csv(out / "metrics.csv"), args.backbones, args.pairs)
    table.round(4).to_csv(out / "selection.csv", index=False)
    print("\nViT layer sweep, development categories (criterion = mean of image AUROC and "
          "AU-PRO):\n" + table.round(4).to_string(index=False))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
