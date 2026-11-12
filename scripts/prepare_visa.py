"""Put the raw VisA dataset into the MVTec folder layout, with the official 1cls split.

VisA (Zou et al., ECCV 2022; CC BY 4.0) is the untouched clean test set (protocol
decision D5). The split is the authors' `split_csv/1cls.csv`, copied verbatim to
configs/visa_split_1cls.csv: per category, train = normal images only; test = normal and
anomalous images. Output per category:

    <out>/<category>/train/good/*.JPG
    <out>/<category>/test/good/*.JPG
    <out>/<category>/test/bad/*.JPG
    <out>/<category>/ground_truth/bad/<stem>_mask.png    binarised to 0 / 255

Images are hard-linked when possible (no extra disk space), else copied; nothing is
resized. Masks are binarised as in the authors' utils/prepare_data.py.

    bash scripts/setup_visa.sh            # download (about 1.8 GB) and run this
    python scripts/prepare_visa.py --raw data/VisA --out data/visa
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def prepare(raw: Path, out: Path, split_file: Path) -> Counter:
    counts: Counter = Counter()
    with open(split_file, newline="") as f:
        for row in csv.DictReader(f):
            category, split, label = row["object"], row["split"], row["label"]
            folder = "good" if label == "normal" else "bad"
            if split == "train" and folder != "good":
                raise ValueError(f"Anomalous image in the training split: {row['image']}")
            src = raw / row["image"]
            if not src.is_file():
                raise FileNotFoundError(f"Missing VisA image {src}; is --raw the VisA folder?")
            place(src, out / category / split / folder / src.name)
            if folder == "bad":
                mask_dst = out / category / "ground_truth" / "bad" / f"{src.stem}_mask.png"
                if not mask_dst.exists():
                    mask = np.array(Image.open(raw / row["mask"]))
                    if mask.ndim == 3:
                        mask = mask.max(axis=2)
                    mask_dst.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.where(mask != 0, 255, 0).astype(np.uint8)).save(mask_dst)
            counts[(category, split, folder)] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", default="data/VisA", help="the extracted VisA folder")
    p.add_argument("--out", default="data/visa")
    p.add_argument("--split-file", default="configs/visa_split_1cls.csv")
    args = p.parse_args(argv)
    counts = prepare(Path(args.raw), Path(args.out), Path(args.split_file))
    cats = sorted({c for c, _, _ in counts})
    print(f"{'category':<12} train  test good  test bad")
    for c in cats:
        print(f"{c:<12} {counts[(c, 'train', 'good')]:>5}  {counts[(c, 'test', 'good')]:>9}  "
              f"{counts[(c, 'test', 'bad')]:>8}")  # fmt: skip
    print(f"Done: {sum(counts.values())} images under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
