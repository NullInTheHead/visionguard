"""A tiny synthetic MVTec AD tree so tests run without the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SIZE = 64

# category -> (image mode, n train, {test folder: n images})
SPEC = {
    "carpet": ("RGB", 6, {"good": 2, "hole": 2, "cut": 1}),
    "screw": ("L", 5, {"good": 2, "scratch_head": 2}),
}


def _save_image(path: Path, mode: str, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 3 if mode == "RGB" else 1
    arr = rng.integers(0, 255, size=(SIZE, SIZE, channels), dtype=np.uint8)
    Image.fromarray(arr.squeeze()).save(path)


def _save_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((SIZE, SIZE), dtype=np.uint8)
    r0, r1, c0, c1 = box
    m[r0:r1, c0:c1] = 255
    Image.fromarray(m).save(path)


@pytest.fixture(scope="session")
def fake_mvtec(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("mvtec")
    rng = np.random.default_rng(0)
    for cat, (mode, n_train, test) in SPEC.items():
        for i in range(n_train):
            _save_image(root / cat / "train" / "good" / f"{i:03d}.png", mode, rng)
        for defect, n in test.items():
            for i in range(n):
                _save_image(root / cat / "test" / defect / f"{i:03d}.png", mode, rng)
                if defect == "good":
                    continue
                # carpet/cut/000 sits in the top-left corner, outside a 0.875 centre crop.
                box = (0, 3, 0, 3) if (cat, defect, i) == ("carpet", "cut", 0) else (20, 40, 24, 44)
                _save_mask(root / cat / "ground_truth" / defect / f"{i:03d}_mask.png", box)
    # Noise that the indexer must ignore.
    (root / "carpet" / "license.txt").write_text("not an image")
    (root / "carpet" / "train" / "good" / ".DS_Store").write_text("")
    return root
