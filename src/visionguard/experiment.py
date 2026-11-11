"""Shared plumbing for experiment scripts: metrics rows, CSV logging, run metadata, figures."""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Preprocess, Sample
from .metrics import au_pro, fpr_at_recall, image_auroc, pixel_auroc

# ------------------------------------------------------------------ metadata


def git_commit(repo: str | Path = ".") -> str:
    """Short commit hash, with ``-dirty`` if there are uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(device: str) -> dict:
    meta = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "device": device,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        if device.startswith("cuda") and torch.cuda.is_available():
            meta["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import timm

        meta["timm"] = timm.__version__
    except ImportError:
        pass
    return meta


# ------------------------------------------------------------------ metrics


def evaluate(labels, scores, scores_max, masks, maps) -> dict:
    """The protocol's threshold-free metrics plus the oracle 95%-recall operating point."""
    op = fpr_at_recall(labels, scores, recall=0.95)
    return {
        "image_auroc": image_auroc(labels, scores),
        "image_auroc_max": image_auroc(labels, scores_max),
        "pixel_auroc": pixel_auroc(masks, maps),
        "au_pro": au_pro(masks, maps, fpr_limit=0.3),
        "fpr_at_95_recall": op.fpr,
        "rejects_per_1000_at_95_recall": op.false_rejects_per_1000,
        "threshold_at_95_recall": op.threshold,
    }


# ------------------------------------------------------------------ CSV logging


def upsert_rows(path: str | Path, rows: Sequence[dict], key: Sequence[str]) -> pd.DataFrame:
    """Append rows to a CSV, replacing earlier rows with the same key (reruns don't duplicate)."""
    path = Path(path)
    new = pd.DataFrame(list(rows))
    if path.is_file():
        old = pd.read_csv(path)
        if set(key) <= set(old.columns):
            merged_keys = new[list(key)].astype(str).agg("|".join, axis=1)
            old_keys = old[list(key)].astype(str).agg("|".join, axis=1)
            old = old[~old_keys.isin(set(merged_keys))]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def scores_frame(pred, split: str) -> pd.DataFrame:
    """Per-image scores for later analysis (thresholds, failure taxonomy)."""
    return pd.DataFrame(
        {
            "id": pred.ids,
            "split": split,
            "label": pred.labels,
            "defect_type": pred.defect_types,
            "score": pred.scores,
            "score_max": pred.scores_max,
        }
    )


# ------------------------------------------------------------------ figures


def heatmap_grid(
    samples: Sequence[Sample],
    pred,
    preprocess: Preprocess,
    out_path: str | Path,
    per_row: int = 4,
    title: str = "",
) -> Path:
    """Three rows: top-scoring defects (true positives), top-scoring good images (false
    positives) and lowest-scoring defects (misses). Heatmap colour scale is shared."""
    import matplotlib.pyplot as plt
    from PIL import Image

    by_id = {s.id: s for s in samples}
    order = np.argsort(pred.scores)
    defect = [i for i in order[::-1] if pred.labels[i] == 1]
    good = [i for i in order[::-1] if pred.labels[i] == 0]
    rows = [
        ("Highest-scoring defects", defect[:per_row]),
        ("Highest-scoring good parts", good[:per_row]),
        ("Lowest-scoring defects", list(reversed(defect[-per_row:])) if defect else []),
    ]
    vmin, vmax = np.percentile(pred.maps, [1.0, 99.9])

    fig, axes = plt.subplots(3, per_row, figsize=(2.3 * per_row, 7.6), squeeze=False)
    for r, (label, idxs) in enumerate(rows):
        for c in range(per_row):
            ax = axes[r, c]
            ax.set_axis_off()
            if c >= len(idxs):
                continue
            i = idxs[c]
            with Image.open(by_id[pred.ids[i]].path) as im:
                img = preprocess.image(im).transpose(1, 2, 0)
            ax.imshow(img)
            ax.imshow(pred.maps[i], cmap="jet", alpha=0.45, vmin=vmin, vmax=vmax)
            if pred.masks[i].any():
                ax.contour(pred.masks[i], levels=[0.5], colors="white", linewidths=1.0)
            ax.set_title(f"{pred.defect_types[i]}  {pred.scores[i]:.2f}", fontsize=7)
        axes[r, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
        )
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def example_panels(
    samples: Sequence[Sample],
    pred,
    preprocess: Preprocess,
    out_path: str | Path,
    n: int = 4,
    title: str = "",
) -> Path:
    """Image | ground truth | predicted heatmap, for ``n`` defective images spread across
    the score range (best-detected at the top, hardest at the bottom)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    by_id = {s.id: s for s in samples}
    defect = [i for i in np.argsort(pred.scores)[::-1] if pred.labels[i] == 1]
    if not defect:
        raise ValueError("No defective images to show.")
    picks = sorted({defect[round(q * (len(defect) - 1))] for q in np.linspace(0, 1, n)},
                   key=lambda i: -pred.scores[i])  # fmt: skip
    vmin, vmax = np.percentile(pred.maps, [1.0, 99.9])

    fig, axes = plt.subplots(len(picks), 3, figsize=(7.2, 2.4 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        with Image.open(by_id[pred.ids[i]].path) as im:
            img = preprocess.image(im).transpose(1, 2, 0)
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(pred.masks[i], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(img)
        axes[r, 2].imshow(pred.maps[i], cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        axes[r, 0].set_ylabel(f"{pred.defect_types[i]}\nscore {pred.scores[i]:.2f}", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, name in zip(axes[0], ["Image", "Ground truth", "Predicted heatmap"], strict=True):
        ax.set_title(name, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------ efficiency


def reset_device_peak(device) -> None:
    """Start a fresh peak-memory window (CUDA only; other devices have no peak counter)."""
    import torch

    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def efficiency_columns(model, pred) -> dict:
    """Protocol section 13: fit stages (s), scoring stages (ms per image), bank, memory."""
    from .patchcore import device_memory_mb, peak_rss_mb

    cols = {f"fit_{k}_s": round(v, 3) for k, v in model.fit_timing.items()}
    cols.update({f"ms_{k}": round(1000 * v, 2) for k, v in pred.timing.items()})
    cols["peak_rss_mb"] = round(peak_rss_mb(), 1)
    cols.update({k: round(v, 1) for k, v in device_memory_mb(model.device).items()})
    return cols


# ------------------------------------------------------------------ E2 figure


def recovery_figure(
    e2: pd.DataFrame,
    e1: pd.DataFrame | None,
    out_path: str | Path,
    metric: str = "image_auroc",
    title: str = "",
    series: Sequence[tuple[str, str | None]] = (("e2b", None),),
    include_zero_shot: bool = True,
) -> Path:
    """Few-shot recovery curve per category: k = 0 (E2a), 1-8 (mean ± sd over seeds) for
    each experiment in ``series`` (e.g. E2b and E2c variants), and k = all (E1) as a
    dashed line. The x axis is categorical, so the uneven steps are shown evenly."""
    import matplotlib.pyplot as plt

    names = {"image_auroc": "Image AUROC", "au_pro": "AU-PRO", "pixel_auroc": "Pixel AUROC"}
    label = names.get(metric, metric)
    cats = list(dict.fromkeys(e2.category))
    ks = sorted(e2[e2.experiment.isin([e for e, _ in series])].k.unique())
    has_zero = include_zero_shot and (e2.experiment == "e2a").any()
    xlabels = (["0"] if has_zero else []) + [str(int(k)) for k in ks]
    offset = 1 if has_zero else 0
    ncols = min(3, len(cats))
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    lows = []
    for ax, cat in zip(axes.ravel(), cats, strict=False):
        ax.set_visible(True)
        d = e2[e2.category == cat]
        for n, (experiment, name) in enumerate(series):
            xs, means, sds = [], [], []
            if n == 0 and has_zero:
                zero = d[d.experiment == "e2a"][metric]
                if len(zero):
                    xs.append(0)
                    means.append(zero.mean())
                    sds.append(0.0)
            for i, k in enumerate(ks):
                v = d[(d.experiment == experiment) & (d.k == k)][metric]
                if len(v):
                    xs.append(i + offset)
                    means.append(v.mean())
                    sds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
            if xs:
                ax.errorbar(np.array(xs) + 0.06 * n, means, yerr=sds, marker="o", ms=4,
                            capsize=3, lw=1.2, color=f"C{n}", label=name)  # fmt: skip
                lows.append(min(m - s for m, s in zip(means, sds, strict=True)))
        if e1 is not None:
            full = e1[(e1.category == cat) & (e1.calibration_fraction == 0.1)][metric]
            if len(full):
                ax.axhline(full.iloc[0], color="0.4", ls="--", lw=1)
                ax.annotate("all", (1.0, full.iloc[0]), xycoords=("axes fraction", "data"),
                            xytext=(-3, 3), textcoords="offset points", ha="right",
                            fontsize=8, color="0.3")  # fmt: skip
        ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_xlabel("reference images of the new product (k)", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(cat, fontsize=9)
        ax.set_ylim(min([0.5] + [v - 0.03 for v in lows]), 1.02)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        if len(series) > 1 and any(name for _, name in series):
            ax.legend(fontsize=7, loc="lower right")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
