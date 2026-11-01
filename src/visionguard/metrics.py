"""Evaluation metrics: image/pixel AUROC, AU-PRO and operating-point metrics.

Conventions
-----------
* Higher score = more anomalous. A sample is flagged when ``score >= threshold``.
* ``masks`` are boolean arrays of shape (N, H, W); ``maps`` are float anomaly
  maps of the same shape, at the evaluation resolution.
* AU-PRO follows the MVTec AD definition: per-region overlap averaged over all
  ground-truth connected components (8-connectivity), integrated over the
  pixel false-positive rate from 0 to ``fpr_limit`` (default 0.3) and divided
  by ``fpr_limit`` so a perfect map scores 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve

_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def _as_labels_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError(f"labels {labels.shape} and scores {scores.shape} differ in size.")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or inf.")
    if labels.all() or not labels.any():
        raise ValueError("Need both positive and negative samples.")
    return labels, scores


def _as_masks_maps(masks, maps) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(maps, dtype=np.float64)
    if masks.ndim == 2:
        masks, maps = masks[None], maps[None]
    if masks.ndim != 3 or masks.shape != maps.shape:
        raise ValueError(f"Expected matching (N, H, W) arrays, got {masks.shape} and {maps.shape}.")
    if not np.isfinite(maps).all():
        raise ValueError("anomaly maps contain NaN or inf.")
    if not masks.any() or masks.all():
        raise ValueError("Need both defect and normal pixels.")
    return masks, maps


# ----------------------------------------------------------------- threshold-free


def image_auroc(labels, scores) -> float:
    """Image-level AUROC (labels: 1 = defective)."""
    labels, scores = _as_labels_scores(labels, scores)
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks, maps) -> float:
    """Pixel-level AUROC over all pixels of all images."""
    masks, maps = _as_masks_maps(masks, maps)
    return float(roc_auc_score(masks.ravel(), maps.ravel()))


def pro_curve(masks, maps) -> tuple[np.ndarray, np.ndarray]:
    """Exact PRO curve: (pixel FPR, mean per-region overlap) at every distinct threshold.

    Every ground-truth region gets equal weight, so a tiny scratch counts as much
    as a large stain; that is the point of PRO compared with pixel AUROC.
    Both returned arrays start at (0, 0) and are non-decreasing.
    """
    masks, maps = _as_masks_maps(masks, maps)

    pro_weight = np.zeros(maps.shape, dtype=np.float64)
    n_regions = 0
    for i in range(masks.shape[0]):
        regions, n = ndimage.label(masks[i], structure=_EIGHT_CONNECTED)
        if n == 0:
            continue
        areas = np.bincount(regions.ravel())
        inv_area = np.zeros(areas.shape, dtype=np.float64)
        inv_area[1:] = 1.0 / areas[1:]
        pro_weight[i] = inv_area[regions]
        n_regions += n
    pro_weight /= n_regions

    normal = ~masks
    fp_weight = normal / normal.sum()

    order = np.argsort(maps.ravel(), kind="stable")[::-1]  # descending score
    sorted_scores = maps.ravel()[order]
    fpr = np.cumsum(fp_weight.ravel()[order])
    pro = np.cumsum(pro_weight.ravel()[order])

    # A threshold admits all tied pixels at once: keep the last index of each run.
    last_of_run = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    fpr = np.r_[0.0, np.minimum(fpr[last_of_run], 1.0)]
    pro = np.r_[0.0, np.minimum(pro[last_of_run], 1.0)]
    return fpr, pro


def au_pro(masks, maps, fpr_limit: float = 0.3) -> float:
    """Area under the PRO curve up to ``fpr_limit``, normalised to [0, 1]."""
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError("fpr_limit must be in (0, 1].")
    fpr, pro = pro_curve(masks, maps)
    idx = int(np.searchsorted(fpr, fpr_limit, side="right"))
    x, y = fpr[:idx], pro[:idx]
    if idx < len(fpr) and x[-1] < fpr_limit:
        y_lim = np.interp(fpr_limit, fpr[idx - 1 : idx + 1], pro[idx - 1 : idx + 1])
        x, y = np.r_[x, fpr_limit], np.r_[y, y_lim]
    return float(np.trapezoid(y, x) / fpr_limit)


# ----------------------------------------------------------------- operating points


@dataclass(frozen=True)
class OperatingPoint:
    """Decision rule ``score >= threshold`` and what it does on a labelled set."""

    threshold: float
    recall: float  # share of defective images flagged (TPR)
    fpr: float  # share of defect-free images flagged (false rejects)

    @property
    def false_rejects_per_1000(self) -> float:
        """Good parts rejected per 1,000 good parts inspected."""
        return 1000.0 * self.fpr


def fpr_at_recall(labels, scores, recall: float = 0.95) -> OperatingPoint:
    """Lowest-FPR operating point that catches at least ``recall`` of defective images.

    The threshold is chosen *using the test labels*, so this is an oracle
    operating point: a threshold-free summary like AUROC, not a deployable rule.
    """
    if not 0.0 < recall <= 1.0:
        raise ValueError("recall must be in (0, 1].")
    labels, scores = _as_labels_scores(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    idx = int(np.argmax(tpr >= recall - 1e-12))
    return OperatingPoint(threshold=float(thr[idx]), recall=float(tpr[idx]), fpr=float(fpr[idx]))


def operating_point_at(labels, scores, threshold: float) -> OperatingPoint:
    """Realised recall and FPR of a threshold fixed in advance (e.g. from normal-only data)."""
    labels, scores = _as_labels_scores(labels, scores)
    flagged = scores >= threshold
    return OperatingPoint(
        threshold=float(threshold),
        recall=float(flagged[labels].mean()),
        fpr=float(flagged[~labels].mean()),
    )
