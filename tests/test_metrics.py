from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from visionguard.metrics import (
    au_pro,
    fpr_at_recall,
    image_auroc,
    operating_point_at,
    pixel_auroc,
    pro_curve,
)

# --------------------------------------------------------------- image level


def test_image_auroc_extremes_and_sklearn_agreement():
    labels = np.array([0, 0, 0, 1, 1])
    assert image_auroc(labels, [0.1, 0.2, 0.3, 0.8, 0.9]) == 1.0
    assert image_auroc(labels, [0.9, 0.8, 0.7, 0.2, 0.1]) == 0.0
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y
    assert image_auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_image_auroc_rejects_single_class_and_nan():
    with pytest.raises(ValueError, match="both"):
        image_auroc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        image_auroc([0, 1], [0.1, np.nan])


def test_fpr_at_recall_hand_example():
    # 20 defective parts scored 10..29, 20 good parts scored 0..19.
    # 95% recall = 19 of 20 defects -> threshold 11 -> good parts 11..19 flagged = 9/20.
    labels = np.r_[np.ones(20), np.zeros(20)]
    scores = np.r_[np.arange(10, 30), np.arange(0, 20)].astype(float)
    op = fpr_at_recall(labels, scores, recall=0.95)
    assert op.threshold == 11.0
    assert op.recall == pytest.approx(0.95)
    assert op.fpr == pytest.approx(0.45)
    assert op.false_rejects_per_1000 == pytest.approx(450.0)


def test_fpr_at_recall_perfect_separation_is_zero():
    op = fpr_at_recall([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert op.fpr == 0.0 and op.recall == 1.0


def test_operating_point_at_fixed_threshold():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.4])
    op = operating_point_at(labels, scores, threshold=0.5)
    assert op.recall == 0.5 and op.fpr == 0.25


# --------------------------------------------------------------- pixel level


def _brute_force_pro(masks, maps, fpr_limit=0.3):
    """Slow reference: evaluate every distinct threshold explicitly."""
    regions = []
    for i in range(masks.shape[0]):
        lab, n = ndimage.label(masks[i], structure=np.ones((3, 3), int))
        regions += [(i, lab == r) for r in range(1, n + 1)]
    normal = ~masks
    fprs, pros = [0.0], [0.0]
    for t in np.unique(maps)[::-1]:
        pred = maps >= t
        fprs.append((pred & normal).sum() / normal.sum())
        pros.append(np.mean([(pred[i] & r).sum() / r.sum() for i, r in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    x, y = fprs[keep], pros[keep]
    if x[-1] < fpr_limit:
        j = int(np.argmax(fprs > fpr_limit))
        x = np.r_[x, fpr_limit]
        y = np.r_[y, np.interp(fpr_limit, fprs[j - 1 : j + 1], pros[j - 1 : j + 1])]
    return fprs, pros, np.trapezoid(y, x) / fpr_limit


def _random_case(seed, n=4, h=12, w=12, levels=None):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, h, w), bool)
    masks[0, 2:5, 2:6] = True  # two regions in one image
    masks[0, 8:10, 8:11] = True
    masks[2, 5:9, 1:3] = True  # image 1 and 3 are defect-free
    maps = rng.random((n, h, w)) + 0.8 * masks
    if levels:  # quantise to create ties
        maps = np.round(maps * levels) / levels
    return masks, maps


@pytest.mark.parametrize("seed,levels", [(0, None), (1, None), (2, 5), (3, 3)])
def test_pro_matches_brute_force(seed, levels):
    masks, maps = _random_case(seed, levels=levels)
    fpr, pro = pro_curve(masks, maps)
    ref_fpr, ref_pro, ref_area = _brute_force_pro(masks, maps)
    np.testing.assert_allclose(fpr, ref_fpr, atol=1e-12)
    np.testing.assert_allclose(pro, ref_pro, atol=1e-12)
    assert au_pro(masks, maps) == pytest.approx(ref_area, abs=1e-12)


def test_pro_perfect_and_inverted_maps():
    masks, _ = _random_case(0)
    assert au_pro(masks, masks.astype(float)) == pytest.approx(1.0)
    assert au_pro(masks, (~masks).astype(float)) == pytest.approx(0.0)


def test_pro_weights_regions_equally():
    """Missing a tiny region costs as much as missing a big one."""
    masks = np.zeros((1, 20, 20), bool)
    masks[0, 1:3, 1:3] = True  # tiny region (4 px)
    masks[0, 8:18, 8:18] = True  # big region (100 px)
    maps = np.zeros((1, 20, 20))
    maps[0, 8:18, 8:18] = 1.0  # finds only the big one
    fpr, pro = pro_curve(masks, maps)
    assert pro[1] == pytest.approx(0.5)  # at the first threshold: one of two regions covered
    assert fpr[1] == 0.0


def test_pixel_auroc_matches_sklearn():
    masks, maps = _random_case(4)
    assert pixel_auroc(masks, maps) == pytest.approx(roc_auc_score(masks.ravel(), maps.ravel()))


def test_pixel_metrics_validate_shapes():
    masks, maps = _random_case(0)
    with pytest.raises(ValueError, match="matching"):
        au_pro(masks, maps[:, :-1])
    with pytest.raises(ValueError, match="defect and normal"):
        au_pro(np.zeros_like(masks), maps)
