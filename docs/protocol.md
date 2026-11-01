# VisionGuard evaluation protocol

**Status: DRAFT v0.2.** Four decisions are still open (D1–D4, section 10). Once they are
settled, this file and `configs/split.yaml` are committed as final **before any experiment
touches a held-out category**. After that, changes go in the change log with a reason.

## 1. Research question

How well does a pretrained-feature anomaly detector hold up under three distinct kinds of
shift — new defects on familiar products, entirely new product categories, and changes in
image acquisition — and how many defect-free reference images does it need to recover
performance on an unfamiliar product?

The three shifts are evaluated by separate experiments with separate protocols and are
**never merged into a single "generalization" number**.

## 2. Data

MVTec AD: 15 categories (5 textures, 10 objects), 3,629 defect-free training images and
1,725 test images (467 defect-free, 1,258 defective with pixel masks; counts verified
against the published numbers by `scripts/inspect_data.py`). Licence CC BY-NC-SA 4.0.
The dataset has no validation split.

### 2.1 Category split (`configs/split.yaml`, decision D1)

| Role | Textures | Objects |
|---|---|---|
| Development (10) | leather, tile, wood | bottle, cable, capsule, metal_nut, pill, toothbrush, zipper |
| Held-out (5) | carpet, grid | hazelnut, screw, transistor |

* Development categories are used for every choice: layers, image size, aggregation,
  coreset ratio, threshold quantile, demo settings.
* Held-out categories are evaluated only with frozen settings, once per reported
  configuration. Their results are never used to change anything.

### 2.2 Reference and calibration images (decision D3)

Within each category, the defect-free training images are split once, by a seeded draw
(`data.calibration_split`, seed 0), into:

* **reference images** (90%) — the only images that may enter a memory bank;
* **calibration images** (10%) — never in a bank; scored only to set a deployable
  decision threshold without seeing any defect.

The Phase 2 reproduction check against anomalib and the published PatchCore numbers uses
all training images (`calibration.fraction: 0`), since that is the published protocol.

## 3. Preprocessing

* Square resize, then centre crop keeping 87.5% of each side (256 → 224 for WideResNet-50).
  Vision transformers keep the same crop fraction at their own input size (D2), so every
  backbone sees the same field of view.
* Images: bilinear resampling. Masks: nearest-neighbour, so they stay binary.
* Normalisation is backbone-specific (ImageNet vs CLIP statistics).
* Pixel metrics are computed at the cropped evaluation resolution. A defect cut away by the
  crop keeps its defective label.
* **Crop check (done, from dataset statistics only).** Of the 1,258 defective test images,
  7 (0.6%) lose more than half their defect area to the 87.5% centre crop (tile 3; carpet,
  grid, screw, wood 1 each). This is negligible, so the crop stays, which also matches the
  published PatchCore protocol. Per-image values: `results/data/defect_images.csv`.

## 4. Method (finalised in Phase 2)

PatchCore without training: patch features from intermediate backbone layers, 3×3 local
neighbourhood aggregation, greedy coreset subsampling of the memory bank, nearest-neighbour
distance as patch anomaly score, image score = maximum patch score, anomaly map upsampled
bilinearly and smoothed with a Gaussian (σ = 4 px). Defaults follow the PatchCore paper;
any change is chosen on development categories only.

## 5. Experiments

| | Memory bank built from | Scored on | Repeats |
|---|---|---|---|
| **E1** In-category | All reference images of the same category (coreset) | That category's test set | 1 (deterministic) |
| **E2a** New category, zero-shot | Union of per-category coresets of the 10 development categories; nothing from the target | Each held-out category's test set | 1 |
| **E2b** New category, few-shot | k reference images of the target category, k ∈ {1, 2, 4, 8}, full patch bank (no coreset) | Each held-out category's test set | 5 seeds (0–4) |
| **E3** Acquisition shift | All reference images, clean | Test images with one corruption applied | 5 types × 5 severities |
| **E3b** Shift recovery (D4) | Clean bank plus k corrupted reference images | Corrupted test images | 5 seeds |

* E1 runs on development categories while choosing settings, and on held-out categories once
  settings are frozen (it is the upper bound of the few-shot curve).
* The few-shot recovery curve for a held-out category runs from k = 0 (E2a) through
  k = 1, 2, 4, 8 (E2b) to k = all (E1).
* E2b may also be run on development categories while developing; only held-out numbers
  are reported as the result.
* E2a builds the shared bank as a union of per-category coresets (equal ratio per category)
  because the raw union of ten categories' patches does not fit in memory.
* E3 corruptions follow the ImageNet-C definitions (Hendrycks & Dietterich, 2019):
  brightness, contrast, Gaussian blur, Gaussian noise, JPEG compression, severities 1–5.
  They are applied to test images only, after resizing to the backbone's input size; blur
  kernels scale with input size so the physical blur is the same for every backbone.

## 6. Metrics

All implemented in `src/visionguard/metrics.py` and unit-tested (AU-PRO against a
brute-force reference).

* **Image AUROC** — image score = maximum of the anomaly map.
* **Pixel AUROC** — over all test pixels of the category.
* **AU-PRO@0.3** — per-region overlap averaged over all ground-truth connected components
  (8-connectivity), integrated over pixel FPR from 0 to 0.3, divided by 0.3.
* **FPR at 95% recall** — the lowest image-level false-positive rate that still flags 95%
  of defective images, also expressed as good parts rejected per 1,000. The threshold is
  picked with test labels, so this is an *oracle* operating point (threshold-free, like AUROC).
* **Deployable operating point (D3)** — threshold set from the category's calibration
  images only (no defects seen); the rule is chosen on development categories (see D3).
  Report the realised recall and false-reject rate on the test set.
* **Efficiency (Phase 4)** — memory-bank size (patches and MB), peak memory, per-image
  latency on CPU and GPU at coreset ratios 1%, 10%, 25%.

Numbers are reported per category, plus the mean over categories. Few-shot results are
reported as mean ± standard deviation over seeds.

## 7. Few-shot sampling

Implemented in `data.sample_k_shot` and covered by tests:

* references are drawn only from the category's reference pool (`train/good` minus
  calibration images); passing any test image raises an error;
* draws are seeded per (seed, category) and independent of file-listing order;
* draws are **nested**: for a given seed, the k = 1 image is part of the k = 2 draw, which
  is part of the k = 4 draw, and so on. Differences along the curve therefore come from the
  added images rather than from a fresh random draw.

## 8. Leakage rules

1. No setting is chosen while looking at held-out category results.
2. No threshold is chosen on the images it is reported on. Report threshold-free metrics,
   the oracle operating point (labelled as such), and the calibration-based threshold.
3. Few-shot references never overlap test images (structurally enforced and tested).
4. Few-shot results always use 5 seeds; a single-seed number is never reported.
5. The split, this protocol and every config are committed before the runs they govern.

## 9. Reproducibility

* Every run writes a CSV row with: experiment, category, backbone, k, seed, metrics,
  config file, git commit, package versions, GPU type.
* Fixed seeds everywhere; one command per experiment (`scripts/run_e1.py`, …).
* Results tables and figures in `results/` are regenerated from the CSVs by
  `scripts/make_figures.py`, never edited by hand.

## 10. Open decisions

**D1 — Held-out split.** *Proposed:* the purposive split in 2.1, chosen from dataset
properties before any results: stochastic vs regular texture (carpet, grid); free-pose
natural object (hazelnut); free-pose object with tiny defects (screw); aligned object with
logical defects (transistor). *Alternative:* a stratified random draw (2 textures,
3 objects) with a recorded seed — more neutral, but may land on five easy categories and
flatten the few-shot curve.

**D2 — Backbones.** *Roadmap:* WideResNet-50, DINOv2 ViT-S/14, CLIP ViT-B/16, all at 224 px.
At 224 px those give 28×28, 16×16 and 14×14 patch grids, and ViT-S is a quarter the size of
ViT-B, so differences would mix pretraining with model size and spatial resolution.
*Proposed:* WideResNet-50 at 224 px as the PatchCore reference, plus three ViT-B models that
differ only in pretraining — supervised (ImageNet-21k → 1k) ViT-B/16, DINOv2 ViT-B/14 and
CLIP ViT-B/16 — with input sizes chosen so every backbone gives a 28×28 grid (448, 392 and
448 px). Remaining confound, stated in the limitations: the three were pretrained on
different data as well as with different objectives, and CLIP runs above its training
resolution.

**D3 — Calibration split and deployable threshold.** *Proposed:* yes, 10% of each
category's training normals (section 2.2), so the factory-line paragraph and the demo
threshold do not rely on test labels. Calibration sets are small (toothbrush 6 images, the
others 21–39), too few to estimate a 99th percentile, which with fewer than 100 images is
just the maximum. *Proposed rule:* threshold = max calibration score × (1 + m), with the
margin m chosen on development categories to reach a target false-reject rate, then frozen.

**D4 — E3b shift recovery.** *Proposed:* include if time allows (about one extra day), on
two corruptions (brightness, blur) at severity 3 with k ∈ {1, 4, 8}, so every shift gets
the same two questions: how much does it hurt, and how many images bring it back?

## Change log

* v0.1 (2026-10-24) — first draft with open decisions D1–D4.
* v0.2 (2026-11-01) — Phase 1 data inventory: counts verified, centre crop kept, calibration-size issue added to D3.
