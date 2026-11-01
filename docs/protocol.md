# VisionGuard evaluation protocol

**Status: FINAL v1.0 (2026-11-01).** Decisions D1–D4 are settled (section 10). This file
and `configs/split.yaml` were committed before any experiment touched a held-out category.
Every later change goes in the change log with a reason, and no change may be motivated by
held-out results.

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

### 2.1 Category split (`configs/split.yaml`, frozen; decision D1)

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

* Square resize, then centre crop keeping 87.5% of each side, so every backbone sees the
  same field of view. Input sizes are chosen so every backbone gives a 28×28 patch grid (D2):

  | Backbone | timm model | Resize → crop | Patch grid | Layers |
  |---|---|---|---|---|
  | WideResNet-50 | `wide_resnet50_2.tv_in1k` | 256 → 224 | 28×28 (layer 3 upsampled from 14×14) | layer 2 + layer 3 |
  | Supervised ViT-B/16 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 512 → 448 | 28×28 | blocks 4 + 8 of 12 |
  | DINOv2 ViT-B/14 | `vit_base_patch14_dinov2.lvd142m` | 448 → 392 | 28×28 | blocks 4 + 8 of 12 |
  | CLIP ViT-B/16 | `vit_base_patch16_clip_224.openai` | 512 → 448 | 28×28 | blocks 4 + 8 of 12 |

* Images: bilinear resampling. Masks: nearest-neighbour, so they stay binary.
* Normalisation uses each model's own pretraining statistics (from its timm config).
* Pixel metrics are computed at the cropped evaluation resolution. A defect cut away by the
  crop keeps its defective label.
* **Crop check (done, from dataset statistics only).** Of the 1,258 defective test images,
  7 (0.6%) lose more than half their defect area to the 87.5% centre crop (tile 3; carpet,
  grid, screw, wood 1 each). This is negligible, so the crop stays, which also matches the
  published PatchCore protocol. Per-image values: `results/data/defect_images.csv`.

## 4. Method

PatchCore (Roth et al., CVPR 2022), no training. Implemented in `src/visionguard/patchcore.py`;
defaults follow the paper and anomalib.

1. **Patch features.** Two intermediate layers (table in section 3). Each layer's map gets
   3×3 average pooling (stride 1) so every patch carries its neighbourhood; the coarser map
   is upsampled bilinearly to 28×28. Each layer's channels are adaptive-average-pooled to
   1,024; the two vectors are concatenated (2,048) and adaptive-average-pooled to 1,024, as
   in the original PatchCore code. That last step averages adjacent dimensions *within*
   each layer's vector (the first 512 output dimensions come from the first layer, the last
   512 from the second); it is not an average across layers. anomalib instead concatenates
   the raw channels (1,536 for WRN-50) without reduction; the reproduction check (section 6)
   shows the two agree within 0.5 AUROC points.
2. **Memory bank.** All patch vectors of the reference images, reduced by greedy coreset
   (k-centre) selection to 10% (distances computed on a random 128-dimensional projection,
   as in the paper). Few-shot banks (E2b) keep every patch, since k ≤ 8 images is small.
3. **Patch score.** Euclidean distance to the nearest memory-bank vector.
4. **Image score.** Maximum patch score, reweighted by how isolated its nearest bank vector
   is (paper eq. 7; softmax over its 9 nearest bank neighbours, as in anomalib). The plain
   maximum is logged too.
5. **Anomaly map.** 28×28 patch scores upsampled bilinearly to the input size and smoothed
   with a Gaussian, σ = 4 px.

ViT layer indices are a starting point. A small layer sweep on development categories is
allowed in Phase 3, before any held-out run; the outcome goes in the change log.

## 5. Experiments

| | Memory bank built from | Scored on | Repeats |
|---|---|---|---|
| **E1** In-category | All reference images of the same category (coreset) | That category's test set | 1 (deterministic) |
| **E2a** New category, zero-shot | Union of per-category coresets of the 10 development categories; nothing from the target | Each held-out category's test set | 1 |
| **E2b** New category, few-shot | k reference images of the target category, k ∈ {1, 2, 4, 8}, full patch bank (no coreset) | Each held-out category's test set | 5 seeds (0–4) |
| **E3** Acquisition shift | All reference images, clean | Test images with one corruption applied | 5 types × 5 severities |
| **E3b** Shift recovery | Clean coreset plus all patches of k corrupted reference images, k ∈ {1, 4, 8} | Test images with brightness or blur, severity 3 | 5 seeds |

* E1 runs on development categories while choosing settings, and on held-out categories once
  settings are frozen (it is the upper bound of the few-shot curve).
* The few-shot recovery curve for a held-out category runs from k = 0 (E2a) through
  k = 1, 2, 4, 8 (E2b) to k = all (E1).
* E2b may also be run on development categories while developing; only held-out numbers
  are reported as the result.
* E2a builds the shared bank as a union of per-category coresets (equal ratio per category)
  because the raw union of ten categories' patches does not fit in memory.
* E3 and E3b run on all 15 categories with frozen settings (nothing is tuned on corrupted
  data); held-out categories are also shown separately.
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
* **Deployable operating point** — threshold = (1 + m) × the maximum image score over the
  category's calibration images, so no defect is seen. The margin m is chosen once on
  development categories (target: 1% false rejects on their defect-free test images) and
  then frozen: **m = 0.13** (v1.1). Report the realised recall and false-reject rate on the
  test set; on development categories these are tuning diagnostics (section 8, rule 2).
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
2. No threshold is chosen on the images it is reported on, with one documented exception:
   the single margin m (section 6) is tuned on the development categories' defect-free test
   images, so development-category operating points at m are **in-sample tuning
   diagnostics**. They are reported as such, next to a leave-one-category-out estimate (m
   chosen on the other nine development categories). The held-out categories, evaluated
   with m frozen, are the independent test of the deployable threshold. Report
   threshold-free metrics, the oracle operating point (labelled as such), and the
   calibration-based threshold.
3. Few-shot references never overlap test images (structurally enforced and tested).
4. Few-shot results always use 5 seeds; a single-seed number is never reported.
5. The split, this protocol and every config are committed before the runs they govern.

## 9. Reproducibility

* Every run writes a CSV row with: experiment, category, backbone, k, seed, metrics,
  config file, git commit, package versions, GPU type.
* Fixed seeds everywhere; one command per experiment (`scripts/run_e1.py`, …).
* Results tables and figures in `results/` are regenerated from the CSVs by
  `scripts/make_figures.py`, never edited by hand.

## 10. Decisions (settled 2026-11-01)

**D1 — Held-out split: purposive.** Chosen from dataset properties before any results:
stochastic vs regular texture (carpet, grid); free-pose natural object (hazelnut); free-pose
object with the smallest defects (screw, median defect 0.29% of the image); aligned object
with logical defects (transistor). A random draw was rejected because it could land on five
easy categories and flatten the few-shot curve. Stated as a limitation: the held-out set was
chosen to be informative, so its numbers are not an unbiased estimate of average difficulty.

**D2 — Backbones: matched ViT-B.** The roadmap's WideResNet-50, DINOv2 ViT-S/14 and CLIP
ViT-B/16 at 224 px would give 28×28, 16×16 and 14×14 grids, and ViT-S is a quarter the size
of ViT-B, so differences would mix pretraining with model size and spatial resolution.
Instead: WideResNet-50 as the PatchCore reference, plus three ViT-B models that differ in
pretraining (supervised, DINOv2 self-supervised, CLIP image–text), all at a 28×28 grid
(section 3). Remaining confounds, stated as limitations: the ViTs were pretrained on
different data as well as with different objectives, and they run above their training
resolution (position embeddings interpolated).

**D3 — Calibration split: yes, 10%.** Calibration sets are small (toothbrush 6 images, the
others 21–39), too few for a 99th-percentile threshold, which with fewer than 100 images is
just the maximum. Hence the max-plus-margin rule in section 6.

**D4 — E3b: included.** Brightness and Gaussian blur at severity 3; k ∈ {1, 4, 8} corrupted
reference images (drawn with `sample_k_shot`, corrupted with the same corruption and
severity) added to the clean coreset; 5 seeds. Every shift then answers the same two
questions: how much does it hurt, and how many images bring it back?

## Change log

* v0.1 (2026-10-24) — first draft with open decisions D1–D4.
* v0.2 (2026-11-01) — Phase 1 data inventory: counts verified, centre crop kept, calibration-size issue added to D3.
* v1.0 (2026-11-01) — D1–D4 settled; backbone table, method details and threshold rule fixed. Frozen before any experiment.
* v1.1 (2026-11-01) — Threshold margin fixed at m = 0.13 by `scripts/choose_margin.py` on the WRN-50 E1 scores of the development categories (mean false-reject rate 0.3%, target ≤ 1%). Frozen for all later experiments and backbones.
* v1.2 (2026-11-01) — After external review: (a) rule 2 of section 8 now states that development-category operating points at m are in-sample tuning diagnostics, reported with a leave-one-category-out estimate (mean recall 0.827, mean false-reject rate 1.25% vs 0.3% in-sample); held-out categories are the independent evaluation. (b) Section 4 corrected to describe the embedding aggregation the code has always used (concatenate, then pool adjacent dimensions, as in the original PatchCore code), not an average across layers. No code or result changed.
