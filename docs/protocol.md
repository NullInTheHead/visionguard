# VisionGuard evaluation protocol

**Status: v1.4 (2026-11-11), frozen.** v1.0 settled decisions D1–D4 (section 10); v1.4
adds the research questions and the analysis plan for Phase 3 (sections 1, 11–14). This file
and `configs/split.yaml` were committed before any experiment touched a held-out category.
Every later change goes in the change log with a reason, and no change may be motivated by
held-out results.

## 1. Research questions

**Guiding question.** Can we build an industrial anomaly detector that remains useful when
the product, the reference data and the imaging conditions change?

"Useful" means at a deployable operating point (a threshold fixed without seeing any
defect), not only by threshold-free ranking metrics such as AUROC.

| | Question | Experiments | Section |
|---|---|---|---|
| **RQ1** Product shift | How well does the detector work on a product it has never seen, and how many defect-free reference images restore it? | E2a, E2b | 5 |
| **RQ2** Reference-data shift | Can we predict and control the false-reject rate on previously unseen products, using a calibration procedure designed without access to their test distributions? | Threshold study on E1/E2 scores | 11 |
| **RQ3** Imaging shift | How much do lighting, blur, noise and compression hurt, and how many corrupted reference images bring performance back? | E3, E3b | 5 |
| **RQ4** Backbones (across RQ1–3) | Which pretraining handles tiny defects, lighting changes and few-shot adaptation best, and at what compute cost? | All, per backbone | 12 |

The shifts are evaluated by separate experiments and are **never merged into a single
"generalization" number**.

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
* E2a builds the shared bank as a union of per-category coresets (10% of each development
  category's reference patches, the E1 setting) because the raw union of ten categories'
  patches does not fit in memory.
* E2b uses the nested draws of section 7 from the target category's reference pool
  (calibration images excluded). It reports threshold-free metrics and the oracle FPR at
  95% recall. A deployable threshold with only k images is part of the threshold study
  (section 11), not of E2b.
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

## 11. Threshold calibration study (RQ2)

**Success criterion.** Not "reach 1% false rejects on these categories", but: for a target
rate α, does a calibration procedure designed on development categories predict and
control the false-reject rate on products whose test data it never saw?

Reported per method and per category: the targeted α, the realised false-reject rate on
defect-free test images, the realised recall, and across categories how often the
realised rate exceeds α (the violation rate). Targets: α ∈ {1%, 5%}.

**Methods compared.**

1. **Max plus margin** (current rule, m = 0.13): threshold = (1 + m) × the highest
   calibration score.
2. **Split conformal**: one fixed memory bank; n defect-free calibration images that never
   enter it are scored against it; threshold = the ⌈(n + 1)(1 − α)⌉-th smallest calibration
   score. Under exchangeability of calibration and test defect-free images, a new
   defect-free image exceeds it with probability at most α. The guarantee is *marginal*
   (over the random calibration draw, not for one fixed threshold) and is non-trivial only
   when n ≥ 1/α − 1 (99 images for α = 1%, 19 for α = 5%). Below that, the method has no
   valid threshold at that α, and that is reported, not hidden.
3. **Extreme-value tail fit**: a generalized Pareto distribution fitted to the upper tail of
   the calibration scores, threshold = its (1 − α) quantile. No finite-sample guarantee;
   assumes the fitted tail extends to unseen products.
4. **Cross-fitted calibration** (K-fold over the reference images: each fold scored against
   a bank built from the other folds): more calibration scores, but they come from
   different banks, are dependent, and are not exchangeable with a test score against the
   full bank. Reported as a **heuristic, with no guarantee claimed**. Known cross-fitted
   variants carry weaker guarantees (for example 1 − 2α for jackknife+ in regression);
   none is claimed here without a matching result for this setting.

**Calibration size.** Run on development categories with calibration fractions 10%, 25%
and 50% of the training images, to measure the trade-off between a larger calibration set
(a valid conformal threshold) and a smaller memory bank (lower AUROC). Categories with too
few training images for a given fraction are reported as such.

**Exchangeability under shift.** Each method is also applied to E3 corrupted test images,
with calibration on clean images, to measure how the false-reject rate breaks when the
exchangeability assumption fails.

**Leakage restriction.** The held-out categories' results at m = 0.13 have been seen
(protocol v1.3). Therefore any new rule, and any parameter of it, is designed on the
development categories only, with leave-one-category-out estimates; its held-out numbers
are labelled *post-hoc*. A clean independent test needs data not yet looked at
(decision D5).

## 12. Backbone comparison (RQ4)

Each backbone is reported on four dimensions, not one AUROC. Defect-size bins are fixed
here, from the Phase 1 inventory (area as a share of the image):

| Bin | Defect area | Meaning |
|---|---|---|
| Tiny | < 0.128% | smaller than one cell of the 28 × 28 patch grid |
| Small | 0.128% to 1% | |
| Medium | 1% to 5% | |
| Large | ≥ 5% | |

1. **Tiny defects**: per size bin, image AUROC (defective images in the bin against all
   defect-free images) and AU-PRO (defective images in the bin plus all defect-free
   images), on E1.
2. **Lighting robustness**: E3 brightness and contrast curves, severities 1–5.
3. **Few-shot adaptation**: E2b curves, k = 1, 2, 4, 8, mean ± standard deviation over seeds.
4. **Compute cost**: the measurements of section 13.

The ViT layer sweep allowed by section 4 is run on development categories before any
held-out ViT result; its outcome goes in the change log.

## 13. Efficiency measurements

Logged for every run (E1, E2, E3), so cost is measured on the same runs as accuracy:

* **Fit**: feature-extraction time and coreset time, separately.
* **Scoring, per image**: image loading, feature extraction (backbone and patch
  embedding), nearest-neighbour search (including reweighting), and map post-processing,
  each timed with the device synchronised.
* **Memory bank**: number of patches and size in MB.
* **Peak memory**: peak process memory (resident set size), and peak device memory where
  the device reports it (CUDA). Apple MPS reports only current allocation, which is logged
  as such.

Phase 4 adds batch-size-1 latency on CPU and GPU and the coreset sweep (1%, 10%, 25%).

## 14. Related work and positioning

* PatchCore (Roth et al., CVPR 2022) established in-category detection on MVTec AD; E1
  reproduces it and is a validation step, not a contribution.
* Few-shot PatchCore has been tuned across CNN backbones, resolution and coreset
  (Santos et al., 2023, "Optimizing PatchCore for few/many-shot anomaly detection"), and
  WinCLIP (Jeong et al., CVPR 2023) addresses zero- and few-shot detection with CLIP.
  Both report threshold-free metrics.
* Robustness to real acquisition changes is benchmarked by RAD (2024; lighting, viewpoint,
  blur) and MVTec AD 2 (2025; test sets with lighting changes).
* Conformal false-alarm control for industrial anomaly detection has been studied on sensor
  time series; we found no study of controlling false rejects on unseen *image* products.

Positioning: VisionGuard's contribution is RQ2 (deployable false-reject control under
product, reference-data and imaging shift) and the matched-backbone comparison of RQ4
(model size and patch grid held fixed, pretraining varied). A full literature search is
done before any novelty claim.

## 15. Open decision

**D5 — Clean test set for RQ2.** Options: VisA (12 products) or MVTec AD 2 (8 scenarios,
including lighting changes). Settled before any new threshold rule is evaluated on data
outside the development categories.

## Change log

* v0.1 (2026-10-24) — first draft with open decisions D1–D4.
* v0.2 (2026-11-01) — Phase 1 data inventory: counts verified, centre crop kept, calibration-size issue added to D3.
* v1.0 (2026-11-01) — D1–D4 settled; backbone table, method details and threshold rule fixed. Frozen before any experiment.
* v1.1 (2026-11-01) — Threshold margin fixed at m = 0.13 by `scripts/choose_margin.py` on the WRN-50 E1 scores of the development categories (mean false-reject rate 0.3%, target ≤ 1%). Frozen for all later experiments and backbones.
* v1.2 (2026-11-01) — After external review: (a) rule 2 of section 8 now states that development-category operating points at m are in-sample tuning diagnostics, reported with a leave-one-category-out estimate (mean recall 0.827, mean false-reject rate 1.25% vs 0.3% in-sample); held-out categories are the independent evaluation. (b) Section 4 corrected to describe the embedding aggregation the code has always used (concatenate, then pool adjacent dimensions, as in the original PatchCore code), not an average across layers. No code or result changed.
* v1.3 (2026-11-01) — Coreset implementation: computed on the CPU with squared distances from one matrix-vector product per step (about 10x faster; selection no longer depends on the GPU type). Floating-point differences break some near-ties differently, so a few late coreset picks change; E1 is rerun on all 15 categories with this code. The margin stays frozen at m = 0.13: it is not re-tuned, because a preview of the held-out threshold results (mean recall 0.80, carpet false-reject rate 14%) has been seen.
* v1.4 (2026-11-11) — Research questions RQ1–RQ4; threshold calibration study with its success criterion, the precise conformal guarantee and the leakage restriction (section 11); backbone reporting on four dimensions with frozen defect-size bins (section 12); efficiency measurements logged in every run (section 13); related work (section 14); open decision D5. E2a coreset ratio fixed at 10%; E2b reports threshold-free metrics only. No setting of v1.3 changed.
