# VisionGuard evaluation protocol

**Status: v1.7 (2026-11-12), frozen.** v1.0 settled decisions D1–D4 (section 10); v1.4
adds the research questions and the analysis plan for Phase 3 (sections 1, 11–14); v1.5
adds post-hoc diagnostics (section 16); v1.6–v1.7 record RQ2 results and designs (sections 11.1–11.2). This file
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
few training images for a given fraction are reported as such. Design and predictions:
section 11.3.

**Exchangeability under shift.** Each method is also applied to E3 corrupted test images,
with calibration on clean images, to measure how the false-reject rate breaks when the
exchangeability assumption fails.

**Leakage restriction.** The held-out categories' results at m = 0.13 have been seen
(protocol v1.3). Therefore any new rule, and any parameter of it, is designed on the
development categories only, with leave-one-category-out estimates; its held-out numbers
are labelled *post-hoc*. A clean independent test needs data not yet looked at
(decision D5: VisA, section 11.5).

### 11.1 Exchangeability check (first result) and follow-up diagnostic

**Result (E1 scores, WRN-50; `scripts/threshold_analysis.py`).** On the development
categories, 13.0% of defect-free test images score above their category's calibration
maximum, against 4.6% expected if calibration and test images were exchangeable. Test
goods score significantly higher than calibration images (one-sided Mann-Whitney,
p < 0.05) in 4 of 10 development categories: leather (47% above the maximum), zipper,
pill, wood. Among held-out categories (post-hoc): carpet (39%) and screw. So the
exchangeability assumption of split conformal fails for these products when the
calibration set is drawn from the training images; a conformal threshold built that way
would not deliver its guarantee there.

**Follow-up diagnostic** (`scripts/calibration_diagnostic.py`, development categories).
Two explanations, with different predictions:

| Explanation | Blocked calibration split (last 10% by file number) | Nearest-image distance |
|---|---|---|
| Near-twins in the training set (a randomly held-back image keeps a near-identical neighbour in the bank) | Calibration scores rise towards the test level; the shift shrinks | Calibration images much closer to the bank than test images under the random split, not under the blocked split |
| Session drift (test images captured under different conditions) | Little change | Test images farther from the bank than calibration images under both splits |

Recorded before the diagnostic is run. Caveat: the blocked split assumes file numbers
follow capture order; if not, it is another random split and only the distances carry
evidence. A mix of both explanations is possible and is reported as such.

**Outcome (2026-11-12).** Near-twins are not supported as the main cause. Leather: the
shift survives the blocked split (+9.1% random, +11.7% blocked) and defect-free test
images are 3.1–3.2× farther from the bank than calibration images under both splits.
Wood: 2.9–3.9× farther under both splits (its score shift is unstable between splits).
Both match session drift. Zipper is mixed: the distance pattern looks like twins (1.8× →
1.05×), but the score shift is unchanged (+10.7% → +10.5%). Pill is inconclusive. The
other six development categories are exchangeable under both splits. With about 22
calibration images, calibration statistics also vary noticeably between splits where
nothing is wrong (capsule +4.6% random, −4.5% blocked).

### 11.2 Deployment calibration (pre-registered)

**Question.** If the drift is between the training session and the test session, can a
threshold calibrated on a few defect-free parts *from the test session* control the
false-reject rate where one calibrated on training images cannot?

**Design** (`scripts/deployment_calibration.py`, from the E1 per-image scores; the memory
bank is the E1 bank and is not changed). For each category, R = 1,000 times: draw half of
the defect-free test images at random as *deployment calibration* images; the other half
are the evaluation goods; every defective test image is used for recall. Each threshold
rule is applied on the same draws:

| Rule | Calibration images | Guarantee claimed |
|---|---|---|
| Training max + margin (current) | held-back training images (E1) | none |
| Training split conformal | held-back training images (E1) | only if exchangeable with test goods |
| Deployment max + margin | deployment calibration images | none |
| Deployment split conformal | deployment calibration images | marginal, by construction (drawn from the same session as the evaluation goods) |

Split conformal: threshold = the ⌈(n + 1)(1 − α)⌉-th smallest calibration score; an image
is rejected when its score is strictly above it. Valid only for n ≥ 1/α − 1; a rule with
too few calibration images at a given α is reported as *not available*, never filled in.

**Targets.** α = 10% (the main target: with half of each test set, 14 of 15 categories
have the 9 calibration images it needs) and α = 5% (4 of 15 categories have the 19 it
needs). α = 1% needs 99 images and is not testable with MVTec AD's test sets; that is
reported as a limit of the data.

**Reported per category, rule and α:** mean false-reject rate on the evaluation goods
over the 1,000 draws (with a 95% interval of the Monte-Carlo mean), the share of draws
whose false-reject rate exceeds α, and mean recall. Development categories are the
evidence; held-out categories are reported as post-hoc.

**Predictions, recorded before the run on real scores.**

1. Deployment split conformal keeps the mean false-reject rate at or below α in every
   category where it is available, up to Monte-Carlo noise.
2. Training split conformal exceeds α in the drifted categories (development: leather,
   zipper, wood, pill; post-hoc: carpet, screw) and stays at or below α in most others.
3. Where the drift is large, deployment calibration raises the threshold, so honest
   false-reject control costs recall. The size of that cost is reported per category.
4. The share of draws exceeding α is substantial even where the mean is controlled,
   because the guarantee is marginal: one deployed threshold can still be unlucky.

**Caveats.** Deployment calibration assumes the calibration parts are verified defect-free.
It needs defect-free images from each new deployment session, which is a real cost on a
production line. The test-session images also serve as evaluation goods, so this measures
validity within one session, not across sessions.

### 11.3 Calibration size (pre-registered)

**Outcome of 11.2 (2026-11-12, `scripts/deployment_calibration.py`).** At α = 10%,
deployment split conformal kept the mean false-reject rate at or below α in all 9
development and all 5 held-out categories where it was available, at its exact expected
rate (prediction 1 held). Training split conformal exceeded α on average in 6 of 9
development categories and 3 of 5 held-out ones, but an exact test of one fixed
calibration draw (the count of rejected goods is beta-binomial(m, n + 1 − k, k) under
exchangeability) puts it beyond chance only in leather and wood (development) and carpet
and screw (post-hoc); pill and zipper were not significant (prediction 2 partly held).
Recall cost where drift is large: pill 0.93 → 0.84, screw 0.99 → 0.87, carpet 1.00 →
0.93 (prediction 3 held). About a quarter of single draws exceeded α even where the mean
was controlled (prediction 4 held).

**Question.** Does a larger calibration set drawn from the *training* images help: more
stable thresholds and a valid conformal threshold at smaller α, at the price of a smaller
memory bank? Or is the failure in drifted products a property of the data that more
calibration images cannot fix?

**Design** (`scripts/calibration_size.py`, WRN-50, frozen E1 settings, development
categories; held-out categories only with `--allow-held-out`, labelled post-hoc).
Calibration fractions 10%, 25% and 50% of each category's training images, three random
calibration draws each (seeds 0, 1, 2; fraction 10% with seed 0 is the E1 split and must
reproduce E1's image AUROC). The bank is a 10% coreset of the remaining images. Reported
per category, fraction and seed: calibration and bank sizes, image AUROC, pixel AUROC,
AU-PRO; for training split conformal at α = 10%, 5% and 1% (where n ≥ 1/α − 1), the
false-reject rate on all defect-free test images, recall, and the exact beta-binomial
p-value; for max + margin (m = 0.13), the false-reject rate and recall.

Calibration sizes (fraction 10 / 25 / 50%): 21–27 / 52–67 / 104–134 images, toothbrush
6 / 15 / 30. α = 1% (99 images) becomes available at 50% in 9 of 10 development
categories, but with 42–167 defect-free test images a 1% rate cannot be resolved; it is
reported, not judged.

**Predictions, recorded before the run.**

1. A smaller bank costs little ranking: mean development image AUROC falls by at most
   0.005 and mean AU-PRO by at most 0.01 from fraction 10% to 50%.
2. More calibration images do not fix drift: at α = 10%, training split conformal stays
   beyond chance (p < 0.05) in leather and wood at every fraction, in at least 2 of 3
   seeds. Because the test gains power with n, pill and zipper may become significant too.
3. Where 11.2 found no drift (tile, capsule, metal_nut, bottle, cable, toothbrush), most
   category-fraction cells stay non-significant, and the spread of the training
   conformal false-reject rate across seeds shrinks as the fraction grows.
4. Max + margin stays far below α everywhere and loses recall as the fraction grows,
   because the maximum of a larger calibration set is higher.

**Outcome (2026-11-12, development categories, 3 seeds; fraction 10% seed 0 reproduces
E1 exactly).**

1. Held. Mean image AUROC 0.990 → 0.989 → 0.988 and AU-PRO 0.919 → 0.918 → 0.918 at
   fractions 10 / 25 / 50%: halving the bank costs almost no ranking.
2. Held. At α = 10%, training split conformal is beyond chance in leather and wood in
   3 of 3 seeds at every fraction (mean false-reject rate: leather 0.93 / 0.92 / 0.85,
   wood 0.30 / 0.42 / 0.42). Zipper becomes significant in 2 of 3 seeds at 25% only;
   pill never. The test gains power with n: at α = 5%, significant cells rise from 4 to 9.
3. Partly held. Where 11.2 found no drift, 0 of 51 cells are beyond chance. The seed
   spread of the false-reject rate does not shrink steadily (mean SD across bottle,
   cable, capsule, metal_nut and tile: 0.047 / 0.027 / 0.037).
4. Held. Max + margin false-reject rate 0.7% → 0.4% → 0.1%; recall 0.79 → 0.73 → 0.66.

Conclusion: a larger training calibration set is nearly free (in ranking) and makes α = 5%
and 1% available, but it does not repair drift; that still needs deployment calibration
(11.2). At fraction 50%, training conformal at α = 1% gives a mean false-reject rate of
2.9%, driven by leather (15.6%), wood (3.5%) and zipper (3.1%).

### 11.4 Extreme-value tail fit (pre-registered)

**Question.** Split conformal needs n ≥ 1/α − 1 calibration images, so a deployment
session with 20–80 defect-free parts can target 10% but rarely 5% or 1%. A fitted tail
can extrapolate to any α, without a guarantee. Does it control the false-reject rate
where conformal is not available, and does it behave like conformal where both are?

**Rule (fixed before any fit on real scores).** Peaks over threshold: with n calibration
scores, the top ⌊0.3 n⌋ form the tail (at least 5 needed, else *not available*); u is the
largest score below them; a generalized Pareto distribution is fitted by maximum
likelihood to the exceedances (location fixed at 0); threshold = u + its
(1 − α n / ⌊0.3 n⌋) quantile. A failed fit is reported as not available. The tail
fraction 0.3 is fixed, not tuned; 0.2 and 0.5 are reported as a sensitivity check only.

**Runs.** (a) *Deployment EVT* is added as a rule to the 11.2 simulation (E1 scores, half
of the defect-free test images as calibration, 1,000 draws), with *training EVT* on the E1
calibration images, at α = 10%, 5% and 1%. (b) *Training EVT* on the 11.3 scores
(fractions 10 / 25 / 50%, 3 seeds), next to training conformal. Development categories are
the evidence; held-out ones post-hoc.

**Predictions, recorded before the run.**

1. Where both are available, EVT and conformal give similar false-reject rates (mean
   absolute difference at most 0.03 at α = 10%, same calibration images), and training
   EVT exceeds α beyond the conformal rate's spread in leather and wood: a tail fit does
   not remove drift.
2. Deployment EVT at α = 5% keeps the mean false-reject rate at or below 2α in at least
   8 of 9 development categories, but above α in some: an approximation, not a guarantee.
3. At α = 1% with 20–80 deployment images, deployment EVT is unreliable: in most
   development categories more than 30% of single draws exceed α.
4. Results move noticeably with the tail fraction (0.2 vs 0.5) at small n, less at n > 100.

**Implementation note.** The maximum-likelihood fit is computed by the one-dimensional
profile likelihood, with the shape kept at or above −1 (below it the likelihood is
unbounded); a unit test checks it fits at least as well as `scipy.stats.genpareto.fit`.

**Outcome (2026-11-12).**

1. Held. On the 11.3 scores at α = 10%, training EVT and training conformal differ by
   0.026 on average (84 cells), and EVT stays far above α in leather (0.90) and wood
   (0.44). EVT is slightly less conservative throughout: on the 11.2 run it exceeds α in
   7 of 9 development categories (conformal 6).
2. Not testable as registered. With half of each test set as deployment calibration, the
   30% tail holds at least 5 points in only 4 of 15 categories (cable among the
   development ones; hazelnut, screw, transistor post-hoc). Cable at α = 5%: 4.2%. The
   three held-out ones: 6.3–7.7%, all above α and below 2α.
3. Held where testable. At α = 1%, deployment EVT gives 2.8–4.1% (3–4× the target), and
   26–47% of single draws exceed α.
4. Held, modestly. At α = 1% the mean false-reject rate across tail fractions 0.2 / 0.3
   / 0.5 is 0.081 / 0.105 / 0.102 with about 22 calibration images and 0.053 / 0.057 /
   0.055 with about 113.

Conclusion: the tail fit does not buy small-α control. Even on products without drift and
with about 113 training calibration images, it gives 2.4% at α = 1% where split conformal
gives 0.7%. Small α needs more calibration images, not extrapolation; split conformal on
deployment images remains the deployable rule.

### 11.5 Clean test on VisA (pre-registered)

**Question.** Does the conclusion of 11.2–11.4 hold on products never used for any
decision: split conformal on deployment calibration images controls the false-reject
rate at α, and how do the other rules behave?

**What VisA can and cannot test.** In VisA's official split, the defect-free test images
of each product are drawn at random from the same pool as its training images (checked
from the split file only: test file numbers spread over the whole range). Training and
test goods are therefore exchangeable by construction. VisA tests validity on new
products without session drift, and whether training calibration then works; it does not
test drift (that would need MVTec AD 2, left for later).

**Frozen settings, nothing tuned on VisA.** E1 exactly as on MVTec AD (`configs/
e1_visa.yaml` extends `configs/e1.yaml`): WRN-50, square resize to 256 and centre crop
224, 10% coreset, 9-neighbour reweighting, σ = 4, calibration fraction 10% (seed 0),
m = 0.13. Then `scripts/threshold_analysis.py` and `scripts/deployment_calibration.py`
unchanged (1,000 half-splits; α = 10%, 5%, 1%). All 12 products are reported, failures
included. Calibration sizes: training 45–91 images (α = 5% available everywhere, α = 1%
nowhere); deployment 25–50 images (α = 10% and 5% everywhere).

**Predictions, recorded before any VisA image is loaded.**

1. Ranking is lower than on MVTec AD: mean image AUROC between 0.93 and 0.97.
2. Deployment split conformal keeps the mean false-reject rate at or below α in all 12
   products, at α = 10% and at α = 5%, up to Monte-Carlo noise.
3. With exchangeable data, training split conformal is beyond chance (exact test,
   p < 0.05) in at most 2 of 12 products at α = 10% (about 0.6 expected by chance alone).
4. Max + margin (m = 0.13) keeps the mean false-reject rate across products at or below
   1%, with lower recall than either conformal rule at α = 10% in every product, and mean
   recall below MVTec AD's 0.83.
5. Training EVT at α = 5% exceeds α (beyond its Monte-Carlo interval) in more products
   than training split conformal does.

**Outcome (2026-11-14; E1 run on the Mac, threshold scripts unchanged).**

1. Missed. Mean image AUROC 0.922, below the registered 0.93–0.97. Ten products score
   0.92–0.99; capsules (0.738) and macaroni2 (0.667) pull the mean down. Both still
   localise (pixel AUROC 0.983 and 0.968), the same pattern as screw on MVTec AD.
2. Held. Deployment split conformal kept the mean false-reject rate at or below α in
   12 of 12 products at α = 10% (0.076–0.099) and at α = 5% (0.031–0.042). As on MVTec
   AD the guarantee is an average: 30–38% of single draws exceed 10%, 23–33% exceed 5%.
3. Held. The exchangeability check finds no significant shift in any product (0 of 12,
   one-sided Mann-Whitney), and training split conformal is beyond chance in 1 of 12
   (pcb1) at α = 10% and 5%, although its realised rate exceeds α in 6 of 12 (one fixed
   draw).
4. Held. Max + margin: mean false-reject rate 0.4%, recall below both conformal rules in
   every product, mean recall 0.44 (MVTec AD: 0.83).
5. Held, narrowly. Training EVT exceeds α = 5% in 7 of 12 products, training conformal in
   6. Deployment EVT exceeds α in 6 / 10 / 12 of 12 at α = 10 / 5 / 1%.

At α = 10%, deployment split conformal catches 79% of defects (mean over products); at 5%,
68%. The deployable rule transfers to new products; what limits it is ranking, not
calibration.

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

### 12.1 ViT layer sweep (pre-registered)

**Design** (`scripts/vit_layer_sweep.py`). For each ViT backbone (supervised ViT-B/16,
DINOv2 ViT-B/14, CLIP ViT-B/16), three pairs of 0-based blocks out of 12: (2, 5), (3, 7)
(the starting point of section 3, "blocks 4 + 8") and (5, 9). Everything else is the E1
setting: input size and 28 × 28 grid of section 3, 10% coreset, calibration 10% (seed 0).
E1 on the 10 development categories only; no held-out or VisA image is scored by any ViT
before the sweep is closed.

**Selection rule.** Per backbone, the pair with the highest mean over development
categories of (image AUROC + AU-PRO) / 2. If another pair is within 0.002 of the best, the
one whose deeper block is shallower wins (cheaper). The selected pairs are then frozen for
every ViT run (E1 held-out, E2, E3, VisA) and recorded in the change log.

**Predictions, recorded before the run.**

1. DINOv2 is best with the deepest pair (5, 9); supervised ViT and CLIP are best with
   (2, 5) or (3, 7).
2. No ViT beats WRN-50's development mean image AUROC (0.991) by more than 0.005: the
   development set is near its ceiling, so the comparison has to come from the four
   dimensions of section 12, not this number.
3. CLIP has the lowest AU-PRO of the three at its selected pair.
4. Scoring a ViT costs at most twice WRN-50's time per image on the same machine.

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

## 15. Decision D5 (settled 2026-11-12): VisA is the clean test set

VisA (Zou et al., ECCV 2022; 12 products, 10,821 images, CC BY 4.0) with the authors'
official one-class split (`configs/visa_split_1cls.csv`, copied verbatim). Chosen over
MVTec AD 2 because its test sets (50–101 defect-free images per product) are large enough
to test α = 5% with deployment calibration in every product. No VisA image has been looked
at. Design and predictions: section 11.5.

## 16. Post-hoc diagnostics

Experiments designed **after** seeing held-out results, to explain a failure. They are
labelled post-hoc wherever reported, set no frozen setting, and are not presented as a
tuned method.

### 16.1 Rotation-augmented few-shot banks (E2c)

**Observation that motivated it (E2, WRN-50).** Few-shot screw stays near chance
(image AUROC 0.55 at k = 8, against 0.975 with all images), and few-shot grid falls below
zero-shot at k = 1–4. Yet screw's pixel AUROC is 0.88–0.92: the defect is the hottest
region inside each image, but good screws score as high as defective ones. Good-screw
scores spread over about 1.0 (10th–90th percentile at k = 8), while a defect adds about
0.3 in E1. Hazelnut, which also lies at random angles but is roughly round, recovers to
0.985 by k = 8.

**Hypothesis.** Few-shot adaptation is limited by pose variation of the good parts: with
k references, a good part at a new orientation is far from every stored patch.

**Design.** E2b exactly (same targets, k, seeds and draws), but each reference image also
enters the memory bank rotated about its centre. Two variants:

* **rot90**: 0°, 90°, 180°, 270°. Exact on the square crop, no empty corners.
* **rot45**: 0° to 315° in 45° steps. Rotation leaves empty corners; patches whose cell, or
  any cell within 2 cells of it, touches an empty pixel are left out of the bank.

Rotation is applied to the preprocessed (cropped) reference image; test images are never
rotated.

**Predictions, recorded before any run.**

| Product | Prediction | Why |
|---|---|---|
| screw | Clear gain at every k, larger for rot45 than rot90 | Elongated part, any orientation |
| grid | Gain | Pattern orientation shifts between photos |
| hazelnut | Little change | Roughly round: rotation changes little |
| carpet | Little change | Random texture, already recovered at k = 1 |
| transistor | No gain, possibly a loss | Always photographed upright; a rotated good transistor can resemble a misplaced-part defect |

**Reading the outcome.** If screw does not improve, pose is not the main cause, and the
hypothesis is rejected as stated. If screw improves and transistor does not, pose
variation explains the gap, and the backbone is not the main limit for these products.
Either outcome is reported.

## Change log

* v0.1 (2026-10-24) — first draft with open decisions D1–D4.
* v0.2 (2026-11-01) — Phase 1 data inventory: counts verified, centre crop kept, calibration-size issue added to D3.
* v1.0 (2026-11-01) — D1–D4 settled; backbone table, method details and threshold rule fixed. Frozen before any experiment.
* v1.1 (2026-11-01) — Threshold margin fixed at m = 0.13 by `scripts/choose_margin.py` on the WRN-50 E1 scores of the development categories (mean false-reject rate 0.3%, target ≤ 1%). Frozen for all later experiments and backbones.
* v1.2 (2026-11-01) — After external review: (a) rule 2 of section 8 now states that development-category operating points at m are in-sample tuning diagnostics, reported with a leave-one-category-out estimate (mean recall 0.827, mean false-reject rate 1.25% vs 0.3% in-sample); held-out categories are the independent evaluation. (b) Section 4 corrected to describe the embedding aggregation the code has always used (concatenate, then pool adjacent dimensions, as in the original PatchCore code), not an average across layers. No code or result changed.
* v1.3 (2026-11-01) — Coreset implementation: computed on the CPU with squared distances from one matrix-vector product per step (about 10x faster; selection no longer depends on the GPU type). Floating-point differences break some near-ties differently, so a few late coreset picks change; E1 is rerun on all 15 categories with this code. The margin stays frozen at m = 0.13: it is not re-tuned, because a preview of the held-out threshold results (mean recall 0.80, carpet false-reject rate 14%) has been seen.
* v1.4 (2026-11-11) — Research questions RQ1–RQ4; threshold calibration study with its success criterion, the precise conformal guarantee and the leakage restriction (section 11); backbone reporting on four dimensions with frozen defect-size bins (section 12); efficiency measurements logged in every run (section 13); related work (section 14); open decision D5. E2a coreset ratio fixed at 10%; E2b reports threshold-free metrics only. No setting of v1.3 changed.
* v1.5 (2026-11-11) — Section 16: post-hoc diagnostics; 16.1 rotation-augmented few-shot banks (E2c), with design and predictions recorded before any run. No frozen setting changed.
* v1.6 (2026-11-11) — Section 11.1: exchangeability check result (calibration and test defect-free images not exchangeable in 4 of 10 development categories) and the twins-vs-drift diagnostic with its predictions, recorded before it is run.
* v1.7 (2026-11-12) — Section 11.1 outcome (twins not supported as main cause; drift in leather and wood, zipper mixed); section 11.2 deployment calibration, design and predictions recorded before the run on real scores. α = 10% added as the main testable target (α = 1% needs 99 calibration images, unavailable in MVTec AD's test sets).
* v1.8 (2026-11-12) — Section 11.2 outcome; section 11.3 calibration-size study, design and predictions recorded before the run. No frozen setting changed.
* v1.9 (2026-11-12) — Section 11.3 outcome (3 of 4 predictions held, one partly). No frozen setting changed.
* v1.10 (2026-11-12) — Section 11.4 extreme-value tail fit: rule, runs and predictions recorded before any fit on real scores. No frozen setting changed.
* v1.11 (2026-11-12) — Section 11.4 outcome and the fit's implementation note.
* v1.12 (2026-11-12) — D5 settled: VisA with its official one-class split is the clean test set (section 15); section 11.5 clean test, frozen settings and predictions recorded before any VisA image is loaded.
* v1.13 (2026-11-14) — Section 11.5 outcome on VisA: 4 of 5 predictions held, the ranking prediction missed. No setting changed.
* v1.14 (2026-11-14) — Section 12.1 ViT layer sweep: candidates, selection rule and predictions recorded before any ViT run.
