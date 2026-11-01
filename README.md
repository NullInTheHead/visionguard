# VisionGuard

[![ci](https://github.com/NullInTheHead/visionguard/actions/workflows/ci.yml/badge.svg)](https://github.com/NullInTheHead/visionguard/actions/workflows/ci.yml)

**Industrial anomaly detection under distribution shift.** Under which kinds of shift does a
pretrained-feature detector keep working, and how few reference images does it need to
recover on a product it has never seen?

> Status: Phase 2 (baseline reproduction); protocol frozen at v1.0. Demo, results and
> findings will appear here as the milestones land: M1 baseline (v0.1) → M2 research
> result (v0.2) → M3 shipped (v1.0).

## Research question

Visual inspection systems are usually fitted to one product under one camera setup. In a
factory, products change, lighting drifts and new lines are added. VisionGuard measures a
PatchCore-style detector under three separate kinds of shift:

| Experiment | Shift | Headline output |
|---|---|---|
| E1 | New defects on a familiar product | Per-category image AUROC, pixel AUROC, AU-PRO |
| E2 | Entirely new product (zero-shot and k-shot) | AUROC vs number of reference images |
| E3 | Image acquisition (blur, noise, lighting, JPEG) | AUROC vs corruption severity |

The full evaluation protocol, including the category split and leakage rules, is in
[`docs/protocol.md`](docs/protocol.md).

## Setup

Needs Python 3.10+. Use a virtual environment (on macOS there is no bare `pip` command):

```bash
git clone https://github.com/NullInTheHead/visionguard.git
cd visionguard
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # data, metrics, tests (no torch needed)
pip install -e ".[models,dev]"     # + torch, timm, faiss for the experiments
pytest                             # runs on a synthetic dataset, no download needed
```

### Get MVTec AD (about 4.9 GB)

MVTec only serves the dataset through its website, so download it in a browser from the
[MVTec AD downloads page](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)
(`mvtec_anomaly_detection.tar.xz`), then extract it:

```bash
bash scripts/setup_mvtec.sh ~/Downloads/mvtec_anomaly_detection.tar.xz data
# -> data/mvtec_anomaly_detection/<category>/{train,test,ground_truth}
```

**Kaggle (recommended GPU).** Upload the extracted `mvtec_anomaly_detection` folder once as
a private Kaggle dataset, attach it to a GPU notebook, then:

```python
!git clone https://github.com/NullInTheHead/visionguard.git
%cd visionguard
!pip install -q -e ".[models,dev]"
# pass --root /kaggle/input/<your-dataset>/mvtec_anomaly_detection to the scripts
```

**Colab.** Keep the `.tar.xz` in Google Drive and extract it to local disk at the start of
each session with `scripts/setup_mvtec.sh` (reading thousands of small files straight from
Drive is slow).

## Phase 1: inspect the data

```bash
python scripts/inspect_data.py --root data/mvtec_anomaly_detection --out results/data
```

Writes per-category and per-defect-type tables (counts, defect sizes, tiny defects, defects
lost to the centre crop, a check against the published counts) and sample grids with masks
overlaid to `results/data/`.

## Phase 2: PatchCore baseline (E1)

Our own PatchCore (`src/visionguard/patchcore.py`) on WideResNet-50, run on the 10
development categories:

```bash
pip install -e ".[models,dev]"
python scripts/run_e1.py                              # ~1-3 min per category on a GPU or Apple M-series
```

Results land in `results/e1/`: `metrics.csv` (one row per category, with the git commit
and package versions), per-image scores in `scores/`, and heatmap grids in `heatmaps/`
(highest-scoring defects, highest-scoring good parts, lowest-scoring defects).

**Check against anomalib.** Run anomalib's PatchCore with the same settings on two
categories, in its own environment (anomalib pins its own torch and lightning versions):

```bash
python -m venv .venv-anomalib && source .venv-anomalib/bin/activate
pip install "anomalib>=2.0,<3" pandas
python scripts/run_anomalib_reference.py --categories bottle capsule
deactivate && source .venv/bin/activate
python scripts/run_e1.py --categories bottle capsule --calibration-fraction 0
```

`--calibration-fraction 0` puts every training image in the memory bank, as anomalib
does. Image AUROC should agree within about 1-2 points. Held-out categories are refused
unless `--allow-held-out` is passed, which is only for runs with frozen settings.

## Repository layout

```
configs/            base.yaml, split.yaml (dev / held-out categories), one YAML per experiment
docs/protocol.md    evaluation protocol, written before any experiment
src/visionguard/
  data.py           MVTec index, preprocessing, calibration split, nested k-shot sampler
  data_report.py    dataset inventory and sample figures
  metrics.py        image/pixel AUROC, AU-PRO, FPR at 95% recall, fixed-threshold metrics
  features.py       frozen backbones (WRN-50, supervised / DINOv2 / CLIP ViT-B) behind one interface
  patchcore.py      patch embedding, greedy coreset, NN scoring, reweighting, smoothed maps
  experiment.py     metrics rows, CSV logging with run metadata, heatmap grids
  corruptions.py    acquisition-shift transforms (Phase 3)
scripts/            setup_mvtec.sh, inspect_data.py, run_e1.py, run_anomalib_reference.py, (run_e2.py, run_e3.py)
results/            CSVs and figures (committed)
tests/              unit tests on a synthetic MVTec tree
app/                demo (Phase 5)
```

## Data licence and attribution

MVTec AD is © MVTec Software GmbH, licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use).
The dataset is not included in this repository. If you use it, cite:

* P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. *MVTec AD — A Comprehensive Real-World
  Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
* P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger. *The MVTec Anomaly
  Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly
  Detection.* IJCV 2021.

The code in this repository is MIT-licensed (see `LICENSE`).
