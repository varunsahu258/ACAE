# ACAE — Adversarial Collaborative Auto-encoder for Top-N Recommendation

> **Replication of:** Yuan, F., Yao, L., & Benatallah, B. (2018).
> *Adversarial Collaborative Auto-encoder for Top-N Recommendation.*
> arXiv:[1808.05361](https://arxiv.org/abs/1808.05361)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Hardware](https://img.shields.io/badge/Hardware-CUDA%20%7C%20Metal%20%7C%20CPU-lightgrey)

---

## Overview

This repository provides a full replication of the ACAE paper, including:

- All 7 models: **ItemPop, MF-BPR, CDAE, NeuMF, AMF, CAE, ACAE**
- Full **Table 3** results (HR@5/10, NDCG@5/10 on MovieLens-1M, CiaoDVD, FilmTrust)
- **Figure 3** training traces and **Figure 4** robustness curves
- Hardware-agnostic device config — runs on any machine

---

## Hardware Support

The code automatically detects and uses the best available hardware:

| Hardware | Backend | Notes |
|---|---|---|
| NVIDIA RTX 4500 Ada (or any CUDA GPU) | CUDA | TF32 + XLA JIT enabled automatically |
| Apple M1/M2/M3 | Metal | `tensorflow-metal` required |
| AMD / other GPU | ROCm / generic | Detected automatically if TF supports it |
| No GPU | CPU | Always works, slower |

No code changes needed — `device_config.py` handles everything.

---

## Directory Structure

```
acae/
├── device_config.py       # Auto-detects GPU/CPU, sets batch sizes
├── model.py               # CAE and ACAE (TensorFlow 2)
├── baselines.py           # ItemPop, MF-BPR, CDAE, NeuMF, AMF
├── trainer.py             # Training routines (Algorithm 1 for ACAE)
├── metrics.py             # HR@N and NDCG@N
├── data_utils.py          # Data loading, leave-one-out split, rating matrix
├── run_experiments.py     # Main script – reproduces Table 3
├── robustness_analysis.py # Reproduces Figure 4 and Table 2
├── plot_figures.py        # Plots Figure 3 and Figure 4 from saved JSONs
├── dry_run.py             # Smoke-test – verifies all scripts work (~60s)
├── setup_cuda_windows.md  # Step-by-step CUDA setup for Windows + NVIDIA
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/acae-replication.git
cd acae-replication
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install tensorflow>=2.12 numpy pandas scipy matplotlib tqdm
```

**Apple Silicon (M1/M2/M3) — add Metal GPU support:**
```bash
pip install tensorflow-macos tensorflow-metal
```

**Windows + NVIDIA GPU — full CUDA setup:**
See [`setup_cuda_windows.md`](setup_cuda_windows.md) for the step-by-step guide.

---

## Dataset Setup

### MovieLens-1M
Downloaded **automatically** on first run. No action needed.

### FilmTrust
1. Download from https://www.librec.net/datasets.html
2. Extract and place `ratings.txt` in `data/filmtrust/`

### CiaoDVD
1. Download from https://www.librec.net/datasets.html
2. Extract and place `movie-ratings.txt` in `data/ciao/`

---

## Quick Start — Dry Run

Before running full experiments, verify that all scripts are working correctly with the dry run. It exercises every code path (data loading, all 6 models, training, evaluation, robustness, plotting) using a tiny data slice and finishes in **under 60 seconds** on any hardware.

```bash
python dry_run.py
```

Example output:
```
──────────────────────────────────────────────────────────
  1 · Data pipeline
──────────────────────────────────────────────────────────
[ PASS ] MovieLens-1M download + slice              4.2s
[ PASS ] leave_one_out_split + rating matrix        0.1s

──────────────────────────────────────────────────────────
  2 · GPU / device setup
──────────────────────────────────────────────────────────
[ PASS ] Pin rating matrix → /GPU:0                 0.1s

──────────────────────────────────────────────────────────
  3 · Model construction
──────────────────────────────────────────────────────────
[ PASS ] ItemPop()                                  0.0s
[ PASS ] MFBPR()                                    0.1s
...

════════════════════════════════════════════════════════
  DRY RUN COMPLETE
════════════════════════════════════════════════════════
  Checks :  17
  Passed :  17
  Failed :  0
  Total  :  38.4s

  All checks passed. Project is ready to run.
```

If any check shows `FAIL`, the error and traceback are printed inline so you can fix it before committing to a multi-hour training run.

---

## Running Experiments

### Single dataset
```bash
python run_experiments.py --dataset movielens
python run_experiments.py --dataset filmtrust
python run_experiments.py --dataset ciao
```

### All datasets (reproduces full Table 3)
```bash
python run_experiments.py --all
```

Results are saved as JSON files in `results/`.

### Robustness analysis (Figure 4 / Table 2)
```bash
python robustness_analysis.py --dataset movielens
python robustness_analysis.py --dataset filmtrust
```

### Plotting (Figure 3 + Figure 4)
Run after experiments:
```bash
python plot_figures.py
```
Figures are saved as PDFs in `figures/`.

---

## Expected Results (Table 3)

| Method | ML-HR@5 | ML-HR@10 | ML-NDCG@5 | ML-NDCG@10 | Ciao-HR@5 | FT-HR@5 |
|--------|---------|----------|-----------|------------|-----------|---------|
| ItemPop | 0.3101 | 0.4458 | 0.2127 | 0.2562 | 0.2425 | 0.6562 |
| MF-BPR | 0.5682 | 0.7163 | 0.4148 | 0.4628 | 0.3454 | 0.8103 |
| CDAE | 0.5746 | 0.7050 | 0.4287 | 0.4709 | 0.3432 | 0.8169 |
| NeuMF | 0.5832 | 0.7250 | 0.4304 | 0.4727 | 0.3702 | 0.8305 |
| AMF | 0.5875 | 0.7264 | 0.4314 | 0.4763 | 0.3798 | 0.8397 |
| **ACAE** | **0.5988** | **0.7379** | **0.4446** | **0.4905** | **0.3814** | **0.8434** |

> **Note:** Small differences (±0.5–1%) from the paper are expected due to random seeds and TF version (paper used TF 1.x). Relative rankings are preserved.

---

## Key Design Choices

| Component | Setting |
|---|---|
| Encoder activation | sigmoid σ(x) |
| Decoder activation | identity f(x) = x |
| Loss function | Binary cross-entropy |
| Adversarial noise | Fast-gradient method (Eq. 4) on W1 and W2 |
| Pre-training optimiser | Adam with fixed LR |
| Adversarial training optimiser | Adagrad (adaptive LR) |
| Evaluation | Leave-one-out, 200 negatives, HR@5/10, NDCG@5/10 |

---

## Hyper-parameters

| Dataset | latent_dim | γ (L2) | ε (noise) | λ1 | λ2 | LR |
|---|---|---|---|---|---|---|
| MovieLens | 50 | 0.01 | 1.0 | 1.0 | 10.0 | 0.001 |
| FilmTrust | 50 | 0.001 | 2.0 | 1.0 | 10.0 | 0.001 |
| CiaoDVD | 50 | 0.001 | 0.5 | 1.0 | 1.0 | 0.001 |

---

## Expected Runtime

| Hardware | MovieLens | FilmTrust | CiaoDVD | Total |
|---|---|---|---|---|
| RTX 4500 Ada (24 GB) | ~25 min | ~8 min | ~12 min | ~45 min |
| Apple M2 (16 GB) | ~60 min | ~20 min | ~30 min | ~110 min |
| CPU only | ~4–8 hrs | ~1–2 hrs | ~2–3 hrs | varies |

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{yuan2018adversarial,
  title   = {Adversarial Collaborative Auto-encoder for Top-N Recommendation},
  author  = {Yuan, Feng and Yao, Lina and Benatallah, Boualem},
  journal = {arXiv preprint arXiv:1808.05361},
  year    = {2018}
}
```

---

## License

MIT
