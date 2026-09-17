# EEG Focus vs. Distraction

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.12-007ACC)](https://mne.tools/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

Classifying **focus vs. distraction** from EEG while phone-style notifications interrupt
a timed word puzzle. 23 participants, 12 models, subject-independent evaluation.

**Best result: EEGNet — accuracy 0.728, ROC-AUC 0.798, subject-independent 5-fold.**

---

## Dataset

| | |
|---|---|
| Participants | 23 analysed (24 recorded, 1 excluded) |
| Trials | 4,600 → **4,544** after cleaning (98.8%) |
| Classes | 2,273 focus / 2,271 distractor |
| Channels | 15 EEG |
| Sampling rate | 500 Hz → 250 Hz (epochs) → 125 Hz (model input) |
| Model input | `(4544, 15, 375)` — 3.0 s window, 2.0–5.0 s |
| Evaluation | subject-independent 5-fold — test subjects never seen in training |

---

## Task

Participants solved a scrambled-word puzzle while a notification interrupted them.
**Focus trials had 1 beep, distractor trials had 3.**

```
t = 0.0 s   fixation "+"                       ← EEG marker is HERE
t = 2.0 s   puzzle + notification + BEEP 1     answer window opens
t = 2.5 s   BEEP 2   ← distractor only
t = 4.0 s   BEEP 3   ← distractor only
t = 5.0 s   trial ends
```

The marker sits at the fixation cross, not the puzzle. Analysis uses **2.0–5.0 s**;
**0.0–2.0 s is a control window** where the two conditions are physically identical.

---

## Results

Subject-independent 5-fold. Control column is the 0.0–2.0 s window, where every model
must land at ~0.50.

| Model | Accuracy | **ROC-AUC** | Cohen's κ | Control AUC | Params |
|---|---|---|---|---|---|
| **LMDA-Net** | 0.723 ± 0.010 | **0.799 ± 0.015** | 0.446 | 0.501 | 3,772 |
| **EEGNet** | **0.728 ± 0.027** | **0.798 ± 0.032** | 0.456 | 0.490 | 1,698 |
| CTNet | 0.706 ± 0.027 | 0.780 ± 0.028 | 0.412 | 0.512 | 21,330 |
| LDA-shrinkage | 0.704 ± 0.028 | 0.775 ± 0.033 | 0.408 | 0.486 | — |
| EEG Conformer | 0.692 ± 0.039 | 0.771 ± 0.033 | 0.385 | 0.504 | 572,906 |
| Xdawn+TS | 0.653 ± 0.026 | 0.711 ± 0.031 | 0.306 | 0.516 | — |
| Conformer-LSTM | 0.642 ± 0.034 | 0.703 ± 0.043 | 0.284 | 0.504 | 17,770 |
| Conformer-Transformer | 0.619 ± 0.062 | 0.663 ± 0.080 | 0.237 | 0.517 | 94,434 |
| LR-bandpower | 0.576 ± 0.017 | 0.610 ± 0.023 | 0.153 | 0.493 | — |
| Riemann-TS | 0.553 ± 0.012 | 0.572 ± 0.018 | 0.107 | 0.493 | — |
| EEG Transformer | 0.518 ± 0.009 | 0.521 ± 0.017 | 0.036 | 0.489 | 175,490 |
| CSP+LDA | 0.504 ± 0.010 | 0.504 ± 0.011 | 0.009 | 0.504 | — |
| *SHUFFLED control* | *0.486 ± 0.026* | *0.477 ± 0.036* | *−0.028* | *0.508* | — |

Per-subject: **23 / 23 above chance** with EEGNet.

---

## The models

**Deep networks** — all trained with one shared loop (`scripts/dl_models.py`): Adam 1e-3,
weight decay 1e-4, batch 64, class-weighted cross-entropy, 300 epochs, early stopping
with patience 40 on validation loss.

| Model | Idea |
|---|---|
| **EEGNet** (Lawhern 2018) | Compact separable CNN. 512 ms temporal kernel, then a spatial filter per feature map. |
| **LMDA-Net** (Miao 2023) | CNN with attention across feature maps; learns its own channel weighting. |
| **CTNet** (Zhao 2024) | EEGNet-style patch embedding → 6 transformer layers over time tokens. |
| **EEG Conformer** (Song 2023) | Wide conv front end with overlapping patches → transformer. |
| **EEG Transformer** | Attention over *raw* temporal patches, no convolution. Scores at chance — the control that shows convolution is doing the work. |
| **Conformer-Transformer** | Conv tokenizer → Conformer blocks (conv *inside* attention) → transformer. |
| **Conformer-LSTM** | Same tokenizer and Conformer block → BiLSTM instead of a transformer. No positional encoding needed. |
| **Context Transformer** | Attends *across* trials from one person rather than within a trial. AUC 0.781 — no better than 0.776 at context size 1, so cross-trial context does not help here. |

**Classical baselines** — LDA-shrinkage, Xdawn+TS, LR-bandpower, Riemann-TS, CSP+LDA.
CSP fails because it looks for oscillatory power while the signal is a time-locked
evoked response.

Two findings worth noting: **time-domain models beat frequency-domain ones** (0.798 vs
0.610), and **capacity hurts** — 1,698 parameters beat 572,906 on 2,789 training trials.

---

## ⚠ What the model actually detects

Re-running every model on the **0.0–2.0 s control window** gives AUC **0.477–0.517**
across all 13 — dead on chance. Nothing leaks, which is what makes the 0.798 believable.
But it also shows what is being detected: decoding rises only *after* each extra beep.

> This is a working detector for **"did an interruption occur, and did the brain
> respond?"** at AUC 0.798 across unseen people. It is **not** a detector of whether
> someone is concentrating.

---

## Pipeline

| Stage | Script | Runtime |
|---|---|---|
| 1 · Build trial table | `01_build_trials.py` | ~5 s |
| 2 · Preprocess EEG | `02_preprocess.py` | ~10 min |
| 3 · Extract features | `03_features.py` | ~3 min |
| 4 · Train & evaluate | `04_train.py` | ~24 min |
| 5 · Figures | `05_report.py` | ~2 min |
| 7 · Band sweep | `07_bands.py` | hours (CPU) |
| 9 · EEG Transformer | `09_transformer.py` | ~10 min (GPU) |
| 10 · Context Transformer | `10_context_transformer.py` | ~22 min (GPU) |
| 11 · Conformer-Transformer | `11_conformer_transformer.py` | ~16 min (GPU) |
| 12 · Conformer-LSTM | `12_conformer_lstm.py` | ~11 min (GPU) |

**Preprocessing:** drop non-brain channels (19 → 15) · `standard_1020` montage ·
50/100 Hz notch · 1–45 Hz band-pass · average reference · ICA (blinks via Fp1/Fp2,
max 3 components) · epoch −0.5 to +5.0 s at 250 Hz, no baseline correction ·
reject trials with peak-to-peak > 200 µV · per-subject, per-channel z-score.

**Evaluation:** splits are by person, never by trial. One split function
(`folds_for()` in `07_bands.py`) is imported by every model script, so no two models
can be scored on different folds.

---

## Getting started

```bash
pip install mne numpy pandas scipy scikit-learn pyriemann torch matplotlib
```

```bash
python scripts/01_build_trials.py
python scripts/02_preprocess.py
python scripts/03_features.py
python scripts/04_train.py         # --quick for a smoke test
python scripts/05_report.py
```

Each deep-learning script also supports `--inspect` (shapes and parameters, trains
nothing), `--quick` (smoke test) and `--report` (rebuild the HTML report only).

Verified with python 3.13.6, mne 1.12.1, numpy 2.3.5, scikit-learn 1.8.0,
pyriemann 0.12, torch 2.11.0+cu128 (CUDA 12.8).

---

## Structure

```
├── scripts/          01-12, the full pipeline + model definitions
├── derivatives/      results CSVs, per-job JSONs, HTML reports
└── figures/          13 PNGs
```

**Not in this repo** (see `.gitignore`): the raw BrainVision recordings
(`Notification/`, ~1.1 GB) and the regenerable binary intermediates
(`derivatives/epochs/`, `bands/`, `features.npz`, `models/`, ~1.1 GB). Run stages 1–3
to rebuild them, or `python scripts/07_bands.py --prepare-only` for the deep-learning
inputs alone.

---

## Notes

**Corrections.** Two participants answered with left/right keys swapped and were
corrected via `key_corrected`. One participant was excluded entirely — wrong experiment
profile and a marker/trial mismatch. Group accuracy rose 78.4% → **84.5%**.

**Verification.** Every stage was re-checked by recomputing from source rather than
trusting the pipeline's own report: **0 failures** across trial counts, EEG↔CSV
alignment, preprocessing parameters and feature values.

---

## Acknowledgements

Built with [MNE-Python](https://mne.tools/), [PyTorch](https://pytorch.org/),
[scikit-learn](https://scikit-learn.org/) and [pyriemann](https://pyriemann.readthedocs.io/).
EEGNet follows Lawhern et al. (2018); CTNet, LMDA-Net and EEG Conformer follow
Zhao et al. (2024), Miao et al. (2023) and Song et al. (2023).
