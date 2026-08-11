# EEG Focus vs. Distraction



[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.12-007ACC)](https://mne.tools/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

---

## Overview

23 participants solved timed word puzzles while phone-style notifications interrupted them.
Each trial carried a notification image plus **1 beep** (focus) or **3 beeps** (distractor).
This repo contains the full pipeline from raw BrainVision recordings to a
subject-independent classifier.

**EEGNet, subject-independent 5-fold: accuracy 0.724 ± 0.028, ROC-AUC 0.794 ± 0.026.**

| | |
|---|---|
| Participants | 23 analysed (24 recorded, 1 excluded) |
| Trials | 4,600 → **4,544** after cleaning (98.8%) |
| Class balance | 2,273 focus / 2,271 distractor |
| Channels | 15 EEG, 500 Hz → 250 Hz |
| Evaluation | Subject-independent 5-fold — test subjects never seen in training |

---

## Results

| Model | Accuracy | **ROC-AUC** | Cohen's κ |
|---|---|---|---|
| **EEGNet** | **0.724 ± 0.028** | **0.794 ± 0.026** | 0.448 |
| LDA-shrinkage | 0.704 ± 0.028 | 0.775 ± 0.033 | 0.408 |
| Xdawn+TS | 0.653 ± 0.026 | 0.711 ± 0.031 | 0.306 |
| LR-bandpower | 0.576 ± 0.017 | 0.610 ± 0.023 | 0.153 |
| Riemann-TS | 0.553 ± 0.012 | 0.572 ± 0.018 | 0.107 |
| CSP+LDA | 0.504 ± 0.010 | 0.504 ± 0.011 | 0.009 |
| *SHUFFLED control* | *0.486 ± 0.026* | *0.477 ± 0.036* | *−0.028* |

Per-subject: **23 / 23 above chance.** CSP fails because it looks for oscillatory power
while the signal is a time-locked evoked response — the right tool for the wrong physics.

---

## ⚠ What the model actually detects

The models classify the siganl into focus and distraction.

Re-running every model on the **0.0–2.0 s control window**, where the conditions are
physically identical, gives AUC **0.486–0.516 for all seven** — EEGNet scores 0.499, dead on
chance. Nothing leaks, which is what makes the 0.794 believable.

Four independent lines of evidence:

1. All 6 models sit at chance on the control window.
2. EEGNet's validation loss bottoms out at epoch 1–2 there (mean best epoch **4** vs. **92**
   on the main window) — no metric involved.
3. Decoding rises only *after* each extra beep (0.648 at +2.75 s, 0.603 at +4.25 s) and is
   flat before the first one.
4. Time-domain models beat frequency-domain ones (EEGNet 0.794 vs. band-power LR 0.610) —
   the signature of an evoked response, not an oscillatory state change.

> **Honest framing:** this is a working detector for *"did an interruption occur, and did the
> brain respond?"* at AUC 0.794 across unseen people. It is **not** a detector of whether
> someone is concentrating.

*Decoding over time — flat at chance while conditions are identical, then two sharp peaks
~250 ms after each extra beep. The clearest evidence of the confound.*

---

## Experiment design

Participants saw a scrambled word and a claim such as *"(count C + N) = 2"*, pressing
**RIGHT** for true or **LEFT** for false within 3 seconds. Times from the `trial_start` marker:

```
t = 0.0 s   fixation "+"                        ← the EEG marker is HERE
t = 2.0 s   puzzle + notification + BEEP 1      answer window opens
t = 2.5 s   BEEP 2   ← distractor trials only
t = 4.0 s   BEEP 3   ← distractor trials only
t = 5.0 s   trial ends
```

> **Careful:** the marker sits at the **fixation cross**, not the puzzle, which appears
> 2 seconds later. Getting this wrong shifts every analysis by 2 seconds.

The manipulation worked behaviourally — accuracy 0.806 (focus) vs. 0.754 (distractor),
t(23) = 4.30, **p = 0.0003**.

---

## Pipeline

| Stage | Script | Runtime |
|---|---|---|
| 1 · Build trial table | `scripts/01_build_trials.py` | ~5 s |
| 2 · Preprocess EEG | `scripts/02_preprocess.py` | ~10 min |
| 3 · Extract features | `scripts/03_features.py` | ~3 min |
| 4 · Train & evaluate | `scripts/04_train.py` | ~24 min |
| 5 · Figures | `scripts/05_report.py` | ~2 min |
| 6 · Colab bundle | `scripts/06_make_colab_bundle.py` | ~2 min |

**Preprocessing:** drop non-brain channels (19 → 15) · `standard_1020` montage · 50/100 Hz
notch · 1–45 Hz FIR band-pass · average reference · ICA (extended Infomax, blinks via Fp1/Fp2
as EOG substitutes, hard cap 3 components) · epoch −0.5 to +5.0 s at 250 Hz, no baseline
correction · reject trials with peak-to-peak > 200 µV.

**Features:** log absolute and relative band power (5 bands × 15 channels × 4 windows),
OAS-shrinkage covariances, ERP bins, and a 21-step sliding-window sweep. Two rules are
enforced in code: only *unsupervised* features are precomputed (CSP and Xdawn are fitted
**inside** each CV fold, since they use labels to build their filters), and no normalisation
is baked into the saved file.

**Evaluation:** splits are by person, never by trial — 5 outer folds (4–5 test subjects
each) with 4 more subjects held out for EEGNet early stopping.

---

## Structure

```
├── Notification/     raw BrainVision recordings + PsychoPy behavioural files
├── scripts/          01–06, the full pipeline
├── derivatives/      trials_master.csv, epochs/, ica/, features.npz, results, models/
└── figures/          7 PNGs
```

---

## Getting started

```bash
pip install mne numpy pandas scipy scikit-learn pyriemann torch matplotlib
```

```bash
python scripts/01_build_trials.py    # ~5 s
python scripts/02_preprocess.py      # ~10 min
python scripts/03_features.py        # ~3 min
python scripts/04_train.py           # ~24 min   (--quick for a smoke test)
python scripts/05_report.py          # ~2 min
```

Verified with python 3.13.6, mne 1.12.1, numpy 2.3.5, scikit-learn 1.8.0, pyriemann 0.12,
torch 2.10.0.

```python
import numpy as np
d = np.load("derivatives/features.npz", allow_pickle=True)
X      = d["bp_rel_post"].reshape(len(d["label"]), -1)   # (4544, 75)
y      = d["label"]                                       # 0 = focus, 1 = distractor
groups = d["subject"]                                     # keeps a person out of both splits
```



---

**Corrections applied.** `ganesh` and `siddhartha` answered with left/right keys swapped
(10.4% and 16.4% accuracy → 89.6% and 83.6% when flipped); corrected via `key_corrected`,
with `condition` untouched. `pranshu` was excluded entirely — wrong experiment profile, 210
EEG markers against 200 CSV trials, and an 11.76 s inter-trial gap. Group accuracy rose from
78.4% → **84.5%**.

**Verification.** Every stage was independently re-checked by recomputing from source rather
than trusting the pipeline's own report: **0 failures** across trial counts, EEG↔CSV
alignment, preprocessing parameters, and feature values (`bp_abs`/`bp_rel` recomputed from
scratch, max difference 1.1e-07).

---

## Acknowledgements

Built with [MNE-Python](https://mne.tools/), [PyTorch](https://pytorch.org/),
[scikit-learn](https://scikit-learn.org/) and [pyriemann](https://pyriemann.readthedocs.io/).
EEGNet follows Lawhern et al. (2018).
