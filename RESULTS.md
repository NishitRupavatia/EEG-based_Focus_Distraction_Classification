# Step 4 — Model Results

**Scripts:** `scripts/04_train.py` (training), `scripts/05_report.py` (figures)
**Evaluation:** subject-independent, 5-fold, 23 subjects, 4,544 trials
**Runtime:** 24 minutes · **Model failures: 0**
**Last updated:** 2026-08-10

---

## 1. Headline

**EEGNet wins: accuracy 0.724 ± 0.028, ROC-AUC 0.794 ± 0.026** on people it has never seen.

Every guardrail passed. But the result means something narrower than "we can detect
distraction" — see §8.

---

## 2. How it was evaluated

**Subject-independent** — split by person, never by trial. The test subjects' data never
appears in training in any form, so the model cannot score well by recognising individual
brains.

```
23 subjects
 └─ OUTER: subject-wise 5-fold          test = 4–5 subjects per fold
     └─ INNER: 4 more subjects held out as VALIDATION
          TRAIN      = 14–15 subjects   ~2,750–2,790 trials
          VALIDATION = 4 subjects       ~760–800 trials   (EEGNet early stopping only)
          TEST       = 4–5 subjects     ~995 trials
```

All 6 models trained on the **same** train set, so the comparison is fair. Only EEGNet
used the validation set. CSP and Xdawn were re-fitted **inside every fold** because they
use the labels to build their filters.

---

## 3. Main results — window 2.0–5.0 s

| Model | Accuracy | Balanced acc. | **ROC-AUC** | F1 (macro) | Cohen's κ |
|---|---|---|---|---|---|
| **EEGNet** | **0.724 ± 0.028** | 0.724 ± 0.028 | **0.794 ± 0.026** | 0.724 | 0.448 |
| LDA-shrinkage | 0.704 ± 0.028 | 0.704 ± 0.028 | 0.775 ± 0.033 | 0.704 | 0.408 |
| Xdawn+TS | 0.653 ± 0.026 | 0.653 ± 0.025 | 0.711 ± 0.031 | 0.652 | 0.306 |
| LR-bandpower | 0.576 ± 0.017 | 0.576 ± 0.017 | 0.610 ± 0.023 | 0.576 | 0.153 |
| Riemann-TS | 0.553 ± 0.012 | 0.553 ± 0.012 | 0.572 ± 0.018 | 0.553 | 0.107 |
| CSP+LDA | 0.504 ± 0.010 | 0.505 ± 0.010 | 0.504 ± 0.011 | 0.475 | 0.009 |
| *SHUFFLED-control* | *0.486 ± 0.026* | *0.486* | *0.477 ± 0.036* | *0.486* | *−0.028* |

Sensitivity and specificity are balanced for every real model (EEGNet 0.721 / 0.727), so
neither class is being favoured.

**Why CSP failed (0.504).** CSP looks for differences in *oscillatory power* between
conditions. Our signal is a *time-locked evoked response*. CSP is the right tool for the
wrong physics — its failure is informative, not a bug.

> **⚠ Xdawn+TS is numerically unstable — read its 0.711 as 0.71 ± 0.02.**
> Discovered while packaging the Colab bundle. Adding noise of **±0.001 µV** — a thousand
> times below any real signal — moves Xdawn's AUC by up to **0.016**, while Riemann-TS
> moves by 0.0001 under the same perturbation. Cause: Xdawn does a generalised
> eigendecomposition, and our covariance is only **rank ~11 of 15** (average referencing
> removes one degree of freedom, ICA component removal takes ~3 more). Near-degenerate
> eigenvalues make the eigenvector directions swing under infinitesimal input changes.
> Do not quote Xdawn to three decimals. The model **ranking is unaffected** — it sits far
> below EEGNet (0.794) and LDA (0.776) either way. Every other model is stable.

---

## 4. 🛡 The control window — the result that matters most

Re-run of every model on **0.0–2.0 s**, where the two conditions are *physically
identical* (no second beep yet). Everything must land at chance:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Xdawn+TS | 0.509 ± 0.016 | 0.516 ± 0.017 |
| SHUFFLED-control | 0.510 ± 0.016 | 0.508 ± 0.026 |
| CSP+LDA | 0.498 ± 0.016 | 0.504 ± 0.015 |
| EEGNet | 0.502 ± 0.015 | **0.499 ± 0.020** |
| Riemann-TS | 0.489 ± 0.013 | 0.493 ± 0.011 |
| LR-bandpower | 0.498 ± 0.021 | 0.493 ± 0.021 |
| LDA-shrinkage | 0.489 ± 0.010 | 0.486 ± 0.018 |

**All seven sit between 0.486 and 0.516.** Nothing leaks. EEGNet — the model most capable
of finding a shortcut — scores 0.499, dead on chance.

This is what makes the 0.794 believable. A pipeline with a subject-identity leak or a
label-alignment bug would score above chance *here too*.

### A second, independent confirmation

EEGNet's early stopping tells the same story without any metric involved:

| | Main window | Control window |
|---|---|---|
| Best epoch per fold | 82, 24, 127, 100, 125 | **2, 2, 16, 1, 1** |
| Mean best epoch | **92** | **4** |

On the main window the network kept improving for ~90 epochs. On the control window
validation loss bottomed out at **epoch 1 or 2** and never improved again — the network
found nothing to learn, then overfit for 40 more epochs until early stopping killed it.

---

## 5. Confusion matrices — main window, all 5 folds pooled (n = 4,544)

| Model | | pred focus | pred distr | correct |
|---|---|---|---|---|
| **EEGNet** | true focus | **1651** | 622 | 72.6% |
| | true distr | 629 | **1642** | 72.3% |
| **LDA-shrinkage** | true focus | 1612 | 661 | 70.9% |
| | true distr | 679 | 1592 | 70.1% |
| **Xdawn+TS** | true focus | 1453 | 820 | 63.9% |
| | true distr | 759 | 1512 | 66.6% |
| **LR-bandpower** | true focus | 1314 | 959 | 57.8% |
| | true distr | 969 | 1302 | 57.3% |
| **Riemann-TS** | true focus | 1262 | 1011 | 55.5% |
| | true distr | 1015 | 1256 | 55.3% |
| **CSP+LDA** | true focus | 843 | 1430 | 37.1% |
| | true distr | 819 | 1452 | 63.9% |
| *SHUFFLED* | true focus | 1107 | 1166 | 48.7% |
| | true distr | 1169 | 1102 | 48.5% |

EEGNet's errors are almost perfectly symmetric (622 vs 629) — no class bias. CSP is the
only model with a lopsided matrix, which is what an uninformative model looks like when
its threshold drifts.

---

## 6. EEGNet training

| Setting | Value |
|---|---|
| Architecture | EEGNet-8,2 — F1=8, D=2, F2=16, ~2,000 parameters |
| Input | 15 channels × 375 samples (2.0–5.0 s at 125 Hz) |
| Dropout | 0.5 (cross-subject setting) |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4 |
| Batch size | 64 |
| Loss | class-weighted cross-entropy |
| **Max epochs allowed** | **300** |
| **Early stopping** | **patience 40** on validation loss, best weights restored |

**Epochs actually run (main window):**

| Fold | Best epoch | Stopped at |
|---|---|---|
| 0 | 82 | 122 |
| 1 | 24 | 64 |
| 2 | 127 | 167 |
| 3 | 100 | 140 |
| 4 | 125 | 165 |
| **mean** | **92** | **132** |

It never came close to the 300 ceiling. The loss curves (`fig3`) show the classic shape:
training loss keeps falling while validation loss flattens near 0.575 and turns upward —
the point where the network stops learning the task and starts memorising the training
subjects. Early stopping catches exactly that moment.

---

## 7. Consistency

**Per fold (AUC, main window)** — EEGNet is stable across every split:

| Model | f0 | f1 | f2 | f3 | f4 | mean |
|---|---|---|---|---|---|---|
| EEGNet | 0.767 | 0.832 | 0.794 | 0.804 | 0.773 | **0.794** |
| LDA-shrinkage | 0.745 | 0.818 | 0.773 | 0.799 | 0.743 | 0.776 |
| Xdawn+TS | 0.676 | 0.754 | 0.697 | 0.728 | 0.697 | 0.710 |
| LR-bandpower | 0.592 | 0.595 | 0.613 | 0.648 | 0.602 | 0.610 |
| Riemann-TS | 0.573 | 0.582 | 0.581 | 0.582 | 0.540 | 0.572 |
| CSP+LDA | 0.506 | 0.517 | 0.511 | 0.493 | 0.491 | 0.504 |

**Per subject (EEGNet, each scored while held out): 23 / 23 above chance.**

- Best: **david 0.921**, Ashish 0.917, digvijay 0.894
- Worst: **naveen 0.587**, vipul 0.688, pragnesh 0.688

Note that pragnesh — the one subject with only 80.5% epoch retention — still reaches
0.688, and the two behaviourally-invalid subjects do fine (rishi 0.805, mahendra 0.758).
That is expected: the focus/distractor label is valid regardless of whether someone
answered the puzzles well.

---

## 8. What the model is actually detecting

**Not attention. Sound.**

The time-resolved analysis (`fig7`, 500 ms sliding window, same 5-fold split):

| Time from trial start | AUC | |
|---|---|---|
| −0.25 → +2.50 s | 0.47 – 0.52 | flat at chance — **conditions physically identical** |
| **+2.75 s** | **0.648** | ← beep 2 fires at 2.5 s |
| +3.00 s | 0.606 | |
| +3.25 → +4.00 s | 0.53 | decays back toward chance |
| **+4.25 s** | **0.603** | ← beep 3 fires at 4.0 s |
| +4.50 s | 0.568 | |

Two sharp peaks, each roughly 250 ms after an extra beep, and **nothing at all** before
the first one. That is the signature of an auditory evoked response, not a sustained
mental state.

Four independent lines of evidence now agree:

1. All 6 models sit at chance on the physically-identical control window (§4).
2. EEGNet's validation loss bottoms out at epoch 1–2 there (§4).
3. Decoding rises only after each extra beep, then decays (§8).
4. Time-domain models beat frequency-domain ones — LDA on ERP bins 0.775 and EEGNet
   0.794, versus band-power LR 0.610 — which is what you see for a time-locked evoked
   response, not an oscillatory state change (`FEATURE.md` §7).

**Honest framing:** this is a working detector for *"did an interruption event occur, and
did the brain respond to it?"* — at AUC 0.794 across unseen people, which is a genuinely
solid result. It is **not** a detector of whether someone is concentrating.

---

## 9. Where my predictions were wrong

Recorded so the reasoning can be audited:

| Prediction | Outcome |
|---|---|
| "Xdawn+TS will win" | **Wrong.** It came 3rd (0.711). EEGNet won at 0.794. |
| "EEGNet unlikely to dramatically beat Xdawn" | **Wrong.** It beat it by 0.083 AUC. |
| "All good models cluster in 0.65–0.78" | **Slightly wrong.** EEGNet reached 0.794, just above the range. |
| "Above 0.85 means hunt for a leak" | Held — 0.794 is below that, and the control window is clean. |
| "ERP features will underperform (timing jitter)" | **Wrong** (from `FEATURE.md`), and for a reason that turned out to matter. |

The pattern in the misses is consistent and worth stating: I repeatedly underweighted
**time-domain** methods. The signal is a sharply time-locked evoked response, so the
models that preserve *when* things happened (EEGNet's temporal convolutions, ERP bins)
beat those that summarise it away (band power) or compress it into 4 spatial filters
(Xdawn). Once the evoked-response nature was clear, this was predictable — I just didn't
update on it early enough.

---

## 10. Figures

| File | Shows |
|---|---|
| `fig1_model_comparison.png` | AUC + accuracy, all models, main vs control window |
| `fig2_confusion_matrices.png` | One panel per model |
| `fig3_eegnet_loss.png` | Training vs validation **loss**, per fold and mean ± sd |
| `fig4_eegnet_accuracy.png` | Training vs validation **accuracy** |
| `fig5_roc.png` | ROC curves, all models |
| `fig6_per_subject.png` | Per-subject AUC, sorted |
| `fig7_time_resolved.png` | **The key figure** — decoding over time |

Colours come from a validated categorical palette (checked with `validate_palette.js`:
lightness band, chroma floor, CVD separation and normal-vision floor all pass). The
contrast check warns on three slots, so every mark carries a printed value.

---

## 11. Output files

```
derivatives/results_cv5.csv        every model × fold × window × metric
derivatives/predictions_cv5.csv    per-trial probabilities (for any re-analysis)
derivatives/eegnet_history.csv     per-epoch train/val loss and accuracy
derivatives/time_resolved_auc.csv  the fig7 numbers
derivatives/train_log.txt          full console log
derivatives/models/eegnet_*.pt     10 trained networks (5 folds × 2 windows)
figures/                           7 PNGs
```

## 12. Reproduce

```bash
python scripts/04_train.py     # ~24 min
python scripts/05_report.py    # ~2 min
```

Add `--quick` to `04_train.py` for a 1-fold, 3-epoch smoke test.

---

## 13. What is still open

- [ ] **Choose the framing** before writing this up. Option A (report as an
      interruption-response detector, AUC 0.794, with §4 and §8 as the honesty evidence)
      is defensible and complete. Option B (an unconfounded within-distractor split) was
      not run — and based on the pre-window results it would very likely land near
      chance, which is itself a reportable finding.
- [ ] Optional: scalp topography of EEGNet's learned spatial filters. If they localise to
      temporal/central sites (auditory cortex), that is a fifth independent confirmation.
- [ ] Optional: LOSO (23-fold) for the classical models as a cross-check on the 5-fold
      estimate. Cheap; EEGNet would be too slow.

**For any future recording, the single change that matters:** give both conditions the
same number of beeps. Everything in §8 traces back to that one design choice.
