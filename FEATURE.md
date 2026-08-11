# Step 3 — Feature Extraction

**Script:** `scripts/03_features.py`
**Status:** ✅ complete and cross-verified (0 failures)
**Runtime:** ~3 minutes
**Last updated:** 2026-08-10

---

## 1. Why this step exists

Each cleaned trial is **15 electrodes × 1,376 time points = 20,640 raw numbers**.
We only have **4,544 trials**. A model handed 20,640 inputs and 4,544 examples will
memorise noise rather than learn anything real.

This script squeezes every trial down to a few hundred numbers that actually mean
something — while deliberately keeping some that are *guaranteed to be meaningless*, as
a cheating detector.

**In:** 23 × `derivatives/epochs/<subject>-epo.fif`
**Out:** `derivatives/features.npz` (66.1 MB) + `derivatives/features_meta.csv` (0.29 MB)

---

## 2. The four time windows

The same four slices are cut from every trial. The first two exist **only to catch
cheating** — in both, the two conditions are physically identical, so any honest model
must score ~0.50 there.

| Name | Window | What is happening | Role |
|---|---|---|---|
| `pre` | 0.0–2.0 s | Fixation cross only. No beep yet. | 🛡 **Control** |
| `early` | 2.0–2.5 s | Beep 1 has played — but both conditions get beep 1. | 🛡 **Control** |
| `late` | 2.5–5.0 s | Beeps 2 and 3 — distractor only. | Signal |
| `post` | 2.0–5.0 s | Whole post-stimulus period. | Signal |

Reminder of the trial layout (times from the `trial_start` marker):

```
0.0s  fixation "+"          2.0s  puzzle + BEEP 1        2.5s  BEEP 2 (distractor only)
                                                          4.0s  BEEP 3 (distractor only)
                                                          5.0s  end
```

---

## 3. What was computed

### 3.1 Absolute band power — `bp_abs_*` → `(4544, 15, 5)`

Energy in each frequency band, per electrode, via Welch's method, then `log10`.

Bands: **delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz**

*Why the log?* Raw EEG power is wildly skewed — a few trials are 100× others. The log
squashes it into a roughly bell-shaped spread, which is what linear models expect.

### 3.2 Relative band power — `bp_rel_*` → `(4544, 15, 5)`

Each band as a **fraction of that trial's total 1–45 Hz power**.

*Why compute this when we already have absolute?* Some people simply produce bigger
signals — thicker skull, better electrode contact, more gel. Absolute power therefore
partly measures *the person*. Relative power cancels that out, which matters enormously
for leave-one-subject-out testing, where the model meets someone it has never seen.

### 3.3 Covariance matrices — `cov_bb_*`, `cov_osc_*` → `(4544, 15, 15)`

How strongly each pair of electrodes moves together. This is what **pyriemann** consumes,
and it usually beats band power because it captures *spatial patterns* ("the back-left
and back-right rise together") rather than isolated per-electrode energy.

- `cov_bb` — broadband, 1–45 Hz
- `cov_osc` — oscillatory, 8–30 Hz only

Estimated with **OAS shrinkage**: with 15 channels and short windows, a plain covariance
estimate is unstable, and shrinkage pulls it back toward something well-behaved.

### 3.4 ERP features — `erp_post` → `(4544, 15, 30)`

Average voltage in **100 ms bins** across the `post` window. Band power throws away
*when* something happened; this keeps it.

No baseline correction was applied — the data is already 1 Hz high-passed so drift is
gone, and the pre-trial baseline is contaminated by the *previous* trial's beeps
(PROGRESS.md §3.3).

### 3.5 Sliding-window sweep — `slide_bp` → `(4544, 15, 5, 21)`

Band power in **500 ms windows stepping every 250 ms**, from −0.5 s to 4.5 s (21 windows).

Nearly free to compute now, and it makes the time-resolved decoding plot in Step 5
trivial — the chart showing a flat line at 0.50 until the second beep, then a jump. That
single picture is the clearest possible evidence of the confound.

---

## 4. Two correctness rules built into the script

### Rule 1 — only *unsupervised* features live here

**CSP and Xdawn are deliberately absent.** Both are excellent EEG methods, but both
**look at the labels** to build their spatial filters. Running them here, across the whole
dataset, would mean they had already seen the test subject's labels before
cross-validation even began — and every score afterwards would be inflated.

They belong **inside** each cross-validation fold, in `04_train.py`.

Band power, plain covariance, and ERP bins never look at labels, so they are safe to
compute once up front. This is one of the most common ways EEG papers accidentally
report results that do not replicate.

### Rule 2 — no normalisation baked in

Features are saved **raw**. Per-subject standardisation happens in `04_train.py`, where
it is visible and auditable, rather than being silently welded into the data file.

---

## 5. Sanity checks — all passed

Run automatically before saving:

```
PASS  all feature blocks have 4544 rows and are finite
PASS  row count 4544 matches epoch files (4544)
PASS  class balance: focus=2273  distractor=2271
PASS  all 23 subjects contribute both classes
PASS  metadata alignment re-verified after concatenation
```

The alignment check is **asserted, not assumed** — earlier in this project a real bug
came from two folders sorting differently (`music_pranshu` sorts after `prashant`, while
`pranshu` sorts before it), silently misaligning 10 subjects. The script now re-derives
`label` from the `condition` column and refuses to run if they disagree.

---

## 6. Independent verification

`bp_abs_post` and `bp_rel_post` were **recomputed from the epoch files by separate code**
and compared against what was stored:

| Check | Result |
|---|---|
| `bp_abs_post` recomputed matches stored | max difference **1.12e-07** |
| `bp_rel_post` recomputed matches stored | max difference **9.84e-08** |
| `band_power()` on a synthetic 10 Hz sine | correctly picks **alpha** |
| Covariances symmetric | ✅ |
| Covariances positive-definite (Riemannian requires it) | ✅ min eigenvalue 0.118 |
| No NaN/Inf anywhere | ✅ |
| `features_meta.csv` rows align with feature rows | ✅ |

---

## 7. Do the features actually work?

A quick logistic-regression check (within-subject, 5-fold, AUC). **0.50 = coin flip.**

| Feature block | AUC | Expectation |
|---|---|---|
| `bp_abs_pre` | 0.510 | 🛡 control — must be ~0.50 ✅ |
| `bp_abs_early` | 0.480 | 🛡 control — must be ~0.50 ✅ |
| `bp_abs_late` | **0.653** | signal ✅ |
| `bp_abs_post` | 0.645 | signal ✅ |
| `bp_rel_pre` | 0.502 | 🛡 control ✅ |
| `bp_rel_early` | 0.478 | 🛡 control ✅ |
| `bp_rel_late` | 0.629 | signal ✅ |
| `bp_rel_post` | 0.621 | signal ✅ |
| **`erp_post`** | **0.751** | ← see below |

Both controls sit at chance and both signal windows rise. The features behave exactly as
the physics demands.

### The one result that overturned my prediction

I expected **ERP features to underperform**, because your triggers were sent by software
and trial timing wobbles by up to 143 ms (PROGRESS.md §3.9). ERP analysis normally needs
millisecond precision.

**They came out best — 0.751, clearly ahead of band power's 0.653.**

I was wrong, and the reason is informative. The thing being detected is an **auditory
evoked response** — a sharp, time-locked voltage deflection following each beep. That is
precisely what ERP features are built to capture, and precisely what band power averages
away. Meanwhile 143 ms of jitter is small next to a 3-second window containing beeps
1.5 s apart.

**This is not good news for the science.** A higher score from ERP features is further
confirmation that the model is detecting *sounds*, not *mental states*. The confound
described in PROGRESS.md §3.1 is now supported by three independent lines of evidence:

1. Both physically-identical control windows decode at chance (0.478–0.510).
2. The distractor−focus difference wave is flat before 2.5 s (0.12–0.13 µV RMS) and
   jumps only after beeps 2 and 3 (0.41 and 0.37 µV RMS).
3. Time-locked ERP features beat frequency features — the signature of an evoked
   response to a stimulus, not a sustained cognitive state.

---

## 8. Frequency resolution — one honest caveat

Welch resolution is 1 ÷ window length, so short windows cannot resolve slow rhythms.

| Window | Length | Resolution | delta | theta | alpha | beta | gamma |
|---|---|---|---|---|---|---|---|
| `pre` | 2.0 s | 1.0 Hz | 3 | 4 | 5 | 17 | 15 |
| `early` | **0.5 s** | **2.0 Hz** | **1** ⚠ | 2 | 3 | 8 | 8 |
| `late` | 2.5 s | 1.0 Hz | 3 | 4 | 5 | 17 | 15 |
| `post` | 3.0 s | 1.0 Hz | 3 | 4 | 5 | 17 | 15 |

**`bp_*_early` delta rests on a single measurement.** It is noise, and should not be
interpreted. That is physics, not a bug — but treat any "finding" in that cell with
suspicion.

A second, smaller caveat: the delta band (1–4 Hz) begins right at the high-pass cutoff,
so it sits partly inside the filter's transition band and may carry some residual drift
alongside genuine neural delta. Consistent with this, delta was the weakest useful band
in earlier testing (AUC 0.552).

---

## 9. What is in the output file

```
derivatives/features.npz                                        66.1 MB
├── bp_abs_{pre,early,late,post}    (4544, 15, 5)      1.4 MB each
├── bp_rel_{pre,early,late,post}    (4544, 15, 5)      1.4 MB each
├── cov_bb_{pre,early,late,post}    (4544, 15, 15)     4.1 MB each
├── cov_osc_{pre,early,late,post}   (4544, 15, 15)     4.1 MB each
├── erp_post                        (4544, 15, 30)     8.2 MB
├── slide_bp                        (4544, 15, 5, 21) 28.6 MB
├── subject, label
└── band_names, window_names, window_bounds, slide_starts, slide_len,
    erp_bin_s, sfreq          ← so the arrays are self-describing

derivatives/features_meta.csv                                    0.29 MB
    subject, trial, condition, label, hit, responded, rt,
    keys_flipped, ptp_uv, truth, key_corrected
```

### How to load

```python
import numpy as np, pandas as pd

d = np.load("derivatives/features.npz", allow_pickle=True)
M = pd.read_csv("derivatives/features_meta.csv")

X       = d["bp_rel_post"].reshape(len(d["label"]), -1)   # (4544, 75)
y       = d["label"]                                       # 0 = focus, 1 = distractor
groups  = d["subject"]                                     # use for leave-one-subject-out
```

`groups` is the important one — it is what keeps a person's trials from appearing in
both the training and test sets.

---

## 10. Next: `04_train.py`

- Four models, weakest to strongest: logistic regression → shrinkage LDA →
  **CSP + LDA** → **Riemannian tangent space + LR** (CSP and Xdawn fitted *inside* each
  fold, per Rule 1 above).
- **Leave-One-Subject-Out is the number that gets reported.** Within-subject scores are
  computed too, for comparison only.
- Three guardrails run automatically every time: shuffled labels (must be ≈0.50), the
  `pre` control window (must be ≈0.50), and the previous-trial carryover check.
- Both framings evaluated side by side — focus vs distractor, and the unconfounded
  within-distractor split by outcome — so the choice is made from numbers, not a guess.

**Realistic expectation:** LOSO AUC **0.65–0.75** is a good result here. Anything above
0.85 should be treated as a bug until proven otherwise.
