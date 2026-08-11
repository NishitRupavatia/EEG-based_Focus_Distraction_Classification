# EEG Focus vs Distraction — Project Progress

**Goal:** build a machine-learning model that classifies EEG trials as **focus** or **distraction**.

**Last updated:** 2026-08-10
**Status:** ✅ pipeline complete end to end — dataset corrected, preprocessing done,
features extracted (`FEATURE.md`), **models trained and evaluated (`RESULTS.md`)**.
**Best model:** EEGNet, subject-independent 5-fold — **accuracy 0.724, ROC-AUC 0.794**.
**Verification:** full independent cross-check of every stage — **0 failures** (§7).
All models score 0.486–0.516 on the physically-identical control window, confirming
no leakage — but also confirming the model detects the notification *sounds*, not
attention (§3.1, `RESULTS.md` §8).

---

## 1. What the dataset is

An EEG experiment where people solve small word puzzles while phone-style notifications
interrupt them.

| Item | Value |
|---|---|
| Original participants | 24 |
| Participants in use now | **23** (pranshu removed — see §3) |
| Trials per participant | 200 |
| Total trials | **4,600** |
| Class balance | Perfect — 100 focus / 100 distractor per person |
| Amplifier | BrainVision LiveAmp (SN 104201-1541) |
| Sampling rate | 500 Hz |
| Recording length | ~18 minutes per person |
| Raw channels | 19 |
| EEG channels used | **15** — Fp1, Fp2, F3, Fz, F4, C3, C4, Cz, T7, T8, P3, P4, Pz, O1, O2 |
| Channels dropped | EOG (broken), x_dir, y_dir, z_dir (motion sensor) |
| Behavioural software | PsychoPy 2026.1.3 |
| Recording dates | 2026-05-15 to 2026-05-27 |

### File layout

```
EEG_signal/
├── Notification/
│   ├── EEG/       23 × (.eeg .vhdr .vmrk)   raw BrainVision recordings
│   ├── CSV/       23 × (.csv .log .psydat)  PsychoPy behavioural files
│   └── excluded/  pranshu's raw files, moved out of the way (not deleted)
├── scripts/
│   ├── 01_build_trials.py     builds the corrected master trial table
│   └── 02_preprocess.py       the EEG cleaning pipeline
├── derivatives/               ← everything produced by the scripts
│   ├── trials_master.csv      4,600 rows: one per trial, labels + corrections
│   ├── epochs/<subj>-epo.fif  cleaned, cut-up EEG ready for machine learning
│   ├── ica/<subj>-ica.fif     the ICA solution for each person (reproducibility)
│   └── preprocessing_report.csv   what happened to each person
└── PROGRESS.md                this file
```

### The task, in plain words

Each person saw a scrambled word and a claim like *"(count C + N) = 2"*, and pressed
**RIGHT arrow** if the claim was true or **LEFT arrow** if false. They had 3 seconds.
While they worked, a notification popped up. In **focus** trials there was **1 beep**;
in **distractor** trials there were **3 beeps**.

### What happens inside one trial

All times are measured from the `trial_start` marker in the EEG file.

```
t = 0.0 s   a "+" appears on screen (fixation)         ← the EEG marker is HERE
t = 2.0 s   puzzle + notification image + BEEP 1
            the answer window opens now
t = 2.5 s   BEEP 2   ← ONLY in distractor trials
t = 4.0 s   BEEP 3   ← ONLY in distractor trials
t = 5.0 s   trial ends / answer deadline
```

> **Careful:** the marker is called `trial_start` but it sits at the **fixation cross**,
> not at the puzzle. The puzzle appears **2 seconds later**. Getting this wrong shifts
> every analysis by 2 seconds.

### Did the experiment actually work?

Yes — distraction measurably hurt performance:

| Measure | Focus | Distractor | Difference | Significance |
|---|---|---|---|---|
| Accuracy | 0.806 | 0.754 | −0.051 | t(23) = 4.30, **p = 0.0003** |
| Median RT | 2.176 s | 2.222 s | +0.046 s | p = 0.08 (not significant) |
| Missed trials | 11.25 | 11.67 | +0.42 | p = 0.72 (not significant) |

---

## 2. Corrections applied to the dataset

### 2.1 `ganesh` and `siddhartha` — answer keys were reversed ✅ FIXED

Their raw accuracy was **10.4%** and **16.4%**. Pure guessing gives 50%. Scoring *far
below* chance is not "being bad at the task" — it is the signature of someone answering
correctly but with the **left/right keys swapped**. Two pieces of evidence confirm it:

1. Flipping the answers gives **89.6%** and **83.6%** — completely normal scores.
2. Their tendency to press "right" was 0.486 and 0.451, i.e. balanced. A person mashing
   one key randomly would show a lopsided value.

**What was changed:** in `derivatives/trials_master.csv`, the `key_corrected` column has
`left`↔`right` swapped for these two people, and `hit` (correct/incorrect) is recomputed
from it. The original values are preserved in `key_raw`, and `keys_flipped = True` marks
the affected rows.

**What was NOT changed:** the `condition` column (focus / distractor). That is the
machine-learning target and it was never wrong — it comes from the experiment design,
not from the participant.

### 2.2 `pranshu` — removed entirely ✅ REMOVED

Three independent problems:

1. His EEG file is named **`music_pranshu`**, not `focusdistraction_pranshu` — it was
   recorded under a different experiment profile.
2. His EEG has **210** `trial_start` markers but his CSV has **200** trials. We pair EEG
   to labels *by order*, so 10 extra markers make every label after the first mismatch
   unreliable.
3. He has one inter-trial gap of **11.76 s** where everyone else is a steady 5.0 s.

**What was done:** his 6 raw files were **moved** to `Notification/excluded/`, not
deleted, so nothing is permanently lost. He is excluded from `trials_master.csv` and from
all preprocessing output.

### 2.3 Accuracy per person after the fixes

Group accuracy rose from 78.4% → **84.5%**.

```
rachit 0.980   rakshit 0.964   Ashish 0.929   nishit 0.909   sathwik 0.909
anurag 0.906   bhaskar 0.899   ganesh 0.896*  vansh  0.893   harshav. 0.879
vipul  0.865   aryan   0.846   aditya 0.842   david  0.837   siddhartha 0.836*
sooraj 0.824   digvijay 0.813  prashant 0.811 pragnesh 0.809 bhawesh 0.805
naveen 0.734   mahendra 0.564  rishi  0.482
                                   * = key-flip applied
```

---

## 3. Known data issues

Ordered by how much they matter.

### 🔴 3.1 The label is confounded with the sound — THE critical issue

The two conditions are **not** "concentrating" vs "distracted". They are **1 beep** vs
**3 beeps**. The notification *image* is identical in both (verified across all 4,800
original trials). The only physical difference is the two extra beeps at 2.5 s and 4.0 s.

I tested where in the trial the classifier actually finds a difference
(band power, 15 channels × 5 frequency bands, logistic regression):

| Time window | Is the stimulus identical here? | Within-person AUC | Across-people AUC |
|---|---|---|---|
| 0.0–2.0 s (fixation, before any beep) | **yes** | 0.489 | 0.523 |
| 2.0–2.5 s (after beep 1, before beep 2) | **yes** | 0.490 | 0.506 |
| 2.5–5.0 s (beeps 2 & 3 have happened) | no | **0.645** | **0.622** |
| 2.0–5.0 s (whole post-stimulus period) | no | 0.634 | 0.607 |
| shuffled labels (sanity check) | — | 0.486 | — |

*AUC 0.50 = coin flip. AUC 1.00 = perfect.*

**Read it like this:** for as long as the two conditions look and sound identical, the
model is at chance. The moment the extra beeps arrive, accuracy jumps. **The model is
detecting beeps, not mental states.**

Two alternatives were ruled out:
- **Not the button press:** which key was pressed decodes at only AUC 0.529, and on
  trials where the person never answered at all, condition still decodes at 0.604.
- **Not random noise:** shuffling the labels gives 0.486, as expected.

**Realistic ceiling for this dataset: AUC ≈ 0.62–0.65 (~60% accuracy).** Any result
above ~0.85 means data leakage — most likely pooling one person's trials across the
train and test sets.

### 🟠 3.2 The EOG (eye) electrode is dead in every recording

The header of every file says `EOG: Out of Range!`. Measured:
- **21 of 24 people: standard deviation = 0.0 µV** (completely flat, no signal)
- **anurag, pragnesh, rakshit: 50,922 / 66,993 / 103,001 µV** (electrode railing)

Consequence: the normal way of removing blinks (subtracting the eye channel) is
impossible. We use ICA with the forehead electrodes as substitutes instead. Also, if
you leave EOG in before average-referencing, it destroys those three recordings.

### 🟠 3.3 There is no clean baseline period anywhere

The 2-second fixation of a trial begins only **1.0 s** after the previous trial's last
beep. Measured consequence: the **previous** trial's condition can be decoded from the
current trial's "clean" window at **AUC 0.553** — *higher* than the current trial's own
condition (0.487). Nothing in this recording is truly free of the notification sounds.

### 🟡 3.4 No high-pass filter on the amplifier

Header says `Low Cutoff = DC`. Median DC offset **6,829 µV**, maximum **28,821 µV**, and
raw peak amplitude up to **38,232 µV** — against real brain signals of about 10 µV.
Handled by the 1 Hz high-pass in preprocessing, but it means the raw files look alarming.

### 🟡 3.5 Mains hum in four people

50 Hz power relative to neighbouring frequencies: **rachit 27.6×, bhaskar 26.5×,
ganesh 11.6×, rakshit 8.7×**. Everyone else ≈ 1.0×. Handled by the notch filter.

### 🟡 3.6 Two people did not really do the task

| Person | Accuracy | Problem |
|---|---|---|
| `rishi` | 0.482 | At chance, and **88 of 200 trials had no answer at all** (44%) |
| `mahendra` | 0.564 | Near chance; pressed "right" only 24% of the time (lopsided) |

They are **still in the dataset** — their EEG is fine and the focus/distractor labels
are valid regardless of whether they answered well. Recommendation: exclude them for any
analysis that uses accuracy or reaction time, keep them for pure condition decoding.

### 🟡 3.7 Three noisy recordings

Median trial peak-to-peak amplitude *before* cleaning: **anurag 295 µV, aditya 233 µV,
pragnesh 199 µV**, against a group median of 124 µV (cleanest is nishit at 46 µV).

**Largely resolved by ICA** — see §4. anurag and aditya now retain 100% and 99.5% of
their trials. Only **pragnesh** is still affected (80.5% retention, 161 trials).

### 🟡 3.8 Missed responses: 10.8% of trials

Overall 497 of 4,600 trials have no keypress. Worst: rishi 88, aditya 54, bhawesh 46,
prashant 36. Plus 4 stray `down` keypresses (sathwik) which count as invalid.

### ⚪ 3.9 Timing jitter of up to ~116 ms

The EEG markers are sent by software, not hardware. Comparing marker times to the
PsychoPy log gives a median error of 6–88 ms (worst: aditya at 88 ms), max ~116 ms.
Fine for frequency-band features; a real limitation for sharp ERP analysis.

### ⚪ 3.10 The EEG markers carry no label

Every one of the 200 markers is the identical string `trial_start`. The focus/distractor
information exists **only** in the CSV, matched by row order. This is exactly why
pranshu became unrecoverable. `trials_master.csv` now stores that pairing explicitly.

---

## 4. Preprocessing completed

Script: `scripts/02_preprocess.py`. Applied identically to all 23 people.

| # | Step | Setting |
|---|---|---|
| 1 | Load raw BrainVision | 19 ch, 500 Hz |
| 2 | Drop non-brain channels | removed EOG, x_dir, y_dir, z_dir → **15 EEG channels** |
| 3 | Electrode positions | `standard_1020` montage |
| 4 | Notch filter | 50 Hz and 100 Hz |
| 5 | Band-pass filter | 1–45 Hz, FIR, `firwin` |
| 6 | Reference | average of all 15 channels |
| 7 | ICA | extended Infomax, 14 components, `random_state=97`; blinks found via Fp1/Fp2 as EOG substitutes (threshold 3.0), muscle via `find_bads_muscle(threshold=0.8)`; components ranked **by score**, max 2 blink + 2 muscle, **hard cap 3 total** |
| 8 | Epoching | −0.5 s to +5.0 s around each `trial_start`, **no baseline correction**, downsampled to 250 Hz |
| 9 | Artifact rejection | drop any trial with peak-to-peak > **200 µV** on any channel |

> **Note on the ICA settings.** The first run used MNE's default muscle threshold of 0.5
> and capped at 4 components. On a 15-channel montage that default flagged up to **10 of
> 14** components as "muscle", which is not credible, and the code then kept the
> *lowest-numbered* ones. Since ICA orders components by variance, that quietly removed
> the biggest components — potentially real brain signal. Fixed: the threshold is now
> 0.8, selection is ranked by artifact score, and the cap is 3. Reassuringly, the blink
> detector found **exactly 1 component for all 23 people**, and it was component 0 for
> 22 of them (blinks are the largest source of variance, so this is expected).

Two deliberate choices:
- **Downsampled to 250 Hz.** Nothing above 45 Hz survives the filter, so the extra
  samples carry no information. Halves the file size for free.
- **Baseline correction deliberately NOT applied.** Because of issue §3.3 the baseline
  window is contaminated. Apply it later if you want it:
  `epochs.apply_baseline((-0.5, 0))`.

### Results

**4,544 of 4,600 trials survived cleaning (98.8%).** Median retention per person: 100%.
Class balance is preserved almost exactly: **2,273 focus / 2,271 distractor**.

| subject | ICA removed | median p2p (µV) | epochs kept | % kept | focus | distractor |
|---|---|---|---|---|---|---|
| Ashish | 0,8,7 | 46.2 | 198 | 99.0 | 100 | 98 |
| aditya | 0,9 | 57.2 | 199 | 99.5 | 99 | 100 |
| anurag | 0,5,9 | 70.0 | 200 | 100.0 | 100 | 100 |
| aryan | 0,3,1 | 57.2 | 199 | 99.5 | 99 | 100 |
| bhaskar | 0,2 | 73.2 | 200 | 100.0 | 100 | 100 |
| bhawesh | 0,1,6 | 66.1 | 200 | 100.0 | 100 | 100 |
| david | 0,2,3 | 69.6 | 200 | 100.0 | 100 | 100 |
| digvijay | 0,5,10 | 42.8 | 199 | 99.5 | 99 | 100 |
| ganesh | 0,12,11 | 55.6 | 199 | 99.5 | 100 | 99 |
| harshavardhan | 0,9,11 | 36.5 | 200 | 100.0 | 100 | 100 |
| mahendra | 0,3,6 | 47.2 | 200 | 100.0 | 100 | 100 |
| naveen | 0,9 | 82.1 | 200 | 100.0 | 100 | 100 |
| nishit | 0,6,7 | 37.3 | 200 | 100.0 | 100 | 100 |
| **pragnesh** | 0,10,7 | 89.2 | **161** | **80.5** | 81 | 80 |
| prashant | 0,9,6 | 67.5 | 200 | 100.0 | 100 | 100 |
| rachit | 1 | 59.4 | 200 | 100.0 | 100 | 100 |
| rakshit | 0,6,10 | 76.9 | 197 | 98.5 | 98 | 99 |
| rishi | 0,2,3 | 43.8 | 200 | 100.0 | 100 | 100 |
| sathwik | 0,10,1 | 61.9 | 199 | 99.5 | 100 | 99 |
| siddhartha | 0,7,1 | 49.4 | 195 | 97.5 | 99 | 96 |
| sooraj | 0,8,3 | 44.4 | 200 | 100.0 | 100 | 100 |
| vansh | 0,6,5 | 35.3 | 199 | 99.5 | 99 | 100 |
| vipul | 0,1,2 | 48.4 | 199 | 99.5 | 99 | 100 |

**The cleaning worked.** Two independent signs:

1. **Amplitudes are now physiological.** Median absolute voltage is **3.6 µV** and the
   99th percentile is **17.7 µV** — exactly the range real EEG lives in. The raw files
   had DC offsets in the tens of *thousands* of µV.
2. **The three "noisy" people were rescued by ICA, not discarded.** Before cleaning,
   anurag's median trial swing was 295 µV and a 200 µV cut-off would have thrown away
   *100%* of his trials. After ICA he keeps **all 200**. aditya (233 µV → keeps 199) and
   pragnesh (199 µV → keeps 161) improved the same way. Removing the blink component
   was what fixed them.

Only **pragnesh** falls below 90% retention (80.5%). He is still usable — 161 balanced
trials — but flag him if his results look odd.

### Verified properties of the output files

```
shape        (n_trials, 15, 1376)     e.g. Ashish -> (198, 15, 1376)
sampling     250 Hz
time range   -0.5 s to +5.0 s
channels     Fp1 Fp2 F3 Fz F4 C3 C4 Cz T7 T8 P3 P4 Pz O1 O2
montage      standard_1020, attached
reference    average (custom_ref_applied = 1)
total size   377 MB across 23 files
```

### How to load the cleaned data

```python
import mne, glob
epochs = mne.read_epochs("derivatives/epochs/aditya-epo.fif")
X = epochs.get_data()                      # (n_trials, 15 channels, 1376 samples)
y = epochs.metadata["label"].values        # 0 = focus, 1 = distractor
```

Every trial's labels and behaviour travel with the data in `epochs.metadata`:
`condition`, `label`, `hit`, `responded`, `rt`, `key_corrected`, `keys_flipped`,
`ptp_uv`, `trial`, `subject`.

---

## 5. What still needs to be done

### ✅ Done — feature extraction (`scripts/03_features.py`, see `FEATURE.md`)
- [x] Log absolute + relative band power, 5 bands × 15 channels × 4 time windows
- [x] Covariance matrices (broadband and 8–30 Hz) for the Riemannian pipeline
- [x] ERP features — 100 ms bins over the post window
- [x] Sliding-window sweep (21 × 500 ms) for the time-resolved decoding plot
- [x] Verified: controls at chance (0.478–0.510), signal windows 0.62–0.65,
      `erp_post` best at 0.751

### ✅ Done — training and evaluation (`scripts/04_train.py`, see `RESULTS.md`)
- [x] 6 models, subject-independent 5-fold, 2 windows, 0 failures, 24 min
- [x] EEGNet **0.794 AUC / 0.724 accuracy**; LDA-shrinkage 0.776; Xdawn+TS 0.710
- [x] Guardrails all pass: control window 0.486–0.516, shuffled labels 0.477
- [x] 23/23 subjects above chance

### ✅ Done — figures (`scripts/05_report.py`)
- [x] Model comparison, confusion matrices, EEGNet train-vs-val loss and accuracy
      curves, ROC, per-subject, and the time-resolved decoding figure

### ✅ Done — Colab bundle (`scripts/06_make_colab_bundle.py`, see `colab_bundle/README.md`)
- [x] `colab_bundle.zip` (349 MB) — notebook + scripts + cleaned data, self-contained
- [x] All 5 classical models verified to reproduce **exactly** from the bundle
- [x] Measured runtimes: **~10 min on a Colab T4 GPU, 60–90 min on CPU**
      (EEGNet is 97% of it, at 2.5 s/epoch on 16 cores)

### Still open
- [ ] **Pick the framing before writing up** — see `RESULTS.md` §13
- [ ] Optional: scalp topography of EEGNet's spatial filters (expect temporal/central
      = auditory cortex, which would be a fifth confirmation of the confound)
- [ ] Optional: LOSO for the classical models as a cross-check on the 5-fold estimate

### Decide the honest framing of the problem — pick one
- [ ] **Option A — keep the current labels.** Report it as *"detecting the brain's
      response to an interruption"*, use the 2.0–5.0 s window, expect AUC ≈ 0.65, and
      publish the 0–2 s control result (0.489) as proof you are not overclaiming.
- [ ] **Option B — use a label that is not confounded.** Within distractor trials only,
      split by outcome (correct vs error, or fast vs slow RT). Both classes then have
      *identical* sound and vision, so any difference is genuinely cognitive. Fewer
      trials per class, but it is a real attention decoder.

### Quality decisions still open
- [ ] Decide whether to exclude `rishi` and `mahendra` (see §3.6). Dropping the
      behaviourally-invalid people raised within-person AUC from 0.634 → 0.651.
- [ ] Decide what to do about `pragnesh` — the only person below 90% retention (80.5%,
      161 trials). Either keep him as-is, or give him a looser personal threshold.
- [ ] Visually inspect the ICA components for 3–4 people to confirm the automatic
      detection picked sensible components:
      `mne.preprocessing.read_ica("derivatives/ica/aditya-ica.fif").plot_components()`.
      Component 0 should look like a blink — strong and same-signed at Fp1/Fp2.

### For any future recording session
- [ ] **Give both conditions the same number of beeps.** This single change fixes §3.1
      and is the difference between "beep detector" and "attention decoder".
- [ ] Make the inter-trial gap 3–4 s and jitter it, so a clean baseline exists (§3.3).
- [ ] Use **blocks** (2 minutes of focus, 2 minutes of distraction) instead of
      interleaved 5-second trials. Sustained attention is a slow-moving state and 3
      seconds is too short to see it.
- [ ] Fix the EOG electrode and check its impedance before every session (§3.2).
- [ ] Turn on the amplifier's 0.1 Hz high-pass (§3.4).
- [ ] **Put the condition into the trigger name** (`focus_start` / `distractor_start`)
      so the EEG file is self-labelling (§3.10).

---

## 6. How to reproduce everything

```bash
python scripts/01_build_trials.py    # ~5 seconds  -> derivatives/trials_master.csv
python scripts/02_preprocess.py      # ~10 minutes -> derivatives/epochs/, ica/, report
```

```bash
python scripts/03_features.py      # ~3 minutes  -> derivatives/features.npz
```

Requires: `mne`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyriemann`.
Verified with mne 1.12.1, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0,
pyriemann 0.12, Python 3.13.6.

---

## 7. Verification record

Every stage was independently re-checked — recomputing from source rather than trusting
what the pipeline reported. **Result: 0 failures.**

| Area | Checked | Outcome |
|---|---|---|
| **A. pranshu removed** | absent from EEG/, CSV/, trials_master, epochs, features; 6 raw files preserved in `excluded/` | ✅ 6/6 |
| **B. Key flip** | swap applied correctly; `hit` recomputed by separate code; only those 2 subjects touched; **condition/label untouched** | ✅ 8/8 |
| **C. Trial counts** | 4,600 → 4,544 consistent across trials_master, epochs, features, features_meta, per subject | ✅ 5/5 |
| **D. EEG↔CSV alignment** | all 23 subjects re-derived from `.vmrk` + PsychoPy logs; `eeg_sample` matches the marker file exactly | ✅ median drift 12.4 ms, worst 143.2 ms (sathwik) |
| **E. Preprocessing** | channels, montage, average reference (max \|mean\| 5.6e-07 µV), sfreq/window, rejection (max kept 187.3 µV), filters, ICA | ✅ 10/10 |
| **F. Features** | `bp_abs`/`bp_rel` recomputed from scratch (max diff 1.1e-07); synthetic 10 Hz sine → alpha; covariances symmetric + positive-definite | ✅ 8/8 |
| **G. Physiological proof** | grand-average difference wave timing | ✅ see below |

### Proof that ICA actually ran

The same pipeline was rebuilt **without** ICA and compared:
**Fp1 variance 637 → 20 µV²** — 97% of the frontal blink energy removed.

### Proof that the alignment is right, and that the confound is real

Grand-average difference wave (distractor − focus) at Cz, RMS in µV. If the alignment
were wrong, the pattern below would be smeared or shifted:

```
-0.5 to 0.0 s   0.09 µV   pre-trial baseline
 0.0 to 2.0 s   0.13 µV   CONTROL — no beep yet
 2.0 to 2.5 s   0.12 µV   CONTROL — after beep 1 only
 2.5 to 3.5 s   0.41 µV   after BEEP 2   ← jumps here
 3.5 to 4.0 s   0.14 µV   gap — falls back
 4.0 to 5.0 s   0.37 µV   after BEEP 3   ← jumps again
```

The difference is flat while the conditions are physically identical, and rises **only**
after each extra beep, landing exactly where the beeps are. This simultaneously confirms
the trial alignment is correct **and** that the class difference is an auditory response
rather than a mental state (§3.1).

### Two false alarms worth recording

The first verification run reported 2 failures — both were faults in the *test*, not the
pipeline:

1. **"50 Hz mains not removed."** The test compared absolute power at 50 Hz against
   alpha. But 50 Hz sits inside the 45 Hz low-pass **transition band** (MNE's default
   width is 11.25 Hz, so full attenuation only arrives near 56 Hz). The correct test is
   whether a *peak* survives. It does not — 50 Hz is a **dip** at 0.114× its neighbours,
   and the four worst raw subjects all pass (rachit 0.153×, bhaskar 0.505×,
   ganesh 0.132×, rakshit 0.130×).
2. **"1 Hz high-pass missing."** The test used 0.5 Hz frequency resolution, which
   resolved only a single DC bin. At full resolution the rolloff is clearly present
   (0.65 < 3.70 < 12.03 × alpha as you descend), and DC drift is gone — max epoch mean
   **2.22 µV**, against raw offsets up to 28,821 µV.
