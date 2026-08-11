"""
STEP 1 - Build one clean master table of every trial in the study.

What this script does, in plain words:
  * Reads each person's PsychoPy CSV (the behaviour file: what they saw, what key
    they pressed, how fast).
  * Reads each person's .vmrk file (the EEG marker file) to get the exact moment
    in the EEG recording where each trial started.
  * Fixes two people whose answer keys were reversed (ganesh, siddhartha).
  * Writes everything into ONE file: derivatives/trials_master.csv

Why we need it:
  The EEG file has 200 markers all named "trial_start" - it does NOT store which
  trial was focus and which was distractor. That information lives only in the CSV.
  So we must pair them up by order and save the result once, carefully.
"""
import os, re, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EEGD = os.path.join(ROOT, "Notification", "EEG")
CSVD = os.path.join(ROOT, "Notification", "CSV")
OUTD = os.path.join(ROOT, "derivatives")
os.makedirs(OUTD, exist_ok=True)

# People whose LEFT/RIGHT answer keys were swapped during the session.
# Their raw accuracy came out at ~10-16%, i.e. almost perfectly WRONG, which is
# the signature of someone doing the task properly but with the mapping reversed.
FLIP_RESPONSES = {"ganesh", "siddhartha"}

# People removed from the study entirely (see PROGRESS.md for the reason).
EXCLUDED = {"pranshu"}

SFREQ = 500.0


def read_markers(vmrk_path):
    """Pull the sample number of every 'trial_start' marker out of a .vmrk file."""
    samples = []
    for line in open(vmrk_path, encoding="utf-8", errors="ignore"):
        if not line.startswith("Mk"):
            continue
        # format:  Mk4=STIM,trial_start,34731,1,0
        parts = line.split("=", 1)[1].split(",")
        if len(parts) >= 3 and parts[1].strip() == "trial_start":
            samples.append(int(parts[2]))
    return np.array(samples, dtype=int)


rows = []
for vhdr in sorted(glob.glob(os.path.join(EEGD, "*.vhdr"))):
    stem = os.path.basename(vhdr)[:-5]          # focusdistraction_aditya
    subject = stem.split("_", 1)[1]             # aditya
    if subject in EXCLUDED:
        print(f"  skipping excluded subject: {subject}")
        continue

    # ---- behaviour ----
    matches = glob.glob(os.path.join(CSVD, f"{subject}_visual*.csv"))
    if not matches:
        print(f"  !! no CSV found for {subject}, skipping")
        continue
    df = pd.read_csv(matches[0])
    df.columns = [c.lstrip("﻿") for c in df.columns]   # strip BOM on first column
    # Real trials are the rows that actually have a condition; the first two rows
    # are the instruction/practice screens and are blank here.
    b = df.dropna(subset=["condition"]).reset_index(drop=True)

    # ---- EEG markers ----
    samples = read_markers(os.path.join(EEGD, stem + ".vmrk"))
    if len(samples) != len(b):
        print(f"  !! {subject}: {len(samples)} EEG markers vs {len(b)} CSV trials - SKIPPED")
        continue

    # ---- fix reversed answer keys ----
    key_raw = b["key_resp.keys"].astype(object)
    flipped = subject in FLIP_RESPONSES
    if flipped:
        key = key_raw.map({"left": "right", "right": "left"}).fillna(key_raw)
    else:
        key = key_raw.copy()

    responded = key.isin(["left", "right"])
    # "right" arrow means the puzzle statement is TRUE, "left" means FALSE.
    hit = ((key == "right") & (b["correct"] == "right")) | \
          ((key == "left") & (b["correct"] == "wrong"))

    rows.append(pd.DataFrame({
        "subject":        subject,
        "trial":          np.arange(len(b)),
        "condition":      b["condition"].values,                       # focus / distractor
        "label":          (b["condition"] == "distractor").astype(int).values,  # 0 / 1
        "eeg_sample":     samples,
        "eeg_time_s":     samples / SFREQ,
        "puzzle":         b["puzzle"].values,
        "truth":          b["correct"].values,                         # right / wrong
        "key_raw":        key_raw.values,
        "key_corrected":  key.values,
        "keys_flipped":   flipped,
        "responded":      responded.values,
        "hit":            (hit & responded).values,
        "rt":             b["key_resp.rt"].values,
    }))
    print(f"  {subject:<14} {len(b)} trials   flipped={flipped}")

T = pd.concat(rows, ignore_index=True)
out = os.path.join(OUTD, "trials_master.csv")
T.to_csv(out, index=False)

print("\n" + "=" * 70)
print(f"subjects : {T.subject.nunique()}")
print(f"trials   : {len(T)}")
print(f"balance  : {T.condition.value_counts().to_dict()}")
print(f"responded: {T.responded.mean():.1%}   accuracy(on responded): {T.loc[T.responded,'hit'].mean():.1%}")
print("\naccuracy per subject AFTER the key fix:")
acc = (T[T.responded].groupby("subject")
       .agg(accuracy=("hit", "mean"), n_resp=("hit", "size"))
       .join(T.groupby("subject")["keys_flipped"].first())
       .sort_values("accuracy").round(3))
print(acc.to_string())
print(f"\nwritten -> {out}")
