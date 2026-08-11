"""
STEP 2 - Clean the raw EEG and cut it into trials ("epochs").

Run order of operations, and WHY each one is there:

  1. Load the raw recording          - 19 channels, 500 samples per second.
  2. Drop EOG + x/y/z_dir            - the EOG electrode is broken in every single
                                       recording (flat or maxed out), and x/y/z are
                                       a motion sensor, not brain activity.
  3. Standard 10-20 montage          - tells MNE where each electrode sits on the head.
  4. Notch filter 50 + 100 Hz        - removes hum from mains electricity.
  5. Band-pass filter 1-45 Hz        - throws away very slow drift (below 1 Hz) and
                                       very fast noise (above 45 Hz). The amplifier
                                       recorded with NO high-pass, so drift is huge
                                       (up to +-30 mV) and this step is essential.
  6. Average reference               - re-expresses every channel relative to the
                                       average of all channels. Standard practice.
  7. ICA                             - separates the signal into independent sources
                                       and removes the ones that look like eye blinks
                                       or muscle tension.
  8. Epoching -0.5 to +5.0 s         - cuts one slice of EEG around each trial marker.
  9. Artifact rejection              - throws away any slice where some channel swings
                                       more than 200 microvolts (almost certainly not brain).

Output:
  derivatives/epochs/<subject>-epo.fif     cleaned trials, labels attached
  derivatives/ica/<subject>-ica.fif        the ICA solution (so this is reproducible)
  derivatives/preprocessing_report.csv     one row per subject, what happened
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EEGD = os.path.join(ROOT, "Notification", "EEG")
DERIV = os.path.join(ROOT, "derivatives")
EPO_D = os.path.join(DERIV, "epochs")
ICA_D = os.path.join(DERIV, "ica")
for d in (EPO_D, ICA_D):
    os.makedirs(d, exist_ok=True)

# ---------------- settings ----------------
DROP_CHANNELS = ["EOG", "x_dir", "y_dir", "z_dir"]
NOTCH_FREQS   = [50, 100]
L_FREQ, H_FREQ = 1.0, 45.0
TMIN, TMAX    = -0.5, 5.0
DECIM         = 2                  # 500 Hz -> 250 Hz (safe: we already cut above 45 Hz)
REJECT_UV     = 200                # peak-to-peak threshold for throwing a trial away
ICA_MAX_BLINK  = 2                 # at most 2 eye-blink components
ICA_MAX_MUSCLE = 2                 # at most 2 muscle components
ICA_MAX_BAD    = 3                 # and never more than 3 in total, out of 14
MUSCLE_THRESH  = 0.8               # stricter than MNE's 0.5 default; 0.5 over-flags
                                   # badly on a 15-channel montage
RANDOM_STATE  = 97

trials = pd.read_csv(os.path.join(DERIV, "trials_master.csv"))
report = []

for vhdr in sorted(glob.glob(os.path.join(EEGD, "*.vhdr"))):
    stem = os.path.basename(vhdr)[:-5]
    subject = stem.split("_", 1)[1]
    meta = trials[trials.subject == subject].reset_index(drop=True)
    if len(meta) == 0:
        print(f"  {subject}: not in trials_master.csv - skipped")
        continue

    r = {"subject": subject}
    print(f"\n=== {subject} ===", flush=True)

    # ---- 1. load ----
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    r["n_ch_raw"] = len(raw.ch_names)
    r["duration_min"] = round(raw.times[-1] / 60, 2)

    # ---- 2. drop non-brain channels ----
    raw.drop_channels([c for c in DROP_CHANNELS if c in raw.ch_names])
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    r["n_ch_eeg"] = len(raw.ch_names)

    # ---- 3. montage ----
    raw.set_montage("standard_1020", match_case=False, on_missing="raise", verbose="ERROR")

    # ---- 4/5. filtering ----
    raw.notch_filter(NOTCH_FREQS, verbose="ERROR")
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="ERROR")

    # ---- 6. average reference ----
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # ---- 7. ICA ----
    # Average referencing removes one degree of freedom, so 15 channels -> rank 14.
    n_comp = len(raw.ch_names) - 1
    ica = mne.preprocessing.ICA(n_components=n_comp, method="infomax",
                                fit_params=dict(extended=True),
                                max_iter="auto", random_state=RANDOM_STATE)
    ica.fit(raw, decim=5, verbose="ERROR")

    def top_by_score(idx, scores, k):
        """Keep the k components with the STRONGEST evidence, not the lowest index.
        ICA components are ordered by variance explained, so slicing by index would
        preferentially delete the biggest components - which may be real brain signal."""
        idx = [int(i) for i in idx]
        if not idx:
            return []
        s = np.abs(np.asarray(scores))
        if s.ndim > 1:                       # find_bads_eog returns one row per ch_name
            s = s.max(axis=0)
        return [i for i in sorted(idx, key=lambda i: -s[i])[:k]]

    # The EOG electrode is dead, so we use the two forehead electrodes (Fp1/Fp2)
    # as stand-ins for an eye channel - blinks are enormous there.
    try:
        eog_idx, eog_scores = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"],
                                                threshold=3.0, verbose="ERROR")
        eog_keep = top_by_score(eog_idx, eog_scores, ICA_MAX_BLINK)
    except Exception as e:
        eog_idx, eog_keep = [], []
        r["eog_error"] = str(e)[:80]
    try:
        mus_idx, mus_scores = ica.find_bads_muscle(raw, threshold=MUSCLE_THRESH,
                                                   verbose="ERROR")
        mus_keep = top_by_score(mus_idx, mus_scores, ICA_MAX_MUSCLE)
    except Exception:
        mus_idx, mus_keep = [], []

    # blinks take priority, then muscle, up to the overall cap
    bad = list(dict.fromkeys(eog_keep + [i for i in mus_keep if i not in eog_keep]))[:ICA_MAX_BAD]
    ica.exclude = bad
    r["ica_n_components"] = n_comp
    r["ica_blink_flagged"] = ",".join(map(str, sorted(int(i) for i in eog_idx)))
    r["ica_muscle_flagged"] = ",".join(map(str, sorted(int(i) for i in mus_idx)))
    r["ica_excluded"] = ",".join(map(str, bad))
    r["ica_n_excluded"] = len(bad)
    ica.save(os.path.join(ICA_D, f"{subject}-ica.fif"), overwrite=True, verbose="ERROR")
    ica.apply(raw, verbose="ERROR")
    print(f"  ICA: removed {len(bad)} of {n_comp} components {bad}", flush=True)

    # ---- 8. epoching ----
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    key = [k for k in event_id if "trial_start" in k][0]
    events = events[events[:, 2] == event_id[key]]
    if len(events) != len(meta):
        print(f"  !! {len(events)} markers vs {len(meta)} trials - SKIPPED")
        r["status"] = "marker mismatch"
        report.append(r)
        continue

    epochs = mne.Epochs(raw, events, event_id={"trial": event_id[key]},
                        tmin=TMIN, tmax=TMAX, baseline=None, metadata=meta,
                        decim=DECIM, preload=True, reject=None, verbose="ERROR")

    # ---- 9. artifact rejection ----
    d = epochs.get_data(copy=True) * 1e6                     # microvolts
    ptp = (d.max(axis=2) - d.min(axis=2)).max(axis=1)        # worst channel per trial
    m = epochs.metadata.copy()
    m["ptp_uv"] = np.round(ptp, 1)
    epochs.metadata = m

    n_before = len(epochs)
    epochs.drop_bad(reject=dict(eeg=REJECT_UV * 1e-6), verbose="ERROR")
    n_after = len(epochs)

    r.update({
        "sfreq_out": epochs.info["sfreq"],
        "n_epochs_before": n_before,
        "n_epochs_kept": n_after,
        "pct_kept": round(100 * n_after / n_before, 1),
        "median_ptp_uv": round(float(np.median(ptp)), 1),
        "n_focus": int((epochs.metadata.condition == "focus").sum()),
        "n_distractor": int((epochs.metadata.condition == "distractor").sum()),
        "status": "ok",
    })
    epochs.save(os.path.join(EPO_D, f"{subject}-epo.fif"), overwrite=True, verbose="ERROR")
    print(f"  kept {n_after}/{n_before} epochs ({r['pct_kept']}%)  "
          f"focus={r['n_focus']} distractor={r['n_distractor']}", flush=True)
    report.append(r)

R = pd.DataFrame(report)
R.to_csv(os.path.join(DERIV, "preprocessing_report.csv"), index=False)
pd.set_option("display.width", 220, "display.max_columns", 30)
print("\n\n" + "=" * 100)
print("PREPROCESSING REPORT")
print(R[["subject", "n_ch_eeg", "duration_min", "ica_n_excluded", "ica_excluded",
         "median_ptp_uv", "n_epochs_before", "n_epochs_kept", "pct_kept",
         "n_focus", "n_distractor", "status"]].to_string(index=False))
print(f"\ntotal epochs kept : {R.n_epochs_kept.sum()} / {R.n_epochs_before.sum()}")
print(f"median retention  : {R.pct_kept.median():.1f}%")
low = R[R.pct_kept < 60]
if len(low):
    print(f"\nLOW RETENTION (<60%) - review these: {low[['subject','pct_kept']].values.tolist()}")
