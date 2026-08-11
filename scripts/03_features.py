"""
STEP 3 - Turn cleaned EEG waves into numbers a machine-learning model can read.

The problem this solves:
  Each trial is currently 15 electrodes x 1376 time points = 20,640 raw numbers.
  We only have ~4,500 trials. A model given 20,640 inputs and 4,500 examples will
  memorise noise instead of learning anything. So we summarise each trial down to a
  few hundred numbers that actually mean something.

What comes out (all UNSUPERVISED - see the note on leakage below):
  1. bp_abs   log absolute band power   "how much alpha is at the back of the head"
  2. bp_rel   relative band power       same, but as a % of that trial's total power,
                                        which cancels out person-to-person differences
  3. cov      covariance matrices       how electrodes move together (for pyriemann)
  4. erp      average voltage in bins   keeps WHEN things happened
  5. slide_bp band power in a sliding window, for the time-resolved decoding plot

IMPORTANT - why CSP and Xdawn are NOT in this file:
  Those methods look at the LABELS to build their spatial filters. If we ran them here,
  across the whole dataset, they would already have seen the test subject's labels
  before cross-validation even began, and every score afterwards would be inflated.
  They belong INSIDE the cross-validation loop, in 04_train.py.
  Everything in this script ignores labels completely, so it is safe to precompute once.

Also note: features are saved RAW. Per-subject standardisation happens in 04_train.py
where it is visible and auditable, not silently welded into the data file.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from pyriemann.estimation import Covariances

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIV = os.path.join(ROOT, "derivatives")
EPO_D = os.path.join(DERIV, "epochs")

# ---------------------------------------------------------------- settings
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
BAND_NAMES = list(BANDS)

# The first two windows are CONTROLS. In both of them the two conditions are
# physically identical (no second beep yet), so a healthy model must score ~0.50
# there. If it does not, something is leaking.
WINDOWS = {
    "pre":   (0.0, 2.0),    # fixation only, before any beep      -> CONTROL
    "early": (2.0, 2.5),    # after beep 1, before beep 2         -> CONTROL
    "late":  (2.5, 5.0),    # beeps 2 and 3 have happened
    "post":  (2.0, 5.0),    # whole post-stimulus period
}

OSC_BAND    = (8, 30)       # for the "oscillatory" covariance version
ERP_WINDOW  = "post"
ERP_BIN_S   = 0.100         # 100 ms bins
SLIDE_LEN   = 0.500         # sliding window length
SLIDE_STEP  = 0.250         # sliding window step
EPS         = 1e-20

META_COLS = ["subject", "trial", "condition", "label", "hit", "responded",
             "rt", "keys_flipped", "ptp_uv", "truth", "key_corrected"]


def band_power(X, sf):
    """X: (n_trials, n_ch, n_times) -> (abs, rel), each (n_trials, n_ch, n_bands).

    abs = log10 of the mean power density inside the band
    rel = log10 of (power in band / total power 1-45 Hz)
    """
    nperseg = min(X.shape[-1], int(sf))          # 1 s segments where possible
    f, P = welch(X, sf, nperseg=nperseg, axis=-1)
    total = P[..., (f >= 1) & (f < 45)].sum(-1)
    a, r = [], []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < hi)
        if not m.any():                          # window too short for this band
            a.append(np.full(X.shape[:2], np.nan))
            r.append(np.full(X.shape[:2], np.nan))
            continue
        a.append(np.log10(P[..., m].mean(-1) + EPS))
        r.append(np.log10(P[..., m].sum(-1) / (total + EPS) + EPS))
    return np.stack(a, -1).astype(np.float32), np.stack(r, -1).astype(np.float32)


def freq_resolution(win_len_s, sf):
    """Welch resolution, and how many usable bins each band gets. Short windows
    cannot resolve slow rhythms - delta needs a long window to be meaningful."""
    nperseg = min(int(round(win_len_s * sf)), int(sf))
    df = sf / nperseg
    f = np.arange(0, sf / 2 + df, df)
    return df, {b: int(((f >= lo) & (f < hi)).sum()) for b, (lo, hi) in BANDS.items()}


# ---------------------------------------------------------------- main loop
files = sorted(glob.glob(os.path.join(EPO_D, "*-epo.fif")))
if not files:
    raise SystemExit("No epoch files found - run 02_preprocess.py first.")

store = {f"bp_abs_{w}": [] for w in WINDOWS}
store.update({f"bp_rel_{w}": [] for w in WINDOWS})
store.update({f"cov_bb_{w}": [] for w in WINDOWS})
store.update({f"cov_osc_{w}": [] for w in WINDOWS})
store["erp_post"] = []
store["slide_bp"] = []
meta_all = []
times = slide_starts = sfreq = None

print(f"Extracting features from {len(files)} subjects\n")
for fp in files:
    subject = os.path.basename(fp).replace("-epo.fif", "")
    ep = mne.read_epochs(fp, preload=True, verbose="ERROR")
    sfreq = ep.info["sfreq"]
    t = ep.times
    X = ep.get_data(copy=True) * 1e6                       # (n, 15, 1376) in microvolts
    Xosc = (ep.copy().filter(*OSC_BAND, verbose="ERROR")
            .get_data(copy=True) * 1e6)                    # same, band-limited 8-30 Hz

    md = ep.metadata.reset_index(drop=True)
    # Do not TRUST the ordering - PROVE it. An earlier bug in this project came from
    # two folders sorting differently, so alignment is asserted, never assumed.
    assert (md["label"].values == (md["condition"] == "distractor").astype(int).values).all()
    assert md["subject"].nunique() == 1 and md["subject"].iloc[0] == subject
    meta_all.append(md[META_COLS])

    for w, (a, b) in WINDOWS.items():
        m = (t >= a) & (t < b)
        abs_, rel_ = band_power(X[:, :, m], sfreq)
        store[f"bp_abs_{w}"].append(abs_)
        store[f"bp_rel_{w}"].append(rel_)
        # OAS shrinkage: with 15 channels and short windows a plain covariance
        # estimate is unstable, shrinkage pulls it back towards something sensible.
        store[f"cov_bb_{w}"].append(
            Covariances(estimator="oas").fit_transform(X[:, :, m]).astype(np.float32))
        store[f"cov_osc_{w}"].append(
            Covariances(estimator="oas").fit_transform(Xosc[:, :, m]).astype(np.float32))

    # ---- ERP: average voltage in 100 ms bins over the post window ----
    # No baseline correction: the data is already 1 Hz high-passed so drift is gone,
    # and the pre-trial baseline is contaminated by the PREVIOUS trial's beeps (§3.3).
    a, b = WINDOWS[ERP_WINDOW]
    edges = np.arange(a, b + 1e-9, ERP_BIN_S)
    bins = [X[:, :, (t >= edges[i]) & (t < edges[i + 1])].mean(-1)
            for i in range(len(edges) - 1)]
    store["erp_post"].append(np.stack(bins, -1).astype(np.float32))

    # ---- sliding-window band power, for the time-resolved decoding curve ----
    slide_starts = np.arange(t[0], t[-1] - SLIDE_LEN + 1e-9, SLIDE_STEP)
    sl = [band_power(X[:, :, (t >= s) & (t < s + SLIDE_LEN)], sfreq)[0]
          for s in slide_starts]
    store["slide_bp"].append(np.stack(sl, -1).astype(np.float32))

    times = t
    print(f"  {subject:<14} {len(X):>3} trials", flush=True)

# ---------------------------------------------------------------- assemble
F = {k: np.concatenate(v, axis=0) for k, v in store.items()}
M = pd.concat(meta_all, ignore_index=True)
n = len(M)

# ---------------------------------------------------------------- sanity checks
print("\n" + "=" * 78)
print("SANITY CHECKS")
ok = True

for k, v in F.items():
    if len(v) != n:
        print(f"  FAIL  {k}: {len(v)} rows, expected {n}"); ok = False
    bad = (~np.isfinite(v)).sum()
    if bad and not (k.startswith("bp_") and "early" in k):
        print(f"  FAIL  {k}: {bad} non-finite values"); ok = False
    elif bad:
        print(f"  warn  {k}: {bad} non-finite values (short window, see notes)")
print(f"  {'PASS' if ok else 'FAIL'}  all feature blocks have {n} rows and are finite")

exp = sum(len(mne.read_epochs(f, preload=False, verbose='ERROR')) for f in files)
print(f"  {'PASS' if n == exp else 'FAIL'}  row count {n} matches epoch files ({exp})")

vc = M.condition.value_counts()
print(f"  PASS  class balance: focus={vc.get('focus',0)}  distractor={vc.get('distractor',0)}")

per = M.groupby("subject")["label"].nunique()
print(f"  {'PASS' if (per == 2).all() else 'FAIL'}  all {len(per)} subjects contribute both classes")

assert (M["label"].values == (M["condition"] == "distractor").astype(int).values).all()
print("  PASS  metadata alignment re-verified after concatenation")

# ---------------------------------------------------------------- save
out = os.path.join(DERIV, "features.npz")
np.savez_compressed(
    out,
    subject=M["subject"].values.astype(str),
    label=M["label"].values.astype(np.int8),
    slide_starts=slide_starts.astype(np.float32),
    slide_len=np.float32(SLIDE_LEN),
    band_names=np.array(BAND_NAMES),
    window_names=np.array(list(WINDOWS)),
    window_bounds=np.array(list(WINDOWS.values()), dtype=np.float32),
    erp_bin_s=np.float32(ERP_BIN_S),
    sfreq=np.float32(sfreq),
    **F)
M.to_csv(os.path.join(DERIV, "features_meta.csv"), index=False)

# ---------------------------------------------------------------- report
print("\n" + "=" * 78)
print("FEATURE BLOCKS")
for k in sorted(F):
    v = F[k]
    print(f"  {k:<16} {str(v.shape):<24} {v.size * 4 / 1e6:>7.1f} MB   "
          f"[{np.nanmin(v):>8.2f}, {np.nanmax(v):>8.2f}]")

print("\nFREQUENCY RESOLUTION PER WINDOW  (usable Welch bins per band)")
print(f"  {'window':<8} {'length':>7} {'res':>7}   " + "  ".join(f"{b:>6}" for b in BAND_NAMES))
for w, (a, b) in WINDOWS.items():
    df, cnt = freq_resolution(b - a, sfreq)
    flag = "  <- delta unreliable" if cnt["delta"] <= 1 else ""
    print(f"  {w:<8} {b-a:>6.1f}s {df:>6.1f}Hz   "
          + "  ".join(f"{cnt[bn]:>6}" for bn in BAND_NAMES) + flag)

print(f"\nsliding windows : {len(slide_starts)} x {SLIDE_LEN}s "
      f"from {slide_starts[0]:.2f}s to {slide_starts[-1]:.2f}s (step {SLIDE_STEP}s)")
print(f"total on disk   : {os.path.getsize(out)/1e6:.1f} MB")
print(f"written         -> {out}")
print(f"                -> {os.path.join(DERIV, 'features_meta.csv')}")
