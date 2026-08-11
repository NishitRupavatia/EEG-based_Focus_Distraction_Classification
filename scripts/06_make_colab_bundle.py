"""
STEP 6 - Package everything needed to train on Google Colab into one folder.

Design decision, made by measurement (not assumption):
  The bundle stores the FULL cleaned epoch at 250 Hz in float32 - exactly the data
  that produced RESULTS.md. Every other array (band power, covariances, ERP bins,
  sliding-window power) is DERIVED in the notebook, so there is one source of truth
  and nothing can drift out of sync.

  Cheaper options were tested and rejected:
    125 Hz         -> shifts CSP by -0.008 and Xdawn by +0.019
    float16        -> shifts Xdawn by a further +0.013
    int16          -> non-monotonic, unreliable
  (Xdawn turns out to be numerically chaotic on this data regardless - see README -
   but the other models reproduce exactly at float32/250 Hz, so that is what we ship.)
"""
import os, glob, json, shutil, warnings
import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIV  = os.path.join(ROOT, "derivatives")
BUNDLE = os.path.join(ROOT, "colab_bundle")
DATA_D = os.path.join(BUNDLE, "data")
os.makedirs(DATA_D, exist_ok=True)

print("reading cleaned epochs ...")
X, meta = [], []
ch_names = times = None
for f in sorted(glob.glob(os.path.join(DERIV, "epochs", "*-epo.fif"))):
    ep = mne.read_epochs(f, preload=True, verbose="ERROR")
    X.append((ep.get_data(copy=True) * 1e6).astype(np.float32))     # microvolts
    meta.append(ep.metadata)
    if ch_names is None:
        ch_names, times, sfreq = ep.ch_names, ep.times.astype(np.float32), ep.info["sfreq"]
    else:
        assert ep.ch_names == ch_names, "channel order differs between subjects"
X = np.concatenate(X)
M = pd.concat(meta, ignore_index=True)
print(f"  X {X.shape}  {X.nbytes/1e6:.0f} MB in memory   sfreq {sfreq} Hz")

# ---- alignment must be proved, not assumed ----
d = np.load(os.path.join(DERIV, "features.npz"), allow_pickle=True)
assert (M["subject"].values == d["subject"]).all(), "epoch order != features.npz order"
assert (M["label"].values == d["label"]).all(), "labels disagree"
assert (M["label"].values == (M["condition"] == "distractor").astype(int).values).all()
print("  alignment verified against features.npz")

out = os.path.join(DATA_D, "colab_data.npz")
print("writing (compressing ~375 MB, this takes a minute) ...")
np.savez_compressed(
    out,
    X=X,
    times=times,
    sfreq=np.float32(sfreq),
    ch_names=np.array(ch_names),
    subject=M["subject"].values.astype(str),
    label=M["label"].values.astype(np.int8),
    condition=M["condition"].values.astype(str),
    trial=M["trial"].values.astype(np.int16),
    hit=M["hit"].values.astype(bool),
    responded=M["responded"].values.astype(bool),
    rt=M["rt"].values.astype(np.float32),
    ptp_uv=M["ptp_uv"].values.astype(np.float32),
)
M.to_csv(os.path.join(DATA_D, "trials_meta.csv"), index=False)

sz = os.path.getsize(out) / 1e6
print(f"\n  {out}")
print(f"  {sz:.0f} MB on disk ({X.nbytes/1e6:.0f} MB raw -> {100*sz/(X.nbytes/1e6):.0f}%)")

# ---- verify the written file round-trips exactly ----
print("\nverifying the written file ...")
z = np.load(out, allow_pickle=True)
assert np.array_equal(z["X"], X), "X changed on write/read"
assert (z["subject"] == M["subject"].values).all()
assert (z["label"] == M["label"].values).all()
assert list(z["ch_names"]) == ch_names
assert np.allclose(z["times"], times)
print("  PASS  X is bit-identical after round-trip")
print(f"  PASS  {len(z['label'])} trials, "
      f"focus={int((z['label']==0).sum())} distractor={int((z['label']==1).sum())}, "
      f"{len(np.unique(z['subject']))} subjects")
print(f"  PASS  channels {list(z['ch_names'])}")
print(f"  PASS  time axis {z['times'][0]:.2f} to {z['times'][-1]:.2f} s")
