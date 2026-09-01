"""
STEP 7 - BAND-RESOLVED sweep: three deep networks x six frequency bands.

WHAT THIS ANSWERS
  04_train.py showed that time-domain models beat frequency-domain ones, which is the
  signature of an evoked response rather than an oscillatory state change. This script
  makes that claim falsifiable band by band: take the SAME network, the SAME folds and
  the SAME protocol, and feed it only delta, only theta, ... only gamma, then the whole
  1-45 Hz signal. If the effect really is a broadband evoked potential, no single narrow
  band should come close to the full band.

THE GRID
  models      EEGNet (Lawhern 2018) | CTNet (Zhao 2024) | LMDA-Net (Miao 2023)
  conditions  delta 1-4 | theta 4-8 | alpha 8-13 | beta 13-30 | gamma 30-45 |
              full 1-45 | control (full band on the 0-2 s window)
  folds       the same subject-independent 5-fold split as 04_train.py

  'control' is the guardrail, not a result: it is the 0.0-2.0 s window where the two
  conditions are physically identical, so every model must land at ~0.50 there.

HOW IT RUNS
  Each (model, condition, fold) is one independent job written to its own file in
  derivatives/band_jobs/. That buys three things on a laptop CPU:
    * PARALLELISM - measured on this machine, one training run is no faster on 12
      torch threads than on 2, so N processes at 2 threads each is a near-linear win.
    * RESUME      - a finished job is never recomputed; kill this script and restart it.
    * ISOLATION   - one architecture blowing up does not lose the other 94 jobs.

  EEGNet is run for the five narrow bands only. Its full-band and control results
  already exist in results_cv5.csv and are reused verbatim, so the headline numbers in
  the README are not silently rewritten by a re-run with a different RNG stream.

USAGE
  python scripts/07_bands.py                 # prepare, run everything, aggregate
  python scripts/07_bands.py --workers 4     # fewer processes (less RAM)
  python scripts/07_bands.py --prepare-only  # just build derivatives/bands/
  python scripts/07_bands.py --aggregate     # re-aggregate finished jobs, run nothing
"""
import os, sys, glob, json, time, argparse, subprocess, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIV  = os.path.join(ROOT, "derivatives")
BANDS_D = os.path.join(DERIV, "bands")
MODELS_D = os.path.join(DERIV, "models")

# Where finished jobs land, and where the aggregated tables are written. Both are
# redirectable so that a short smoke run can exercise this exact orchestrator without
# writing a single byte into the real derivatives/ that the report scripts read.
JOBS_D = os.environ.get("EEG_JOBS_DIR") or os.path.join(DERIV, "band_jobs")
OUT_D  = os.environ.get("EEG_OUT_DIR") or DERIV
for d in (BANDS_D, JOBS_D, MODELS_D, OUT_D):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------- config
SEED       = 42            # must match 04_train.py or the folds differ
N_FOLDS    = 5
N_VAL_SUBS = 4
DECIM      = 2             # 250 -> 125 Hz, the rate every network here was tuned for

# Same edges as 03_features.py. 'full' is the 1-45 Hz band the pipeline already
# delivers, so no extra filtering is applied to it - it IS the main window.
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma", "full"]

POST_WIN = (2.0, 5.0)      # main analysis window
PRE_WIN  = (0.0, 2.0)      # control - conditions physically identical here

CONDITIONS = {
    **{b: dict(window=POST_WIN, filt=BANDS[b]) for b in BANDS},
    "full":    dict(window=POST_WIN, filt=None),
    "control": dict(window=PRE_WIN,  filt=None),
}

DEEP_MODELS = ["EEGNet", "CTNet", "LMDA-Net", "EEG Conformer"]
# EEGNet's full/control numbers are inherited from 04_train.py (see module docstring).
SKIP = {("EEGNet", "full"), ("EEGNet", "control")}

PRETTY_BAND = {"delta": "Delta", "theta": "Theta", "alpha": "Alpha", "beta": "Beta",
               "gamma": "Gamma", "full": "Full Band", "control": "Control (0-2 s)"}


def job_seed(model, cond, fold):
    """Deterministic per-job seed - parallel order can never change a result."""
    return (abs(hash((model, cond, fold))) % 10_000) + SEED


def folds_for(subs):
    """Rebuilds 04_train.py's split exactly: same rng, same permutation, same order."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(subs))
    te_folds = [subs[ix] for ix in np.array_split(perm, N_FOLDS)]
    out = []
    for i, te in enumerate(te_folds):
        pool = np.array([s for s in subs if s not in te])
        va = pool[np.random.default_rng(SEED + i).permutation(len(pool))[:N_VAL_SUBS]]
        tr = np.array([s for s in pool if s not in va])
        out.append((tr, va, te))
    return out


# ============================================================ stage 1: prepare
def zscore_raw(X, subjects):
    """Per-subject, per-channel standardisation. Unsupervised, so legal on test data."""
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(subjects):
        m = subjects == s
        mu = X[m].mean(axis=(0, 2), keepdims=True)
        sd = X[m].std(axis=(0, 2), keepdims=True)
        out[m] = (X[m] - mu) / np.where(sd < 1e-12, 1.0, sd)
    return out


def prepare(force=False):
    """Band-pass each condition and cache it as a ready-to-train float32 array.

    Filtering happens on the FULL -0.5 to 5.0 s epoch and only then is the analysis
    window cropped out. Filtering the 3 s crop directly would smear its edges - with
    delta that is a ~0.8 s filter kernel eating into the very transient we care about.
    """
    todo = [c for c in CONDITIONS
            if force or not os.path.exists(os.path.join(BANDS_D, f"{c}.npy"))]
    if not todo:
        print("bands/ already complete - nothing to prepare")
        return
    print(f"preparing {len(todo)} condition(s): {', '.join(todo)}", flush=True)

    # Imported only when there is actually filtering to do. The cached bands are plain
    # .npy, so training them back needs torch but not MNE - and on a machine where the
    # two live in different interpreters, requiring MNE here would block the sweep.
    import mne
    mne.set_log_level("ERROR")

    files = sorted(glob.glob(os.path.join(DERIV, "epochs", "*-epo.fif")))
    assert files, "no epochs found - run 02_preprocess.py first"

    acc = {c: [] for c in todo}
    subs, labels = [], []
    for k, f in enumerate(files):
        ep = mne.read_epochs(f, preload=True, verbose="ERROR")
        subs.append(ep.metadata["subject"].values)
        labels.append(ep.metadata["label"].values)
        t = ep.times
        for c in todo:
            w, band = CONDITIONS[c]["window"], CONDITIONS[c]["filt"]
            e = ep.copy().filter(*band, verbose="ERROR") if band else ep
            X = (e.get_data(copy=True) * 1e6).astype(np.float32)
            acc[c].append(X[:, :, (t >= w[0]) & (t < w[1])][:, :, ::DECIM])
            del e
        del ep
        print(f"  [{k+1}/{len(files)}] {os.path.basename(f)}", flush=True)

    subs = np.concatenate(subs)
    labels = np.concatenate(labels).astype(int)

    d = np.load(os.path.join(DERIV, "features.npz"), allow_pickle=True)
    assert (subs == d["subject"]).all(), "epoch order does not match features.npz"
    assert (labels == d["label"].astype(int)).all(), "labels disagree with features.npz"

    np.save(os.path.join(BANDS_D, "subject.npy"), subs.astype(str))
    np.save(os.path.join(BANDS_D, "label.npy"), labels)
    for c in todo:
        X = zscore_raw(np.concatenate(acc[c]), subs)
        np.save(os.path.join(BANDS_D, f"{c}.npy"), X)
        print(f"  {c:<8} {X.shape}  {X.nbytes/1e6:.0f} MB", flush=True)
        del acc[c], X
    print("prepare done")


def load_condition(cond):
    X = np.load(os.path.join(BANDS_D, f"{cond}.npy"), mmap_mode="r")
    subs = np.load(os.path.join(BANDS_D, "subject.npy"), allow_pickle=True)
    y = np.load(os.path.join(BANDS_D, "label.npy"))
    return X, subs, y


# ============================================================== stage 2: one job
def run_job(model, cond, fold):
    """Train one network on one band for one fold. Writes a self-describing JSON."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from dl_models import train_dl
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                                 precision_recall_fscore_support, cohen_kappa_score,
                                 confusion_matrix)

    out_json = os.path.join(JOBS_D, f"{model}__{cond}__f{fold}.json")
    X, subs, y = load_condition(cond)
    all_subs = np.array(sorted(np.unique(subs)))
    tr_s, va_s, te_s = folds_for(all_subs)[fold]
    itr, iva, ite = (np.isin(subs, s) for s in (tr_s, va_s, te_s))

    t0 = time.time()
    p, hist, net, best_ep, stopped = train_dl(
        model, np.ascontiguousarray(X[itr]), y[itr],
        np.ascontiguousarray(X[iva]), y[iva],
        np.ascontiguousarray(X[ite]),
        tag=f"{model}_{cond}_fold{fold}", seed=job_seed(model, cond, fold))
    dt = time.time() - t0

    yte = y[ite]
    yh = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, yh, average=None,
                                                    labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yte, yh, labels=[0, 1]).ravel()
    row = dict(
        model=model, condition=cond, band=cond, fold=fold,
        accuracy=accuracy_score(yte, yh),
        balanced_accuracy=balanced_accuracy_score(yte, yh),
        auc=roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan"),
        precision_focus=pr[0], recall_focus=rc[0], f1_focus=f1[0],
        precision_distr=pr[1], recall_distr=rc[1], f1_distr=f1[1],
        f1_macro=(f1[0] + f1[1]) / 2, kappa=cohen_kappa_score(yte, yh),
        sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(yte)),
        n_train=int(itr.sum()), n_val=int(iva.sum()), n_test=int(ite.sum()),
        test_subjects="|".join(te_s), best_epoch=int(best_ep),
        stopped_epoch=int(stopped), train_seconds=round(dt, 1),
        n_params=int(sum(q.numel() for q in net.parameters() if q.requires_grad)),
        n_times=int(X.shape[2]))

    payload = dict(row=row,
                   preds=dict(subject=list(map(str, subs[ite])),
                              y_true=[int(v) for v in yte],
                              y_prob=[float(v) for v in p]),
                   history=hist)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, out_json)          # atomic: a half-written job never looks finished

    # Keep the trained weights for the main windows - but never from a redirected
    # (smoke) run, whose 3-epoch nets must not sit in models/ pretending to be real.
    if cond in ("full", "control") and OUT_D == DERIV:
        torch.save(net.state_dict(),
                   os.path.join(MODELS_D, f"{model.lower().replace('-','')}_{cond}_fold{fold}.pt"))
    print(f"[done] {model:<9} {cond:<8} fold{fold}  acc {row['accuracy']:.3f}  "
          f"auc {row['auc']:.3f}  {dt/60:.1f} min  (ep {stopped}, best {best_ep})", flush=True)


# ========================================================== stage 3: orchestrate
def all_jobs():
    return [(m, c, f) for m in DEEP_MODELS for c in CONDITIONS
            for f in range(N_FOLDS) if (m, c) not in SKIP]


def pending(jobs):
    return [j for j in jobs
            if not os.path.exists(os.path.join(JOBS_D, f"{j[0]}__{j[1]}__f{j[2]}.json"))]


def orchestrate(workers, threads):
    """Run pending jobs in a pool of subprocesses, heaviest architecture first.

    LMDA-Net is several times the cost of the rest, so it is dispatched first: starting
    the long jobs early is what keeps the pool from ending with one straggler running
    alone.
    """
    jobs = all_jobs()
    todo = pending(jobs)
    # heaviest first, measured on this machine: LMDA-Net ~21 s/epoch, Conformer ~8,
    # the other two ~4. Unknown names sort last rather than raising.
    order = {"LMDA-Net": 0, "EEG Conformer": 1, "CTNet": 2, "EEGNet": 3}
    todo.sort(key=lambda j: (order.get(j[0], 99), j[1] != "full", j[1], j[2]))
    print(f"\n{len(jobs)} jobs total, {len(jobs)-len(todo)} already done, "
          f"{len(todo)} to run on {workers} workers x {threads} threads\n", flush=True)
    if not todo:
        return

    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               EEG_TORCH_THREADS=str(threads))
    me = os.path.abspath(__file__)
    running, t0, done = [], time.time(), 0
    while todo or running:
        while todo and len(running) < workers:
            m, c, f = todo.pop(0)
            proc = subprocess.Popen([sys.executable, me, "--job", m, c, str(f)], env=env)
            running.append((proc, (m, c, f)))
            print(f"  -> launched {m} {c} fold{f}   ({len(todo)} queued)", flush=True)
        time.sleep(2.0)
        for item in list(running):
            proc, spec = item
            if proc.poll() is not None:
                running.remove(item)
                done += 1
                if proc.returncode != 0:
                    print(f"  !! FAILED (exit {proc.returncode}) {spec} - "
                          f"excluded, not scored as 0.5", flush=True)
                el = (time.time() - t0) / 60
                print(f"  [{done}] {el:.0f} min elapsed, {len(todo)+len(running)} left",
                      flush=True)
    print(f"\nall jobs finished in {(time.time()-t0)/60:.0f} min")


# ============================================================ stage 4: aggregate
def inherit_eegnet_rows():
    """EEGNet's full/control rows come from 04_train.py, mapped onto this schema."""
    src = os.path.join(DERIV, "results_cv5.csv")
    psrc = os.path.join(DERIV, "predictions_cv5.csv")
    if not os.path.exists(src):
        return pd.DataFrame(), pd.DataFrame()
    R = pd.read_csv(src)
    R = R[(R.model == "EEGNet")].copy()
    R["condition"] = R.window.map({"post": "full", "pre": "control"})
    R["band"] = R["condition"]
    R["inherited"] = True
    P = pd.DataFrame()
    if os.path.exists(psrc):
        P = pd.read_csv(psrc)
        P = P[P.model == "EEGNet"].copy()
        P["condition"] = P.window.map({"post": "full", "pre": "control"})
    return R.drop(columns=["window"]), P.drop(columns=["window"], errors="ignore")


def merge_main_window(R):
    """Fold CTNet and LMDA-Net into the main-window tables that 05_report.py draws.

    The band sweep's 'full' condition IS 04_train.py's post window - the arrays are
    bit-identical (same crop, same decimation, same per-subject z-score), so the new
    networks slot straight into the existing model comparison. Written to *_all.csv
    rather than edited into results_cv5.csv: the file behind the published README
    numbers stays exactly as 04_train.py left it.
    """
    src = os.path.join(DERIV, "results_cv5.csv")
    if not os.path.exists(src):
        print("no results_cv5.csv - skipping the main-window merge")
        return
    w_of = {"full": "post", "control": "pre"}
    new = [m for m in DEEP_MODELS if m != "EEGNet"]

    base = pd.read_csv(src)
    add = R[R.model.isin(new) & R.condition.isin(w_of) & ~R.inherited].copy()
    add["window"] = add.condition.map(w_of)
    A = pd.concat([base, add.drop(columns=[c for c in ("condition", "band", "inherited",
                                                       "train_seconds", "n_params", "n_times",
                                                       "best_epoch", "stopped_epoch")
                                           if c in add])], ignore_index=True)
    A.to_csv(os.path.join(OUT_D, "results_all.csv"), index=False)

    psrc = os.path.join(DERIV, "predictions_cv5.csv")
    if os.path.exists(psrc):
        bp = pd.read_csv(os.path.join(OUT_D, "band_predictions.csv"))
        bp = bp[bp.model.isin(new) & bp.condition.isin(w_of)].copy()
        bp["window"] = bp.condition.map(w_of)
        pd.concat([pd.read_csv(psrc), bp.drop(columns=["condition"])],
                  ignore_index=True).to_csv(
            os.path.join(OUT_D, "predictions_all.csv"), index=False)

    # training curves for every deep net, on the same tag scheme 05_report.py expects
    hsrc = os.path.join(DERIV, "eegnet_history.csv")
    parts = []
    if os.path.exists(hsrc):
        h = pd.read_csv(hsrc)
        h["model"] = "EEGNet"
        parts.append(h)
    bh = pd.read_csv(os.path.join(OUT_D, "band_history.csv"))
    bh = bh[bh.model.isin(new)].copy()
    bh["cond"] = bh.tag.str.split("_").str[1]
    bh = bh[bh.cond.isin(w_of)]
    bh["tag"] = ("fold" + bh.tag.str.extract(r"fold(\d+)")[0] + "_" + bh.cond.map(w_of))
    parts.append(bh.drop(columns=["cond"]))
    pd.concat(parts, ignore_index=True).to_csv(
        os.path.join(OUT_D, "dl_history_all.csv"), index=False)
    print("\nmerged -> results_all.csv, predictions_all.csv, dl_history_all.csv")


def aggregate():
    rows, preds, hists = [], [], []
    for f in sorted(glob.glob(os.path.join(JOBS_D, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d["row"])
        pr = d["preds"]
        preds.append(pd.DataFrame(dict(model=d["row"]["model"], condition=d["row"]["condition"],
                                       fold=d["row"]["fold"], **pr)))
        hists += d["history"]
    if not rows:
        print("no finished jobs to aggregate")
        return None

    R = pd.DataFrame(rows)
    R["inherited"] = False
    Ri, Pi = inherit_eegnet_rows()
    if len(Ri):
        R = pd.concat([R, Ri], ignore_index=True)
        if len(Pi):
            preds.append(Pi)
    R.to_csv(os.path.join(OUT_D, "band_results.csv"), index=False)
    pd.concat(preds, ignore_index=True).to_csv(
        os.path.join(OUT_D, "band_predictions.csv"), index=False)
    pd.DataFrame(hists).to_csv(os.path.join(OUT_D, "band_history.csv"), index=False)

    # ---- the table the paper wants: accuracy, models x bands ----
    for metric in ("accuracy", "auc"):
        piv = (R[R.condition != "control"]
               .pivot_table(index="model", columns="condition", values=metric, aggfunc="mean")
               .reindex(index=[m for m in DEEP_MODELS if m in set(R.model)],
                        columns=[b for b in BAND_ORDER]))
        piv.to_csv(os.path.join(OUT_D, f"band_table_{metric}.csv"))
        print("\n" + "=" * 78)
        print(f"{metric.upper()}  -  subject-independent 5-fold mean")
        print("=" * 78)
        print(f"{'model':<12}" + "".join(f"{PRETTY_BAND[b]:>12}" for b in piv.columns))
        for m in piv.index:
            print(f"{m:<12}" + "".join(
                f"{100*piv.loc[m,b]:>11.2f}%" if pd.notna(piv.loc[m, b]) else f"{'-':>12}"
                for b in piv.columns))

    ctrl = R[R.condition == "control"]
    if len(ctrl):
        print("\n" + "=" * 78)
        print("CONTROL WINDOW 0-2 s  (conditions identical - everything must be ~0.50)")
        print("=" * 78)
        g = ctrl.groupby("model")[["accuracy", "auc"]].mean()
        for m in g.index:
            flag = "" if abs(g.loc[m, "auc"] - 0.5) < 0.06 else "   <-- CHECK: too far from chance"
            print(f"  {m:<12} acc {g.loc[m,'accuracy']:.3f}   auc {g.loc[m,'auc']:.3f}{flag}")

    merge_main_window(R)

    miss = pending(all_jobs())
    if miss:
        print(f"\nWARNING: {len(miss)} job(s) never produced a result and are EXCLUDED "
              f"from every mean above (not scored as 0.5):")
        for m, c, f in miss[:20]:
            print(f"  {m} {c} fold{f}")
    print("\nwritten -> band_results.csv, band_predictions.csv, band_history.csv, "
          "band_table_accuracy.csv, band_table_auc.csv")
    return R


# ===================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", nargs=3, metavar=("MODEL", "COND", "FOLD"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--force-prepare", action="store_true")
    a = ap.parse_args()

    if a.job:
        import torch
        torch.set_num_threads(int(os.environ.get("EEG_TORCH_THREADS", 2)))
        run_job(a.job[0], a.job[1], int(a.job[2]))
    elif a.aggregate:
        aggregate()
    elif a.prepare_only:
        prepare(force=a.force_prepare)
    else:
        prepare(force=a.force_prepare)
        orchestrate(a.workers, a.threads)
        aggregate()
