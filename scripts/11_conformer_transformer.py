"""
STEP 11 - Train and evaluate the Conformer-Transformer, SUBJECT-INDEPENDENT.

WHAT THIS IS
  A sixth deep network - conv tokenizer -> Conformer blocks -> reduction/projection ->
  Transformer blocks -> CLS head - run through EXACTLY the protocol the other five
  already went through, so its number can sit in the same table as theirs and mean the
  same thing. The architecture itself, and why each dimension is what it is, is
  documented in scripts/conformer_transformer.py.

  The question it answers: every other model here either convolves and then attends
  (CTNet, EEG Conformer) or does one without the other (EEGNet, LMDA-Net, EEG
  Transformer). None puts a convolution INSIDE the attention block. This one does.

HOW FAIRNESS IS GUARANTEED (not asserted - reused)
  * SPLITS      folds_for() is IMPORTED from 07_bands.py, not reimplemented. The same
                function that produced every other model's folds produces these, and
                check_split() re-derives and asserts it on every single job.
  * INPUT       derivatives/bands/{full,control}.npy - the exact float32 arrays the
                other five networks trained on. Same 2-5 s crop, same decimation to
                125 Hz, same per-subject z-score. Not recomputed, not re-filtered.
  * TRAINING    dl_models.train_dl(), UNMODIFIED - Adam 1e-3, weight decay 1e-4,
                batch 64, class-weighted CE, 300 epochs, patience 40 on validation
                loss, best-validation weights restored before the test set is touched.
                No scheduler, no warmup, no clipping, no augmentation - because none of
                the other five got any of those either. If you want a schedule, run it
                as a separate labelled row, not as this one.
  * SEEDS       job_seed() from 07_bands.py, which is sha1-based, so a job draws the
                same seed in every process on every machine with no environment set.
  * REPORT      the CSS and the table/curve helpers are imported from 09_transformer.py
                rather than copied, so the two reports cannot drift apart.

WHAT IT DOES NOT TOUCH
  Jobs go to derivatives/conformer_jobs/ - not band_jobs/, not transformer_jobs/ - so
  no existing aggregate() glob sees them and no finished sweep is disturbed. No
  existing result file, figure or checkpoint is overwritten. The only edit outside
  this script and its model module is two lines in dl_models.py (an import and a
  registry entry).

CONDITIONS
  full     2.0-5.0 s, 1-45 Hz   the result
  control  0.0-2.0 s, 1-45 Hz   the guardrail. The two conditions are physically
                                identical in this window, so anything much above
                                AUC 0.50 here means a leak, not a discovery.

OUTPUT
  derivatives/conformer_jobs/*.json            one self-describing file per (cond, fold)
  derivatives/conformer_transformer_report.html  one self-contained rendered report
  derivatives/models/cftf_{cond}_fold{n}.pt    trained weights

USAGE
  python scripts/11_conformer_transformer.py                  # run everything, report
  python scripts/11_conformer_transformer.py --quick          # 1 fold, 3 epochs - smoke
  python scripts/11_conformer_transformer.py --report         # rebuild the report only
  python scripts/11_conformer_transformer.py --conditions full   # skip the guardrail
  python scripts/11_conformer_transformer.py --inspect        # shapes + params, no run
"""
import os, sys, glob, json, time, argparse, importlib.util, subprocess, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(SCRIPTS)
DERIV   = os.path.join(ROOT, "derivatives")
BANDS_D = os.path.join(DERIV, "bands")
MODELS_D = os.path.join(DERIV, "models")
sys.path.insert(0, SCRIPTS)

# Redirectable so a smoke run can drive this exact orchestrator without writing a byte
# into the real derivatives/ that the report scripts read.
JOBS_D = os.environ.get("EEG_CT_JOBS_DIR") or os.path.join(DERIV, "conformer_jobs")
OUT_D  = os.environ.get("EEG_CT_OUT_DIR") or DERIV
for d in (JOBS_D, OUT_D, MODELS_D):
    os.makedirs(d, exist_ok=True)

MODEL      = "Conformer-Transformer"    # key in dl_models.DL_MODELS
SLUG       = "cftf"
N_FOLDS    = 5
CONDITIONS = ["full", "control"]
PRETTY     = {"full": "Main window 2.0-5.0 s", "control": "Control window 0.0-2.0 s"}
REPORT_NAME = "conformer_transformer_report.html"


def _load_by_path(name, filename):
    """Import a digit-prefixed sibling script by path - `import 07_bands` is a SyntaxError."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_B7 = _load_by_path("_b7", "07_bands.py")
folds_for = _B7.folds_for          # THE split function, not a copy of it
SEED = _B7.SEED
WINDOWS = {c: _B7.CONDITIONS[c]["window"] for c in CONDITIONS}


def job_seed(cond, fold):
    """Deterministic per-job seed, from 07_bands.py's sha1 implementation.

    Reused rather than reimplemented for the same reason folds_for() is: two copies of
    a seeding rule is two chances to have a different one.
    """
    return _B7.job_seed(MODEL, cond, fold)


# --------------------------------------------------------------- shared data
def load_condition(cond):
    """The same cached arrays the other five networks trained on. mmap - ~100 MB each."""
    f = os.path.join(BANDS_D, f"{cond}.npy")
    assert os.path.exists(f), f"{f} missing - run 07_bands.py --prepare-only first"
    X = np.load(f, mmap_mode="r")
    subs = np.load(os.path.join(BANDS_D, "subject.npy"), allow_pickle=True).astype(str)
    y = np.load(os.path.join(BANDS_D, "label.npy")).astype(int)
    assert X.shape[0] == len(y) == len(subs), "cached arrays disagree on trial count"
    return X, subs, y


def check_split(subs):
    """Independent re-derivation of the folds + the leakage assertions, every run.

    folds_for() is already the shared function, so this cannot 'fix' a mismatch - it
    exists to fail loudly if 07_bands.py is ever edited in a way that changes the split
    out from under results this script has already written.
    """
    all_subs = np.array(sorted(np.unique(subs)))
    rng = np.random.default_rng(SEED)
    te_ref = [all_subs[ix] for ix in np.array_split(rng.permutation(len(all_subs)), N_FOLDS)]
    rows = []
    for i, (tr, va, te) in enumerate(folds_for(all_subs)):
        assert list(te) == list(te_ref[i]), f"fold {i} test set drifted from 04_train.py's"
        assert not (set(tr) & set(te)), f"fold {i}: train/test subject overlap"
        assert not (set(va) & set(te)), f"fold {i}: val/test subject overlap"
        assert not (set(tr) & set(va)), f"fold {i}: train/val subject overlap"
        assert len(set(tr) | set(va) | set(te)) == len(all_subs), f"fold {i}: lost a subject"
        rows.append(dict(fold=i, n_train_subj=len(tr), n_val_subj=len(va),
                         n_test_subj=len(te), test_subjects=", ".join(map(str, te))))
    return rows


def inspect():
    """Print what will be trained on what, and stop. No model is fitted."""
    from conformer_transformer import ConformerTransformer, SFREQ_DL
    print(f"\n{MODEL}  -  dataset and architecture\n" + "=" * 74)
    for cond in CONDITIONS:
        X, subs, y = load_condition(cond)
        w = WINDOWS[cond]
        sfreq = X.shape[2] / (w[1] - w[0])
        print(f"\n{cond:<8} {tuple(X.shape)} {X.dtype}  {w[0]}-{w[1]} s @ {sfreq:g} Hz"
              f"  ({X.nbytes/1e6:.0f} MB)")
        print(f"         {len(np.unique(subs))} subjects, "
              f"{int((y==0).sum())} focus / {int((y==1).sum())} distractor")
        net = ConformerTransformer(X.shape[1], X.shape[2])
        for name, shp in net.shape_trace(batch=64):
            print(f"           {name:<44} {shp}")
        g = net.param_groups()
        print(f"         params: tokenizer {g['tokenizer']:,} + conformer {g['conformer']:,}"
              f" + reduce {g['reduce']:,} + transformer {g['transformer']:,}"
              f" + head {g['head']:,} = {g['total']:,}")
    X, subs, y = load_condition("full")
    print("\nsubject-independent split (re-derived and asserted):")
    for r in check_split(subs):
        print(f"  fold {r['fold']}: train {r['n_train_subj']:>2} | val {r['n_val_subj']} "
              f"| test {r['n_test_subj']}   test={r['test_subjects']}")
    print("\nseeds (sha1, reproducible with no environment set):")
    for c in CONDITIONS:
        print(f"  {c:<8} " + "  ".join(f"f{f}={job_seed(c, f)}" for f in range(N_FOLDS)))


# ------------------------------------------------------------------- one job
def run_job(cond, fold, quick=False):
    """Train the network on one condition for one fold. Writes one self-describing JSON."""
    import torch
    from dl_models import train_dl
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                                 precision_recall_fscore_support, cohen_kappa_score,
                                 confusion_matrix)

    out_json = os.path.join(JOBS_D, f"{SLUG}__{cond}__f{fold}.json")
    X, subs, y = load_condition(cond)
    check_split(subs)
    all_subs = np.array(sorted(np.unique(subs)))
    tr_s, va_s, te_s = folds_for(all_subs)[fold]
    itr, iva, ite = (np.isin(subs, s) for s in (tr_s, va_s, te_s))

    # belt and braces: the masks that actually index the data must be disjoint too
    assert not (itr & ite).any() and not (iva & ite).any() and not (itr & iva).any()

    kw = dict(max_epochs=3, patience=2) if quick else {}
    t0 = time.time()
    p, hist, net, best_ep, stopped = train_dl(
        MODEL, np.ascontiguousarray(X[itr]), y[itr],
        np.ascontiguousarray(X[iva]), y[iva],
        np.ascontiguousarray(X[ite]),
        tag=f"{MODEL}_{cond}_fold{fold}", seed=job_seed(cond, fold), **kw)
    dt = time.time() - t0

    yte, yh = y[ite], (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, yh, average=None,
                                                    labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yte, yh, labels=[0, 1]).ravel()
    row = dict(
        model=MODEL, condition=cond, fold=fold,
        accuracy=accuracy_score(yte, yh),
        balanced_accuracy=balanced_accuracy_score(yte, yh),
        auc=roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan"),
        precision_focus=pr[0], recall_focus=rc[0], f1_focus=f1[0],
        precision_distr=pr[1], recall_distr=rc[1], f1_distr=f1[1],
        f1_macro=(f1[0] + f1[1]) / 2, kappa=cohen_kappa_score(yte, yh),
        sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(yte)),
        n_train=int(itr.sum()), n_val=int(iva.sum()), n_test=int(ite.sum()),
        train_subjects="|".join(map(str, tr_s)), val_subjects="|".join(map(str, va_s)),
        test_subjects="|".join(map(str, te_s)),
        best_epoch=int(best_ep), stopped_epoch=int(stopped), train_seconds=round(dt, 1),
        seed=int(job_seed(cond, fold)), quick=bool(quick),
        # every architectural fact comes from the fitted model, not from a constant
        # retyped here - a config drift can then never be invisible in the report
        **net.config())

    payload = dict(row=row,
                   preds=dict(subject=list(map(str, subs[ite])),
                              y_true=[int(v) for v in yte],
                              y_prob=[float(v) for v in p]),
                   history=hist)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, out_json)          # atomic: a half-written job never looks finished

    if OUT_D == DERIV and not quick:   # never let a 3-epoch net into models/
        torch.save(net.state_dict(), os.path.join(MODELS_D, f"{SLUG}_{cond}_fold{fold}.pt"))
    print(f"[done] {MODEL} {cond:<8} fold{fold}  acc {row['accuracy']:.3f}  "
          f"auc {row['auc']:.3f}  {dt/60:.1f} min  (ep {stopped}, best {best_ep})", flush=True)


# --------------------------------------------------------------- orchestrate
def all_jobs(conds, n_folds):
    return [(c, f) for c in conds for f in range(n_folds)]


def pending(jobs):
    return [j for j in jobs
            if not os.path.exists(os.path.join(JOBS_D, f"{SLUG}__{j[0]}__f{j[1]}.json"))]


def _todo(conds, n_folds):
    # the result before the guardrail, so a half-finished run still says something
    return sorted(pending(all_jobs(conds, n_folds)), key=lambda j: (j[0] != "full", j[0], j[1]))


def run_inline(conds, threads, n_folds, quick, device):
    """Run every pending job sequentially in THIS process. The mode for a GPU.

    One CUDA context costs a few hundred MB of VRAM and the jobs serialise on the one
    device regardless, so N worker processes on a 4 GB laptop card would spend their
    memory on N copies of the runtime rather than on the work. Resume is unaffected -
    each job writes its JSON the moment it finishes.
    """
    import torch
    from dl_models import resolve_device, device_name
    os.environ["EEG_DEVICE"] = device
    torch.set_num_threads(threads)
    jobs, todo = all_jobs(conds, n_folds), _todo(conds, n_folds)
    print(f"\n{len(jobs)} jobs, {len(jobs)-len(todo)} already done, {len(todo)} to run "
          f"INLINE on {device_name(resolve_device(device))}\n", flush=True)
    t0, failed = time.time(), []
    for k, (c, f) in enumerate(todo, 1):
        try:
            run_job(c, f, quick=quick)
        except Exception as e:                     # one bad job must not lose the rest
            failed.append((c, f, f"{type(e).__name__}: {e}"))
            print(f"  !! FAILED {c} fold{f} - excluded, NOT scored as 0.5 "
                  f"({type(e).__name__}: {str(e)[:120]})", flush=True)
        print(f"  [{k}/{len(todo)}] {(time.time()-t0)/60:.0f} min elapsed", flush=True)
    if failed:
        print(f"\n{len(failed)} job(s) failed and are excluded from every mean: {failed}")
    print(f"\nall jobs finished in {(time.time()-t0)/60:.0f} min")


def orchestrate(conds, workers, threads, n_folds, quick, device="cpu"):
    """Run pending jobs in a pool of subprocesses. The mode for a multi-core CPU."""
    jobs, todo = all_jobs(conds, n_folds), _todo(conds, n_folds)
    print(f"\n{len(jobs)} jobs, {len(jobs)-len(todo)} already done, {len(todo)} to run "
          f"on {workers} workers x {threads} threads (device {device})\n", flush=True)
    if not todo:
        return
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               EEG_TORCH_THREADS=str(threads), EEG_DEVICE=device)
    me = os.path.abspath(__file__)
    running, t0, done = [], time.time(), 0
    while todo or running:
        while todo and len(running) < workers:
            c, f = todo.pop(0)
            cmd = [sys.executable, me, "--job", c, str(f)] + (["--quick"] if quick else [])
            running.append((subprocess.Popen(cmd, env=env), (c, f)))
            print(f"  -> launched {c} fold{f}   ({len(todo)} queued)", flush=True)
        time.sleep(2.0)
        for item in list(running):
            proc, spec = item
            if proc.poll() is not None:
                running.remove(item)
                done += 1
                if proc.returncode != 0:
                    print(f"  !! FAILED (exit {proc.returncode}) {spec} - excluded, "
                          f"NOT scored as 0.5", flush=True)
                print(f"  [{done}] {(time.time()-t0)/60:.0f} min elapsed, "
                      f"{len(todo)+len(running)} left", flush=True)
    print(f"\nall jobs finished in {(time.time()-t0)/60:.0f} min")


# ------------------------------------------------------------------- reading
def read_jobs():
    rows, preds, hists = [], [], []
    for f in sorted(glob.glob(os.path.join(JOBS_D, f"{SLUG}__*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d["row"])
        preds.append(pd.DataFrame(dict(condition=d["row"]["condition"],
                                       fold=d["row"]["fold"], **d["preds"])))
        hists += d["history"]
    if not rows:
        return None, None, None
    return (pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.DataFrame(hists))


def peer_results():
    """The other models' main-window numbers, for the side-by-side table."""
    frames = []
    bf = os.path.join(DERIV, "band_results.csv")
    if os.path.exists(bf):
        b = pd.read_csv(bf)
        frames.append(b[b.condition == "full"][["model", "fold", "accuracy", "auc", "kappa"]])
    rf = os.path.join(DERIV, "results_cv5.csv")
    if os.path.exists(rf):
        r = pd.read_csv(rf)
        r = r[(r.window == "post") & (r.model != "SHUFFLED-control")]
        frames.append(r[["model", "fold", "accuracy", "auc", "kappa"]])
    # the EEG Transformer's own jobs are not in either file - read them where they live
    tj = sorted(glob.glob(os.path.join(DERIV, "transformer_jobs", "*__full__f*.json")))
    if tj:
        rec = []
        for f in tj:
            with open(f) as fh:
                d = json.load(fh)["row"]
            rec.append({k: d[k] for k in ("model", "fold", "accuracy", "auc", "kappa")})
        frames.append(pd.DataFrame(rec))
    if not frames:
        return pd.DataFrame()
    P = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "fold"])
    return P[P.model != MODEL]


def paired_vs_best(P):
    """Per-SUBJECT paired comparison against the strongest existing model.

    Fold-level n=5 has no power worth quoting. Every subject is a test subject in
    exactly one fold, so per-subject AUC gives 23 paired observations instead - which
    is the comparison that can actually answer 'is this better than what we had'.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import wilcoxon

    def per_subject(df):
        out = {}
        for s, g in df.groupby("subject"):
            if g.y_true.nunique() > 1:
                out[s] = roc_auc_score(g.y_true, g.y_prob)
        return out

    mine = per_subject(P[P.condition == "full"])
    best = None
    pf = os.path.join(DERIV, "predictions_cv5.csv")
    if os.path.exists(pf):
        d = pd.read_csv(pf)
        d = d[(d.window == "post") & (d.model == "EEGNet")]
        if len(d):
            best = ("EEGNet", per_subject(d.rename(columns={"y_prob": "y_prob"})))
    if best is None:
        return None
    name, theirs = best
    common = sorted(set(mine) & set(theirs))
    if len(common) < 6:
        return None
    a = np.array([mine[s] for s in common])
    b = np.array([theirs[s] for s in common])
    try:
        stat, pv = wilcoxon(a, b)
    except ValueError:                             # all differences zero
        stat, pv = float("nan"), 1.0
    return dict(peer=name, n_subjects=len(common), mine=float(a.mean()),
                theirs=float(b.mean()), delta=float((a - b).mean()),
                delta_sd=float((a - b).std(ddof=1)),
                n_better=int((a > b).sum()), wilcoxon_p=float(pv))


# -------------------------------------------------------------------- report
# CSS and the table/curve helpers come from 09_transformer.py rather than a copy, so
# the two reports cannot drift apart. Importing it by path also runs its module-level
# setup, which only resolves paths and mkdirs - it trains nothing.
_B9 = _load_by_path("_b9", "09_transformer.py")
CSS, _tbl, _curves_png = _B9.CSS, _B9._tbl, _B9._curves_png


def build_report(R, P, H, quick=False):
    """One self-contained HTML file. Every number is also embedded as JSON so the
    report can be read back programmatically - a rendered page cannot be re-parsed."""
    full, ctrl = R[R.condition == "full"], R[R.condition == "control"]
    # Architecture facts must describe the MAIN window - the control window is 2 s, so
    # it has 23 patches instead of 36 and would silently mis-document the model. Sorted
    # job filenames put 'control' first, so R.iloc[0] is exactly the wrong row.
    meta = (full if len(full) else R).iloc[0]
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"

    cards = []
    if len(full):
        cards += [("Accuracy", f"{full.accuracy.mean():.3f}", f"± {full.accuracy.std():.3f}"),
                  ("ROC-AUC", f"{full.auc.mean():.3f}", f"± {full.auc.std():.3f}"),
                  ("Cohen's κ", f"{full.kappa.mean():.3f}", f"± {full.kappa.std():.3f}")]
    if len(ctrl):
        flag = "at chance ✓" if abs(ctrl.auc.mean() - 0.5) < 0.06 else "CHECK — off chance"
        cards.append(("Control AUC", f"{ctrl.auc.mean():.3f}", flag))
    cards += [("Parameters", f"{int(meta.n_params):,}",
               f"{meta.conf_depth} conformer + {meta.tf_depth} transformer"),
              ("Tokens", f"{int(meta.n_patches)} → {int(meta.n_reduced)}",
               f"{meta.patch_s:.2f} s every {meta.hop_s:.2f} s")]
    cards_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in cards)

    # ---- shape trace, from the real model at the real dimensions ----
    from conformer_transformer import ConformerTransformer
    net = ConformerTransformer(int(meta.n_channels), int(meta.n_times))
    trace = "\n".join(f"  {name:<44} {shp}" for name, shp in net.shape_trace(batch=64))
    g = net.param_groups()
    pg_tbl = _tbl(pd.DataFrame([dict(stage=k, parameters=v) for k, v in g.items()
                                if k != "total"] + [dict(stage="TOTAL", parameters=g["total"])]),
                  fmt={"parameters": lambda v: f"{int(v):,}"}, right=["parameters"])

    # ---- per-fold ----
    cols = ["fold", "accuracy", "balanced_accuracy", "auc", "f1_macro", "kappa",
            "sensitivity", "specificity", "n_test", "best_epoch", "stopped_epoch",
            "seed", "test_subjects"]
    per_fold = full[cols].copy()
    per_fold["test_subjects"] = per_fold.test_subjects.str.replace("|", ", ", regex=False)
    fold_tbl = _tbl(per_fold, fmt={c: f3 for c in cols[1:8]}, right=cols[1:12])

    # ---- comparison with the other models ----
    peers = peer_results()
    comp_html = "<p>No peer results found — run 04_train.py / 07_bands.py first.</p>"
    if len(peers) and len(full):
        mine = full[["fold", "accuracy", "auc", "kappa"]].assign(model=MODEL)
        allm = pd.concat([peers, mine], ignore_index=True)
        gg = (allm.groupby("model")
              .agg(accuracy=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                   auc=("auc", "mean"), auc_sd=("auc", "std"),
                   kappa=("kappa", "mean"), folds=("fold", "nunique"))
              .sort_values("auc", ascending=False).reset_index())
        gg["Accuracy"] = gg.apply(lambda r: f"{r.accuracy:.3f} ± {r.acc_sd:.3f}", axis=1)
        gg["ROC-AUC"] = gg.apply(lambda r: f"{r.auc:.3f} ± {r.auc_sd:.3f}", axis=1)
        gg["Cohen's κ"] = gg.kappa.map(f3)
        show = gg[["model", "Accuracy", "ROC-AUC", "Cohen's κ", "folds"]]
        rows = []
        for _, r in show.iterrows():
            cls = ' class="me"' if r["model"] == MODEL else ""
            rows.append(f"<tr{cls}>" + "".join(
                f'<td class="{"r" if c != "model" else ""}">{r[c]}</td>'
                for c in show.columns) + "</tr>")
        th = "".join(f'<th class="{"r" if c != "model" else ""}">{c}</th>' for c in show.columns)
        comp_html = (f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
                     f'<tbody>{"".join(rows)}</tbody></table></div>')

    # ---- paired per-subject test ----
    pair_html = ""
    pair = paired_vs_best(P) if P is not None and len(P) else None
    if pair:
        verdict = ("no detectable difference" if pair["wilcoxon_p"] > 0.05
                   else ("BETTER" if pair["delta"] > 0 else "WORSE"))
        pair_html = (
            f'<div class="note"><b>Per-subject paired comparison vs. {pair["peer"]}.</b> '
            f'Mean per-subject AUC {pair["mine"]:.3f} vs. {pair["theirs"]:.3f}, '
            f'Δ = {pair["delta"]:+.3f} ± {pair["delta_sd"]:.3f} over '
            f'{pair["n_subjects"]} subjects; better on {pair["n_better"]}/'
            f'{pair["n_subjects"]}. Wilcoxon signed-rank p = {pair["wilcoxon_p"]:.3f} '
            f'— <b>{verdict}</b>. Every subject is a test subject in exactly one fold, '
            f'so these 23 observations are paired on the same trials.</div>')

    # ---- confusion matrix ----
    conf_html = ""
    if len(full):
        s = full[["tn", "fp", "fn", "tp"]].sum()
        tot = int(s.sum())
        conf_html = f"""
<div class="scroll"><table>
<thead><tr><th></th><th class="r">predicted focus</th><th class="r">predicted distractor</th>
<th class="r">correct</th></tr></thead><tbody>
<tr><td><b>true focus</b></td><td class="r">{int(s.tn):,}</td><td class="r">{int(s.fp):,}</td>
<td class="r">{100*s.tn/max(s.tn+s.fp,1):.1f}%</td></tr>
<tr><td><b>true distractor</b></td><td class="r">{int(s.fn):,}</td><td class="r">{int(s.tp):,}</td>
<td class="r">{100*s.tp/max(s.fn+s.tp,1):.1f}%</td></tr>
</tbody></table></div>
<p class="sub">Summed over the {full.fold.nunique()} folds — every one of the {tot:,} trials
appears exactly once, each scored by a model that never saw that person.</p>"""

    # ---- control guardrail ----
    ctrl_html = "<p>Control window not run.</p>"
    if len(ctrl):
        d = abs(ctrl.auc.mean() - 0.5)
        klass, verdict = ("note ok", "At chance, as required — nothing leaks.") if d < 0.06 \
            else ("note warn", "OFF CHANCE — investigate before trusting the main number.")
        ctrl_html = (_tbl(ctrl[["fold", "accuracy", "auc", "kappa", "n_test"]],
                          fmt={c: f3 for c in ("accuracy", "auc", "kappa")})
                     + f'<div class="{klass}"><b>AUC {ctrl.auc.mean():.3f} '
                       f'± {ctrl.auc.std():.3f}.</b> {verdict}</div>')

    # ---- per-subject ----
    subj_html = ""
    if P is not None and len(P):
        from sklearn.metrics import roc_auc_score
        pf = P[P.condition == "full"]
        recs = []
        for s, gsub in pf.groupby("subject"):
            recs.append(dict(subject=s, n=len(gsub),
                             accuracy=((gsub.y_prob >= 0.5).astype(int) == gsub.y_true).mean(),
                             auc=roc_auc_score(gsub.y_true, gsub.y_prob)
                             if gsub.y_true.nunique() > 1 else np.nan))
        S = pd.DataFrame(recs).sort_values("auc", ascending=False)
        above = int((S.auc > 0.5).sum())
        subj_html = (f'<p>{above} / {len(S)} subjects above chance.</p>'
                     + '<div class="scroll">' + _tbl(S, fmt={"accuracy": f3, "auc": f3})
                     + "</div>")

    _, subs, _ = load_condition("full")
    split_tbl = _tbl(pd.DataFrame(check_split(subs)),
                     right=["fold", "n_train_subj", "n_val_subj", "n_test_subj"])

    curves = _curves_png(H) if H is not None and len(H) else None
    curves_html = (f'<img src="data:image/png;base64,{curves}" alt="training curves">'
                   if curves else "")

    payload = dict(model=MODEL, generated=time.strftime("%Y-%m-%d %H:%M"), quick=bool(quick),
                   architecture={k: (int(v) if isinstance(v, (int, np.integer)) else v)
                                 for k, v in net.config().items()},
                   parameters=g, paired=pair,
                   results=json.loads(R.to_json(orient="records")))
    blob = json.dumps(payload, indent=1)

    warn = ('<div class="note warn"><b>SMOKE TEST.</b> Trained for 3 epochs on one fold. '
            'These numbers are meaningless.</div>') if quick else ""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conformer-Transformer — Focus vs. Distractor</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Conformer-Transformer</h1>
<p class="sub">Conv tokenizer → Conformer → reduction → Transformer · subject-independent
5-fold · {PRETTY['full']} · 125 Hz · generated {time.strftime('%Y-%m-%d %H:%M')}</p>
{warn}
<div class="cards">{cards_html}</div>

<h2>Architecture</h2>
<p>Every other network here either convolves and then attends, or does one without the
other. This one puts a <b>depthwise convolution inside every attention block</b>
(Gulati-style Conformer: ½FFN → self-attention → conv → ½FFN), so local shape and global
context are refined together at each depth rather than local structure being fixed up
front.</p>
<p>Tokens are <b>time</b>, never channels — the {meta.n_channels} electrodes are collapsed
by a depthwise spatial convolution in the tokenizer, as in EEGNet. The trial becomes
{meta.n_patches} overlapping patches of {meta.patch_samples} samples
({meta.patch_s:.2f} s) every {meta.hop_samples} ({meta.hop_s:.2f} s), the Conformer stage
runs at that resolution, and a strided reduction hands the Transformer stage
{meta.n_reduced} tokens with a learned absolute positional embedding and a CLS token.</p>
<h3>Tensor shapes — traced through the real model, batch 64</h3>
<pre class="trace">{trace}</pre>
<div class="scroll">{pg_tbl}</div>
<div class="note">{meta.conf_depth} Conformer blocks (d={meta.d_conf}, {meta.heads} heads,
FFN ×2, depthwise kernel {meta.conv_k}) → reduce /{meta.reduce_r} + project
{meta.d_conf}→{meta.d_tf} → {meta.tf_depth} Transformer blocks (d={meta.d_tf},
{meta.heads} heads, MLP ×2) → {meta.pool} readout. Dropout 0.3 / head 0.5 ·
{int(meta.n_params):,} trainable parameters. Deliberately small: on this dataset the
existing results are monotone the wrong way — 573 k params scored 0.771, 1.7 k scored
0.798.</div>

<h2>Result — {PRETTY['full']}</h2>
<div class="scroll">{fold_tbl}</div>

<h3>Training curves (main window)</h3>
{curves_html}

<h2>Comparison with the existing models</h2>
<p>Same folds, same subjects, same 125 Hz input, same normalisation, same optimiser and
early-stopping rule. Read straight from the existing result files.</p>
{comp_html}
{pair_html}

<h2>Confusion matrix</h2>
{conf_html}

<h2>Control window — the leakage guardrail</h2>
<p>0.0–2.0 s, before the notification. The two conditions are <em>physically identical</em>
here, so any real skill in this window would be a bug. Because the readout is a CLS token
rather than a flatten, the control model has the same parameter count as the main one bar
the positional table — so this is a like-for-like comparison.</p>
{ctrl_html}

<h2>Subject-independent split</h2>
<p>Produced by <span class="mono">folds_for()</span> imported from
<span class="mono">07_bands.py</span> — the same function that split the data for every
other model here, not a copy of it. Re-derived and asserted on every job.</p>
<div class="scroll">{split_tbl}</div>
<ul class="chk">
<li><b>✓</b> test subjects never appear in training or validation</li>
<li><b>✓</b> model selection (early stopping) uses validation subjects only</li>
<li><b>✓</b> normalisation is per-subject, label-free, fitted on each person's own trials</li>
<li><b>✓</b> class weights computed from the training labels only</li>
<li><b>✓</b> best-validation weights restored before the test set is touched</li>
<li><b>✓</b> per-job seeds are sha1-derived, so a run is reproducible with no environment set</li>
</ul>

<h2>Per-subject</h2>
{subj_html}

<footer>Every number above is embedded below as JSON — a rendered page cannot be
re-parsed, so the data travels with it.
<script type="application/json" id="report-data">{blob}</script>
</footer>
</div></body></html>"""

    out = os.path.join(OUT_D, REPORT_NAME)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


def read_data(path=None):
    """Read the report's own numbers back out of the rendered HTML."""
    path = path or os.path.join(OUT_D, REPORT_NAME)
    txt = open(path, encoding="utf-8").read()
    marker = '<script type="application/json" id="report-data">'
    a = txt.index(marker) + len(marker)
    return json.loads(txt[a:txt.index("</script>", a)])


def report(conds=None, quick=False):
    R, P, H = read_jobs()
    if R is None:
        print("no finished jobs - nothing to report")
        return
    out = build_report(R, P, H, quick=quick)
    full, ctrl = R[R.condition == "full"], R[R.condition == "control"]
    print("\n" + "=" * 72)
    print(f"{MODEL}  -  subject-independent 5-fold")
    print("=" * 72)
    if len(full):
        for m in ("accuracy", "balanced_accuracy", "auc", "f1_macro", "kappa"):
            print(f"  {m:<20} {full[m].mean():.3f} +- {full[m].std():.3f}")
    if len(ctrl):
        d = abs(ctrl.auc.mean() - 0.5)
        print(f"  {'control auc':<20} {ctrl.auc.mean():.3f}"
              f"{'   <-- CHECK: too far from chance' if d >= 0.06 else '   (at chance, ok)'}")
    pair = paired_vs_best(P) if P is not None and len(P) else None
    if pair:
        print(f"\n  vs {pair['peer']} per-subject AUC: {pair['mine']:.3f} vs "
              f"{pair['theirs']:.3f}, delta {pair['delta']:+.3f} +- {pair['delta_sd']:.3f}, "
              f"better on {pair['n_better']}/{pair['n_subjects']}, "
              f"wilcoxon p = {pair['wilcoxon_p']:.3f}")
    miss = pending(all_jobs(conds or CONDITIONS, N_FOLDS))
    if miss:
        print(f"\nWARNING: {len(miss)} job(s) missing and EXCLUDED from the means above "
              f"(not scored as 0.5): {miss}")
    print(f"\nwritten -> {out}")


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", nargs=2, metavar=("COND", "FOLD"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--device", default="auto",
                    help="cuda | cpu | auto (default: GPU if one is present)")
    ap.add_argument("--inspect", action="store_true",
                    help="print shapes, parameters, split and seeds; train nothing")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="1 fold, 3 epochs - a smoke test, not a result")
    a = ap.parse_args()

    if a.job:
        import torch
        torch.set_num_threads(int(os.environ.get("EEG_TORCH_THREADS", 2)))
        run_job(a.job[0], int(a.job[1]), quick=a.quick)
    elif a.inspect:
        inspect()
    elif a.report:
        report()
    else:
        from dl_models import resolve_device
        dev = resolve_device(a.device).type
        conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
        n_folds = 1 if a.quick else N_FOLDS
        if a.quick:
            print("*** QUICK SMOKE TEST - 1 fold, 3 epochs, results are meaningless ***\n")
        if dev == "cuda":                  # see run_inline - one CUDA context, not N
            run_inline(conds, a.threads, n_folds, a.quick, "cuda")
        else:
            orchestrate(conds, a.workers, a.threads, n_folds, a.quick, dev)
        report(conds, quick=a.quick)
