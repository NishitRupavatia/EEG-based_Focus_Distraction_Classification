"""
STEP 9 - Train and evaluate the standalone EEG Transformer, SUBJECT-INDEPENDENT.

WHAT THIS IS
  A fifth deep network - attention over raw temporal EEG patches, with no convolutional
  front end - run through EXACTLY the protocol the other four already went through, so
  its number can sit in the same table as theirs and mean the same thing.

HOW FAIRNESS IS GUARANTEED (not asserted - reused)
  * SPLITS      folds_for() is IMPORTED from 07_bands.py, not reimplemented. The same
                function that produced EEGNet's / CTNet's / LMDA-Net's / Conformer's
                folds produces these. A re-derivation is asserted against it anyway.
  * INPUT       derivatives/bands/{full,control}.npy - the exact float32 arrays the
                other networks trained on. Same 2-5 s crop, same decimation to 125 Hz,
                same per-subject z-score. Not recomputed, not re-filtered: reused.
  * TRAINING    dl_models.train_dl(), unmodified - Adam 1e-3, weight decay 1e-4,
                batch 64, class-weighted CE, 300 epochs, patience 40 on validation
                loss, best-validation weights restored before the test set is touched.

  Because all three come from the existing code, there is no way for this script to
  quietly train on a different split or a differently normalised input.

WHAT IT DOES NOT TOUCH
  Its jobs go to derivatives/transformer_jobs/, NOT derivatives/band_jobs/, so
  07_bands.py's aggregate() glob never sees them and the finished band sweep is not
  disturbed. No existing results file is overwritten. The four networks in
  dl_models.py are untouched; the only edit there is one registry line.

CONDITIONS
  full     2.0-5.0 s, 1-45 Hz   the result
  control  0.0-2.0 s, 1-45 Hz   the guardrail. The two conditions are physically
                                identical in this window, so anything much above
                                AUC 0.50 here means a leak, not a discovery.

OUTPUT
  derivatives/transformer_jobs/*.json   one self-describing file per (condition, fold)
  derivatives/transformer_report.html   one self-contained rendered report

USAGE
  python scripts/09_transformer.py                    # run everything, then report
  python scripts/09_transformer.py --workers 3        # fewer processes (less RAM)
  python scripts/09_transformer.py --quick            # 1 fold, 3 epochs - smoke test
  python scripts/09_transformer.py --report           # rebuild the report only
  python scripts/09_transformer.py --conditions full  # skip the control window
"""
import os, sys, glob, json, time, base64, argparse, importlib.util, subprocess, warnings
from io import BytesIO

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
JOBS_D = os.environ.get("EEG_TF_JOBS_DIR") or os.path.join(DERIV, "transformer_jobs")
OUT_D  = os.environ.get("EEG_TF_OUT_DIR") or DERIV
for d in (JOBS_D, OUT_D, MODELS_D):
    os.makedirs(d, exist_ok=True)

MODEL      = "EEG Transformer"          # key in dl_models.DL_MODELS
SLUG       = "eegtransformer"
N_FOLDS    = 5
CONDITIONS = ["full", "control"]
PRETTY     = {"full": "Main window 2.0-5.0 s", "control": "Control window 0.0-2.0 s"}


# --------------------------------------------------------------- shared split
def _load_bands_module():
    """Import 07_bands.py by path - the digit prefix makes it unimportable by name."""
    spec = importlib.util.spec_from_file_location("_b7", os.path.join(SCRIPTS, "07_bands.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_B7 = _load_bands_module()
folds_for = _B7.folds_for          # THE split function, not a copy of it
job_seed = _B7.job_seed
SEED = _B7.SEED


def load_condition(cond):
    """The same cached arrays the other four networks trained on. mmap - 100 MB each."""
    f = os.path.join(BANDS_D, f"{cond}.npy")
    assert os.path.exists(f), f"{f} missing - run 07_bands.py --prepare-only first"
    X = np.load(f, mmap_mode="r")
    subs = np.load(os.path.join(BANDS_D, "subject.npy"), allow_pickle=True).astype(str)
    y = np.load(os.path.join(BANDS_D, "label.npy"))
    return X, subs, y


def check_split(subs):
    """Independent re-derivation of the folds + the leakage assertions, every run.

    folds_for() is already the shared function, so this cannot 'fix' a mismatch - it
    exists to fail loudly if 07_bands.py is ever edited in a way that changes the split
    out from under the results this script has already written.
    """
    all_subs = np.array(sorted(np.unique(subs)))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_subs))
    te_ref = [all_subs[ix] for ix in np.array_split(perm, N_FOLDS)]
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


# ------------------------------------------------------------------- one job
def run_job(cond, fold, quick=False):
    """Train the transformer on one condition for one fold. Writes one JSON."""
    import torch
    import dl_models
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
        tag=f"{MODEL}_{cond}_fold{fold}", seed=job_seed(MODEL, cond, fold), **kw)
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
        n_params=int(sum(q.numel() for q in net.parameters() if q.requires_grad)),
        n_channels=int(X.shape[1]), n_times=int(X.shape[2]),
        n_tokens=int(net.n_tokens), n_patches=int(net.n_patches),
        patch_samples=int(net.patch), emb_dim=int(net.pos.shape[-1]),
        depth=len(net.blocks), heads=net.blocks[0].attn.num_heads, quick=bool(quick))

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
    jobs = all_jobs(conds, n_folds)
    todo = sorted(pending(jobs), key=lambda j: (j[0] != "full", j[0], j[1]))
    print(f"\n{len(jobs)} jobs, {len(jobs)-len(todo)} already done, {len(todo)} to run "
          f"INLINE on {device_name(resolve_device(device))}\n", flush=True)
    t0, failed = time.time(), []
    for k, (c, f) in enumerate(todo, 1):
        try:
            run_job(c, f, quick=quick)
        except Exception as e:
            failed.append((c, f, f"{type(e).__name__}: {e}"))
            print(f"  !! FAILED {c} fold{f} - excluded, NOT scored as 0.5 "
                  f"({type(e).__name__}: {str(e)[:120]})", flush=True)
        print(f"  [{k}/{len(todo)}] {(time.time()-t0)/60:.0f} min elapsed", flush=True)
    if failed:
        print(f"\n{len(failed)} job(s) failed and are excluded from every mean: {failed}")
    print(f"\nall jobs finished in {(time.time()-t0)/60:.0f} min")


def orchestrate(conds, workers, threads, n_folds, quick, device="cpu"):
    jobs = all_jobs(conds, n_folds)
    todo = pending(jobs)
    todo.sort(key=lambda j: (j[0] != "full", j[0], j[1]))   # the result before the guardrail
    print(f"\n{len(jobs)} jobs, {len(jobs)-len(todo)} already done, {len(todo)} to run "
          f"on {workers} workers x {threads} threads (device {device})\n", flush=True)
    if not todo:
        return
    # PYTHONHASHSEED pins job_seed(): it hashes a tuple of strings, and Python randomises
    # string hashing per process. Without this the weight init is not reproducible across
    # runs. Set here rather than in 07_bands.py, which has 35 finished jobs to preserve.
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               EEG_TORCH_THREADS=str(threads), PYTHONHASHSEED="0", EEG_DEVICE=device)
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
    return (pd.DataFrame(rows),
            pd.concat(preds, ignore_index=True),
            pd.DataFrame(hists))


def peer_results():
    """The other models' main-window numbers, for the side-by-side table."""
    frames = []
    bf = os.path.join(DERIV, "band_results.csv")
    if os.path.exists(bf):
        b = pd.read_csv(bf)
        b = b[b.condition == "full"][["model", "fold", "accuracy", "auc", "kappa"]]
        frames.append(b)
    rf = os.path.join(DERIV, "results_cv5.csv")
    if os.path.exists(rf):
        r = pd.read_csv(rf)
        r = r[(r.window == "post") & (r.model != "SHUFFLED-control")]
        frames.append(r[["model", "fold", "accuracy", "auc", "kappa"]])
    if not frames:
        return pd.DataFrame()
    P = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "fold"])
    return P[P.model != MODEL]


# -------------------------------------------------------------------- report
def _png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    return base64.b64encode(buf.getvalue()).decode()


def _curves_png(H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = H[H.tag.str.contains("_full_")]
    if not len(H):
        return None
    folds = sorted(H.tag.unique())
    fig, axes = plt.subplots(1, len(folds), figsize=(3.0 * len(folds), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tg in zip(axes, folds):
        h = H[H.tag == tg]
        ax.plot(h.epoch, h.train_loss, color="#2a78d6", lw=1.6, label="train")
        ax.plot(h.epoch, h.val_loss, color="#eb6834", lw=1.6, label="validation")
        be = int(h.best_epoch.iloc[0])
        ax.axvline(be, color="#8a8981", ls="--", lw=1)
        ax.set_title(f"fold {tg.split('fold')[-1]}  (best {be})", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_facecolor("#fcfcfb")
    axes[0].set_ylabel("loss", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    out = _png(fig)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    return out


def _tbl(df, fmt=None, right=None):
    fmt = fmt or {}
    right = set(right or [c for c in df.columns if df[c].dtype.kind in "fi"])
    th = "".join(f'<th class="{"r" if c in right else ""}">{c}</th>' for c in df.columns)
    trs = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            s = fmt[c](v) if c in fmt else ("" if pd.isna(v) else str(v))
            tds.append(f'<td class="{"r" if c in right else ""}">{s}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8981;--surface:#fcfcfb;--card:#fff;
--grid:#e5e4e0;--blue:#2a78d6;--warn:#b8531f;--ok:#1baf7a}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 72px}
h1{font-size:27px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:18px;margin:40px 0 10px;padding-top:18px;border-top:1px solid var(--grid)}
h3{font-size:14px;margin:22px 0 8px;color:var(--ink2);text-transform:uppercase;
letter-spacing:.06em}
p{color:var(--ink2);margin:8px 0}
.sub{color:var(--muted);font-size:13px;margin:0 0 26px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}
.card{flex:1 1 150px;background:var(--card);border:1px solid var(--grid);
border-radius:9px;padding:13px 15px}
.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:23px;font-weight:600;margin-top:3px;letter-spacing:-.02em}
.card .n{font-size:11px;color:var(--muted);margin-top:2px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0 4px}
th,td{padding:7px 11px;border-bottom:1px solid var(--grid);text-align:left;
white-space:nowrap}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
font-weight:600;border-bottom:1.5px solid var(--grid)}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:hover{background:#f6f5f2}
tr.me td{background:#eaf2fd;font-weight:600}
tr.me:hover td{background:#e0ecfc}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
pre.trace{background:var(--card);border:1px solid var(--grid);border-radius:9px;
padding:15px 17px;overflow-x:auto;font-size:12.5px;line-height:1.75;margin:6px 0}
.note{border-left:3px solid var(--blue);background:var(--card);padding:11px 15px;
border-radius:0 8px 8px 0;margin:14px 0;font-size:13.5px;color:var(--ink2)}
.note.warn{border-left-color:var(--warn)}
.note.ok{border-left-color:var(--ok)}
.chk{list-style:none;padding:0;margin:10px 0;font-size:13.5px;color:var(--ink2)}
.chk li{padding:3px 0}
.chk b{color:var(--ok);font-weight:600}
img{max-width:100%;border:1px solid var(--grid);border-radius:9px;background:#fff}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--grid);
color:var(--muted);font-size:12px}
@media (prefers-color-scheme:dark){
:root{--ink:#f2f1ee;--ink2:#b9b7b1;--muted:#8a8981;--surface:#16150f;--card:#1e1d17;
--grid:#33322b}
tbody tr:hover{background:#26251e}
tr.me td{background:#1b2b41}tr.me:hover td{background:#22354f}
img{background:#fcfcfb}}
"""


def build_report(R, P, H, quick=False):
    """One self-contained HTML file. Every number is also embedded as JSON so the
    report can be read back programmatically - a rendered page cannot be re-parsed."""
    full = R[R.condition == "full"]
    ctrl = R[R.condition == "control"]
    # Architecture facts must describe the MAIN window - the control window is 2 s, so
    # it has 10 patches instead of 15 and would silently mis-document the model. Sorted
    # job filenames put 'control' first, so R.iloc[0] is exactly the wrong row.
    meta = (full if len(full) else R).iloc[0]
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    f0 = lambda v: "" if pd.isna(v) else f"{v:,.0f}"

    # ---- header cards ----
    cards = []
    if len(full):
        cards += [("Accuracy", f"{full.accuracy.mean():.3f}", f"± {full.accuracy.std():.3f}"),
                  ("ROC-AUC", f"{full.auc.mean():.3f}", f"± {full.auc.std():.3f}"),
                  ("Cohen's κ", f"{full.kappa.mean():.3f}", f"± {full.kappa.std():.3f}")]
    if len(ctrl):
        flag = "at chance ✓" if abs(ctrl.auc.mean() - 0.5) < 0.06 else "CHECK — off chance"
        cards.append(("Control AUC", f"{ctrl.auc.mean():.3f}", flag))
    cards += [("Parameters", f"{int(meta.n_params):,}", f"{meta.emb_dim}d × {meta.depth} layers"),
              ("Tokens", f"{int(meta.n_tokens)}", f"{meta.n_patches} patches + CLS")]
    cards_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in cards)

    # ---- shape trace, from the real model at the real dimensions ----
    from eeg_transformer import EEGTransformer
    net = EEGTransformer(int(meta.n_channels), int(meta.n_times))
    trace = "\n".join(f"  {name:<40} {shp}"
                      for name, shp in net.shape_trace(int(meta.n_channels),
                                                       int(meta.n_times), batch=64))

    # ---- per-fold ----
    cols = ["fold", "accuracy", "balanced_accuracy", "auc", "f1_macro", "kappa",
            "sensitivity", "specificity", "n_test", "best_epoch", "stopped_epoch",
            "test_subjects"]
    per_fold = full[cols].copy()
    per_fold["test_subjects"] = per_fold.test_subjects.str.replace("|", ", ", regex=False)
    fold_tbl = _tbl(per_fold, fmt={c: f3 for c in cols[1:8]},
                    right=cols[1:11])

    # ---- comparison with the other models ----
    peers = peer_results()
    comp_html = "<p>No peer results found — run 04_train.py / 07_bands.py first.</p>"
    if len(peers) and len(full):
        mine = full[["fold", "accuracy", "auc", "kappa"]].assign(model=MODEL)
        allm = pd.concat([peers, mine], ignore_index=True)
        g = (allm.groupby("model")
             .agg(accuracy=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                  auc=("auc", "mean"), auc_sd=("auc", "std"),
                  kappa=("kappa", "mean"), folds=("fold", "nunique"))
             .sort_values("auc", ascending=False).reset_index())
        g["Accuracy"] = g.apply(lambda r: f"{r.accuracy:.3f} ± {r.acc_sd:.3f}", axis=1)
        g["ROC-AUC"] = g.apply(lambda r: f"{r.auc:.3f} ± {r.auc_sd:.3f}", axis=1)
        g["Cohen's κ"] = g.kappa.map(f3)
        show = g[["model", "Accuracy", "ROC-AUC", "Cohen's κ", "folds"]]
        rows = []
        for _, r in show.iterrows():
            cls = ' class="me"' if r["model"] == MODEL else ""
            rows.append(f"<tr{cls}>" + "".join(
                f'<td class="{"r" if c != "model" else ""}">{r[c]}</td>'
                for c in show.columns) + "</tr>")
        th = "".join(f'<th class="{"r" if c != "model" else ""}">{c}</th>'
                     for c in show.columns)
        comp_html = (f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
                     f'<tbody>{"".join(rows)}</tbody></table></div>')

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
        for s, g in pf.groupby("subject"):
            recs.append(dict(subject=s, n=len(g),
                             accuracy=((g.y_prob >= 0.5).astype(int) == g.y_true).mean(),
                             auc=roc_auc_score(g.y_true, g.y_prob)
                             if g.y_true.nunique() > 1 else np.nan))
        S = pd.DataFrame(recs).sort_values("auc", ascending=False)
        above = int((S.auc > 0.5).sum())
        subj_html = (f'<p>{above} / {len(S)} subjects above chance.</p>'
                     + '<div class="scroll">'
                     + _tbl(S, fmt={"accuracy": f3, "auc": f3}) + "</div>")

    # ---- split table ----
    _, subs, _ = load_condition("full")
    split_tbl = _tbl(pd.DataFrame(check_split(subs)), right=["fold", "n_train_subj",
                                                             "n_val_subj", "n_test_subj"])

    curves = _curves_png(H) if H is not None and len(H) else None
    curves_html = (f'<img src="data:image/png;base64,{curves}" alt="training curves">'
                   if curves else "")

    payload = dict(model=MODEL,
                   generated=time.strftime("%Y-%m-%d %H:%M"),
                   quick=bool(quick),
                   architecture=dict(
                       n_channels=int(meta.n_channels), n_times=int(meta.n_times),
                       sfreq=125.0, patch_samples=int(meta.patch_samples),
                       n_patches=int(meta.n_patches), n_tokens=int(meta.n_tokens),
                       emb_dim=int(meta.emb_dim), depth=int(meta.depth),
                       heads=int(meta.heads), n_params=int(meta.n_params),
                       pooling="cls"),
                   results=json.loads(R.to_json(orient="records")))
    blob = json.dumps(payload, indent=1)

    warn = ('<div class="note warn"><b>SMOKE TEST.</b> Trained for 3 epochs on one fold. '
            'These numbers are meaningless.</div>') if quick else ""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG Transformer — Focus vs. Distractor</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>EEG Transformer</h1>
<p class="sub">Standalone temporal-token transformer · subject-independent 5-fold ·
{PRETTY['full']} · 125 Hz · generated {time.strftime('%Y-%m-%d %H:%M')}</p>
{warn}
<div class="cards">{cards_html}</div>

<h2>Architecture</h2>
<p>Attention runs over <b>temporal</b> tokens: the trial is cut into
{meta.patch_samples}-sample ({meta.patch_samples/125:.2f} s) slices, each flattened
across all {meta.n_channels} electrodes and projected to a {meta.emb_dim}-d token.
Channels are never the sequence. A learnable positional embedding is added to the
{meta.n_tokens}-token sequence before the first encoder block, and a CLS token carries
the classification.</p>
<h3>Tensor shapes — traced through the real model, batch 64</h3>
<pre class="trace">{trace}</pre>
<div class="note">{meta.depth} pre-norm encoder blocks · {meta.heads} heads ·
MLP {meta.emb_dim}→{int(meta.emb_dim)*4}→{meta.emb_dim} · dropout 0.5 ·
{int(meta.n_params):,} trainable parameters. No convolution anywhere in the model —
that is the point of the comparison.</div>

<h2>Result — {PRETTY['full']}</h2>
<div class="scroll">{fold_tbl}</div>

<h3>Training curves (main window)</h3>
{curves_html}

<h2>Comparison with the existing models</h2>
<p>Same folds, same subjects, same 125 Hz input, same normalisation, same optimiser
and early-stopping rule. Read straight from the existing result files.</p>
{comp_html}

<h2>Confusion matrix</h2>
{conf_html}

<h2>Control window — the leakage guardrail</h2>
<p>0.0–2.0 s, before the notification. The two conditions are <em>physically identical</em>
here, so any real skill in this window would be a bug.</p>
{ctrl_html}

<h2>Subject-independent split</h2>
<p>Produced by <span class="mono">folds_for()</span> imported from
<span class="mono">07_bands.py</span> — the same function that split the data for EEGNet,
CTNet, LMDA-Net and EEG Conformer, not a copy of it. Re-derived and asserted on every run.</p>
<div class="scroll">{split_tbl}</div>
<ul class="chk">
<li><b>✓</b> test subjects never appear in training or validation</li>
<li><b>✓</b> model selection (early stopping) uses validation subjects only</li>
<li><b>✓</b> normalisation is per-subject, label-free, fitted on each person's own trials</li>
<li><b>✓</b> class weights computed from the training labels only</li>
<li><b>✓</b> best-validation weights restored before the test set is touched</li>
</ul>

<h2>Per-subject</h2>
{subj_html}

<footer>Every number above is embedded below as JSON — a rendered page cannot be
re-parsed, so the data travels with it.
<script type="application/json" id="report-data">{blob}</script>
</footer>
</div></body></html>"""

    out = os.path.join(OUT_D, "transformer_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


def read_data(path=None):
    """Recover the embedded numbers from a rendered report.

    The point of the JSON block: an HTML page is for reading, but a result you cannot
    parse back is a result you cannot check. This makes the report the single artefact
    without making it a dead end.
    """
    path = path or os.path.join(OUT_D, "transformer_report.html")
    txt = open(path, encoding="utf-8").read()
    marker = '<script type="application/json" id="report-data">'
    a = txt.index(marker) + len(marker)
    return json.loads(txt[a:txt.index("</script>", a)])


def report(quick=False):
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
    miss = pending(all_jobs(CONDITIONS, N_FOLDS))
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
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="1 fold, 3 epochs - a smoke test, not a result")
    a = ap.parse_args()

    if a.job:
        import torch
        torch.set_num_threads(int(os.environ.get("EEG_TORCH_THREADS", 2)))
        run_job(a.job[0], int(a.job[1]), quick=a.quick)
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
        report(quick=a.quick)
