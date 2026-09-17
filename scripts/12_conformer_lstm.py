"""
STEP 12 - Train and evaluate the Conformer-LSTM, SUBJECT-INDEPENDENT.

WHAT THIS IS
  A seventh deep network - conv tokenizer -> Conformer block -> BiLSTM -> readout ->
  head - run through EXACTLY the protocol the other six already went through. The
  architecture and the reasoning behind every dimension are in scripts/conformer_lstm.py.

  The question it answers: 11_conformer_transformer.py put a transformer on top of the
  Conformer stage and scored AUC 0.663 with 94 k parameters, overfitting from epoch ~20
  and stalling outright on two folds. This swaps the transformer for a bidirectional
  LSTM at 17.8 k parameters and changes NOTHING else - the tokenizer and the Conformer
  block are the same objects, imported from the same module. So the difference between
  the two numbers is the sequence model and not a re-tuned network.

HOW FAIRNESS IS GUARANTEED (not asserted - reused)
  * SPLITS      folds_for() and check_split() are IMPORTED from 11, which imports
                folds_for from 07_bands.py. One split function, six models deep.
  * INPUT       derivatives/bands/{full,control}.npy via 11's load_condition - the exact
                float32 arrays every other network trained on.
  * TRAINING    dl_models.train_dl(), UNMODIFIED - Adam 1e-3, weight decay 1e-4,
                batch 64, class-weighted CE, 300 epochs, patience 40 on validation
                loss, best-validation weights restored before the test set is touched.
  * LAYERS      ConvTokenizer and ConformerBlock are imported from
                conformer_transformer.py, so the two Conformer-* models cannot drift.
  * SEEDS       07_bands.job_seed(), sha1-based and reproducible with no environment.
  * READOUT     chosen in a validation-only phase before any test trial was scored -
                see conformer_lstm.READOUT. Picking it by test AUC would be the exact
                selection bias that made the previous prediction wrong.
  * REPORT      CSS and the table/curve helpers come from 11 (which takes them from 09),
                so all three reports render identically.

WHAT IT DOES NOT TOUCH
  Jobs go to derivatives/conformer_lstm_jobs/ - not band_jobs/, transformer_jobs/ or
  conformer_jobs/ - so no existing aggregate() glob sees them and no finished sweep is
  disturbed. No existing result file, figure or checkpoint is overwritten.

OUTPUT
  derivatives/conformer_lstm_jobs/*.json      one self-describing file per (cond, fold)
  derivatives/conformer_lstm_report.html      one self-contained rendered report
  derivatives/models/cflstm_{cond}_fold{n}.pt trained weights

USAGE
  python scripts/12_conformer_lstm.py                  # run everything, then report
  python scripts/12_conformer_lstm.py --quick          # 1 fold, 3 epochs - smoke test
  python scripts/12_conformer_lstm.py --report         # rebuild the report only
  python scripts/12_conformer_lstm.py --inspect        # shapes + params + seeds, no run
  python scripts/12_conformer_lstm.py --conditions full   # skip the guardrail
"""
import os, sys, glob, json, time, argparse, importlib.util, subprocess, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(SCRIPTS)
DERIV   = os.path.join(ROOT, "derivatives")
MODELS_D = os.path.join(DERIV, "models")
sys.path.insert(0, SCRIPTS)

JOBS_D = os.environ.get("EEG_CL_JOBS_DIR") or os.path.join(DERIV, "conformer_lstm_jobs")
OUT_D  = os.environ.get("EEG_CL_OUT_DIR") or DERIV
for d in (JOBS_D, OUT_D, MODELS_D):
    os.makedirs(d, exist_ok=True)

MODEL      = "Conformer-LSTM"           # key in dl_models.DL_MODELS
SLUG       = "cflstm"
N_FOLDS    = 5
CONDITIONS = ["full", "control"]
PRETTY     = {"full": "Main window 2.0-5.0 s", "control": "Control window 0.0-2.0 s"}
REPORT_NAME = "conformer_lstm_report.html"


def _load_by_path(name, filename):
    """Import a digit-prefixed sibling script by path - `import 11_...` is a SyntaxError."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Everything generic is taken from 11 rather than reimplemented. 11 in turn takes the
# split from 07 and the report chrome from 09, so there is exactly one definition of
# each of these in the repository.
_B11 = _load_by_path("_b11", "11_conformer_transformer.py")
folds_for      = _B11.folds_for
load_condition = _B11.load_condition
check_split    = _B11.check_split
paired_vs_best = _B11.paired_vs_best
CSS, _tbl, _curves_png = _B11.CSS, _B11._tbl, _B11._curves_png
WINDOWS = _B11.WINDOWS
SEED = _B11.SEED


def job_seed(cond, fold):
    """07_bands.py's sha1 seed, keyed on this model's name."""
    return _B11._B7.job_seed(MODEL, cond, fold)


def inspect():
    """Print what will be trained on what, and stop. No model is fitted."""
    from conformer_lstm import ConformerLSTM, SFREQ_DL
    print(f"\n{MODEL}  -  dataset and architecture\n" + "=" * 74)
    for cond in CONDITIONS:
        X, subs, y = load_condition(cond)
        w = WINDOWS[cond]
        print(f"\n{cond:<8} {tuple(X.shape)} {X.dtype}  {w[0]}-{w[1]} s @ "
              f"{X.shape[2] / (w[1] - w[0]):g} Hz  ({X.nbytes/1e6:.0f} MB)")
        print(f"         {len(np.unique(subs))} subjects, "
              f"{int((y==0).sum())} focus / {int((y==1).sum())} distractor")
        net = ConformerLSTM(X.shape[1], X.shape[2])
        for name, shp in net.shape_trace(batch=64):
            print(f"           {name:<44} {shp}")
        g = net.param_groups()
        print(f"         params: tokenizer {g['tokenizer']:,} + conformer {g['conformer']:,}"
              f" + recurrent {g['recurrent']:,} + head {g['head']:,} = {g['total']:,}")
        print(f"         readout '{net.readout}', no positional encoding, "
              f"{net.n_steps} recurrent steps")
    _, subs, _ = load_condition("full")
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
    return sorted(pending(all_jobs(conds, n_folds)), key=lambda j: (j[0] != "full", j[0], j[1]))


def run_inline(conds, threads, n_folds, quick, device):
    """Run every pending job sequentially in THIS process. The mode for a GPU."""
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
    """Every other model's main-window numbers, including the Conformer-Transformer.

    11's version is reused for the four band-sweep models, the classical models and the
    EEG Transformer; the only thing added here is 11's OWN jobs, which that function
    excludes by construction because it excludes its own MODEL.
    """
    frames = [_B11.peer_results()]
    cj = sorted(glob.glob(os.path.join(DERIV, "conformer_jobs", "*__full__f*.json")))
    if cj:
        rec = []
        for f in cj:
            with open(f) as fh:
                d = json.load(fh)["row"]
            rec.append({k: d[k] for k in ("model", "fold", "accuracy", "auc", "kappa")})
        frames.append(pd.DataFrame(rec))
    P = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return P[P.model != MODEL].drop_duplicates(["model", "fold"])


# -------------------------------------------------------------------- report
def build_report(R, P, H, quick=False):
    """One self-contained HTML file, with every number also embedded as JSON."""
    full, ctrl = R[R.condition == "full"], R[R.condition == "control"]
    # Architecture facts must describe the MAIN window: the control window is 2 s, so it
    # has 23 tokens instead of 36. Sorted filenames put 'control' first, so R.iloc[0] is
    # exactly the wrong row.
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
               f"{meta.conf_depth} conformer + Bi{str(meta.rnn).upper()} {meta.hidden}"),
              ("Recurrent steps", f"{int(meta.n_steps)}",
               f"{meta.patch_s:.2f} s every {meta.hop_s:.2f} s")]
    cards_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in cards)

    from conformer_lstm import ConformerLSTM
    net = ConformerLSTM(int(meta.n_channels), int(meta.n_times))
    trace = "\n".join(f"  {name:<44} {shp}" for name, shp in net.shape_trace(batch=64))
    g = net.param_groups()
    pg_tbl = _tbl(pd.DataFrame([dict(stage=k, parameters=v) for k, v in g.items()
                                if k != "total"] + [dict(stage="TOTAL",
                                                         parameters=g["total"])]),
                  fmt={"parameters": lambda v: f"{int(v):,}"}, right=["parameters"])

    cols = ["fold", "accuracy", "balanced_accuracy", "auc", "f1_macro", "kappa",
            "sensitivity", "specificity", "n_test", "best_epoch", "stopped_epoch",
            "seed", "test_subjects"]
    per_fold = full[cols].copy()
    per_fold["test_subjects"] = per_fold.test_subjects.str.replace("|", ", ", regex=False)
    fold_tbl = _tbl(per_fold, fmt={c: f3 for c in cols[1:8]}, right=cols[1:12])

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
            f'{pair["n_subjects"]}. Wilcoxon signed-rank p = {pair["wilcoxon_p"]:.3g} '
            f'— <b>{verdict}</b>.</div>')

    conf_html = ""
    if len(full):
        s = full[["tn", "fp", "fn", "tp"]].sum()
        conf_html = f"""
<div class="scroll"><table>
<thead><tr><th></th><th class="r">predicted focus</th><th class="r">predicted distractor</th>
<th class="r">correct</th></tr></thead><tbody>
<tr><td><b>true focus</b></td><td class="r">{int(s.tn):,}</td><td class="r">{int(s.fp):,}</td>
<td class="r">{100*s.tn/max(s.tn+s.fp,1):.1f}%</td></tr>
<tr><td><b>true distractor</b></td><td class="r">{int(s.fn):,}</td><td class="r">{int(s.tp):,}</td>
<td class="r">{100*s.tp/max(s.fn+s.tp,1):.1f}%</td></tr>
</tbody></table></div>
<p class="sub">Summed over the {full.fold.nunique()} folds — every one of the
{int(s.sum()):,} trials appears exactly once, each scored by a model that never saw
that person.</p>"""

    ctrl_html = "<p>Control window not run.</p>"
    if len(ctrl):
        d = abs(ctrl.auc.mean() - 0.5)
        klass, verdict = ("note ok", "At chance, as required — nothing leaks.") if d < 0.06 \
            else ("note warn", "OFF CHANCE — investigate before trusting the main number.")
        ctrl_html = (_tbl(ctrl[["fold", "accuracy", "auc", "kappa", "n_test"]],
                          fmt={c: f3 for c in ("accuracy", "auc", "kappa")})
                     + f'<div class="{klass}"><b>AUC {ctrl.auc.mean():.3f} '
                       f'± {ctrl.auc.std():.3f}.</b> {verdict}</div>')

    subj_html = ""
    if P is not None and len(P):
        from sklearn.metrics import roc_auc_score
        pf = P[P.condition == "full"]
        recs = []
        for s, gs in pf.groupby("subject"):
            recs.append(dict(subject=s, n=len(gs),
                             accuracy=((gs.y_prob >= 0.5).astype(int) == gs.y_true).mean(),
                             auc=roc_auc_score(gs.y_true, gs.y_prob)
                             if gs.y_true.nunique() > 1 else np.nan))
        S = pd.DataFrame(recs).sort_values("auc", ascending=False)
        subj_html = (f'<p>{int((S.auc > 0.5).sum())} / {len(S)} subjects above chance.</p>'
                     + '<div class="scroll">' + _tbl(S, fmt={"accuracy": f3, "auc": f3})
                     + "</div>")

    _, subs, _ = load_condition("full")
    split_tbl = _tbl(pd.DataFrame(check_split(subs)),
                     right=["fold", "n_train_subj", "n_val_subj", "n_test_subj"])
    curves = _curves_png(H) if H is not None and len(H) else None
    curves_html = (f'<img src="data:image/png;base64,{curves}" alt="training curves">'
                   if curves else "")

    payload = dict(model=MODEL, generated=time.strftime("%Y-%m-%d %H:%M"), quick=bool(quick),
                   architecture={k: (int(v) if isinstance(v, (bool, int, np.integer)) else v)
                                 for k, v in net.config().items()},
                   parameters=g, paired=pair,
                   results=json.loads(R.to_json(orient="records")))
    blob = json.dumps(payload, indent=1)
    warn = ('<div class="note warn"><b>SMOKE TEST.</b> Trained for 3 epochs on one fold. '
            'These numbers are meaningless.</div>') if quick else ""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conformer-LSTM — Focus vs. Distractor</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Conformer-LSTM</h1>
<p class="sub">Conv tokenizer → Conformer → BiLSTM → {meta.readout} readout ·
subject-independent 5-fold · {PRETTY['full']} · 125 Hz ·
generated {time.strftime('%Y-%m-%d %H:%M')}</p>
{warn}
<div class="cards">{cards_html}</div>

<h2>Architecture</h2>
<p>The tokenizer and the Conformer block are the <b>same objects</b> as in
<span class="mono">11_conformer_transformer.py</span>, imported from the same module.
The only change is what consumes the token sequence: a bidirectional
{str(meta.rnn).upper()} instead of a reduction stage plus transformer blocks. So the gap
between the two models' numbers is the sequence model and not a re-tuned network.</p>
<p>Three consequences. Attention costs ~4d² per block whatever the sequence length, so
the sibling spent 17,088 parameters attending over 13 tokens where a Bi{str(meta.rnn).upper()}
does the job for {int(g['recurrent']):,} and scales with <em>length</em>, which is
{int(meta.n_steps)} here. Recurrence is order-aware by construction, so there is
<b>no positional encoding at all</b> — which is why the parameter count is
<em>identical</em> for the 375-sample main window and the 250-sample control window.
And classifying a fixed-length, time-locked window is an integrate-in-order problem,
which is the prior recurrence encodes directly.</p>
<h3>Tensor shapes — traced through the real model, batch 64</h3>
<pre class="trace">{trace}</pre>
<div class="scroll">{pg_tbl}</div>
<div class="note">{meta.conf_depth} Conformer block (d={meta.dim}, {meta.heads} heads,
FFN ×1, depthwise kernel {meta.conv_k}) → Bi{str(meta.rnn).upper()}({meta.dim}→{meta.hidden},
{meta.lstm_layers} layer) → <b>{meta.readout}</b> readout · dropout 0.3 / head 0.5 ·
{int(meta.n_params):,} trainable parameters. The readout was chosen on mean validation
loss across all five folds <em>before any test trial was scored</em>. Recurrent weights
are orthogonally initialised with the forget-gate bias at 1.0, so the cell starts
willing to remember — the sibling model had two folds never leave chance.</div>

<h2>Result — {PRETTY['full']}</h2>
<div class="scroll">{fold_tbl}</div>

<h3>Training curves (main window)</h3>
{curves_html}

<h2>Comparison with every other model</h2>
<p>Same folds, same subjects, same 125 Hz input, same normalisation, same optimiser and
early-stopping rule. Read straight from the existing result files.</p>
{comp_html}
{pair_html}

<h2>Confusion matrix</h2>
{conf_html}

<h2>Control window — the leakage guardrail</h2>
<p>0.0–2.0 s, before the notification. The two conditions are <em>physically identical</em>
here, so any real skill in this window would be a bug. Nothing in this model depends on
the window length, so the control model is parameter-for-parameter the same size.</p>
{ctrl_html}

<h2>Subject-independent split</h2>
<p>Produced by <span class="mono">folds_for()</span>, imported through
<span class="mono">11_conformer_transformer.py</span> from
<span class="mono">07_bands.py</span> — one split function for every model in this
repository. Re-derived and asserted on every job.</p>
<div class="scroll">{split_tbl}</div>
<ul class="chk">
<li><b>✓</b> test subjects never appear in training or validation</li>
<li><b>✓</b> model selection (early stopping) uses validation subjects only</li>
<li><b>✓</b> the readout hyperparameter was chosen on validation loss, not test AUC</li>
<li><b>✓</b> normalisation is per-subject, label-free, fitted on each person's own trials</li>
<li><b>✓</b> class weights computed from the training labels only</li>
<li><b>✓</b> best-validation weights restored before the test set is touched</li>
<li><b>✓</b> per-job seeds are sha1-derived and reproducible with no environment set</li>
</ul>
<div class="note warn">cuDNN's recurrent kernels are nondeterministic, so a GPU run of
this model is not bit-for-bit repeatable even with the seed pinned. The split, the
initial weights and the batch composition are.</div>

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
              f"wilcoxon p = {pair['wilcoxon_p']:.3g}")
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
        if dev == "cuda":                  # one CUDA context, not N
            run_inline(conds, a.threads, n_folds, a.quick, "cuda")
        else:
            orchestrate(conds, a.workers, a.threads, n_folds, a.quick, dev)
        report(conds, quick=a.quick)
