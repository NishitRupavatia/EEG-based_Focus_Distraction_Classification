"""
STEP 10 - Cross-trial Context Transformer, SUBJECT-INDEPENDENT.

WHAT THIS IS
  An EEGNet-style encoder turns each complete 3 s trial into one 64-d embedding, and a
  transformer then attends ACROSS a group ("context") of 64 trials from the same
  person, emitting a Focus/Distractor label for every trial in the group at once.
  There is no positional embedding: the trials are a set, not a sequence.

  See context_transformer.py for why the architecture is shaped this way. This file is
  the protocol around it - splits, preprocessing, training, checkpoints, metrics, and
  the two controls that decide whether the headline number means anything.

WHY IT MIGHT BEAT PER-TRIAL EEGNET
  A subject-independent split hands the model a person it has never seen, whose
  amplitude scale and electrode impedances are unknown. One trial gives it no way to
  tell the effect from the person. Sixty-four trials from that person, seen together,
  let attention estimate and subtract the common part - an unsupervised, test-time,
  per-subject calibration that a per-trial model structurally cannot do.

WHY IT MIGHT BE CHEATING, AND HOW THAT IS MEASURED
  A set model given 64 trials that are roughly half Focus can rank them and split at
  the median, scoring well while learning nothing per trial. Three things test for it:

    context_size=1   the same architecture with a context of one, which removes
                     cross-trial attention entirely. The gap 64 - 1 IS the effect.
    SKEW STRESS      the test set re-scored on contexts deliberately built at 90/10,
                     and at 100/0 where every trial in the context shares one label.
                     A median-splitter is catastrophically wrong there; a genuine
                     per-trial discriminant barely notices.
    random contexts  composition is never pinned at 32/32 - it varies binomially, so
                     "half of them are distractors" is not a reliable prior to exploit.

HOW FAIRNESS WITH THE OTHER MODELS IS GUARANTEED (reused, not asserted)
  SPLITS    folds_for() is IMPORTED from 07_bands.py - the same function that split the
            data for EEGNet, CTNet, LMDA-Net, EEG Conformer and the stage-9 transformer.
            Re-derived and asserted against on every run.
  INPUT     derivatives/bands/{full,control}.npy - the exact float32 arrays the other
            networks trained on. Same crop, same decimation to 125 Hz, same per-subject
            z-score. Not recomputed, not re-filtered: reused. If they are missing they
            are built by calling 07_bands.py's own prepare().
  TRAINING  the shared protocol: Adam 1e-3, weight decay 1e-4, class-weighted CE,
            300 epochs, patience 40 on validation loss, best-validation weights restored
            before the test set is touched, EEGNet's max-norm constraint after each step.
            Batch size is fixed in TRIALS (256), not contexts, so every context size
            takes the same number of optimiser steps over the same number of trials.

CONDITIONS
  full     2.0-5.0 s   the result
  control  0.0-2.0 s   the guardrail. The conditions are physically identical here, so
                       anything much above AUC 0.50 means a leak, not a discovery.

OUTPUT
  derivatives/context_jobs/*.json           one self-describing file per job
  derivatives/models/ctxtf_*.pt             best-validation checkpoint per job
  derivatives/context_transformer_report.html   one self-contained rendered report

USAGE
  python scripts/10_context_transformer.py                 # discover, run all, report
  python scripts/10_context_transformer.py --inspect       # just show what data exists
  python scripts/10_context_transformer.py --quick         # smoke test, meaningless
  python scripts/10_context_transformer.py --report        # rebuild the report only
  python scripts/10_context_transformer.py --context-sizes 1,64
  python scripts/10_context_transformer.py --conditions full
"""
import os, sys, glob, json, time, math, base64, hashlib, argparse, importlib.util
import subprocess, warnings
from io import BytesIO

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPTS  = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(SCRIPTS)
DERIV    = os.path.join(ROOT, "derivatives")
BANDS_D  = os.path.join(DERIV, "bands")
EPOCHS_D = os.path.join(DERIV, "epochs")
sys.path.insert(0, SCRIPTS)

# Redirectable, so a smoke test can drive this exact orchestrator without writing a
# byte into the real derivatives/ that the report reads. MODELS_D hangs off OUT_D for
# the same reason: a shortened run must not drop a 3-epoch checkpoint into the real
# models/ directory under the name a finished job would use.
JOBS_D   = os.environ.get("EEG_CTX_JOBS_DIR") or os.path.join(DERIV, "context_jobs")
OUT_D    = os.environ.get("EEG_CTX_OUT_DIR") or DERIV
MODELS_D = os.path.join(OUT_D, "models")
for d in (JOBS_D, OUT_D, MODELS_D):
    os.makedirs(d, exist_ok=True)

MODEL = "Context Transformer"
SLUG  = "ctxtf"

# The sweep. context_size=1 is the ablation that removes cross-trial attention; 64 is
# the headline. 4 and 16 turn two points into a curve, which is what tells you whether
# the benefit saturates or is still climbing.
CONTEXT_SIZES = [1, 4, 16, 64]
MAIN_CONTEXT  = 64
CONDITIONS    = ["full", "control"]
# The control window is a guardrail, not a result: it only needs the two ends of the
# sweep, which halves its cost.
CONTROL_SIZES = [1, MAIN_CONTEXT]
PRETTY = {"full": "Main window 2.0-5.0 s", "control": "Control window 0.0-2.0 s"}

# Shared optimisation settings - deliberately the same as dl_models.py, read from the
# environment at call time so a smoke test can shorten the schedule without editing code.
MAX_EPOCHS   = int(os.environ.get("EEG_MAX_EPOCHS", 300))
PATIENCE     = int(os.environ.get("EEG_PATIENCE", 40))
TRIAL_BATCH  = 256          # batch measured in TRIALS, so every context size matches
LR           = 1e-3
WEIGHT_DECAY = 1e-4

# Fraction of the context that is Focus, for the skew stress test. 0.5 is the natural
# composition; 1.0 and 0.0 are contexts where every trial shares one label.
SKEW_RATIOS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]
N_STABILITY = 5             # re-score the test set under this many context assignments


# =============================================================== shared split
def _load_bands_module():
    """Import 07_bands.py by path - the digit prefix makes it unimportable by name."""
    spec = importlib.util.spec_from_file_location("_b7", os.path.join(SCRIPTS, "07_bands.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_B7 = _load_bands_module()
folds_for  = _B7.folds_for          # THE split function, not a copy of it
SEED       = _B7.SEED
N_FOLDS    = _B7.N_FOLDS
N_VAL_SUBS = _B7.N_VAL_SUBS
WINDOWS    = {c: _B7.CONDITIONS[c]["window"] for c in _B7.CONDITIONS}


def job_seed(cond, S, fold):
    """Deterministic per-job seed. sha1, not hash() - hash() randomises per process
    unless PYTHONHASHSEED is pinned, and the GPU path runs jobs inline in one process
    where that is easy to forget. This is reproducible with no environment at all."""
    h = hashlib.sha1(f"{SLUG}|{cond}|{S}|{fold}".encode()).digest()
    return SEED + int.from_bytes(h[:4], "big") % 10_000


# ========================================================= dataset discovery
def discover(cond=None, verbose=True, allow_build=True):
    """Inspect what is on disk, work out the format, and build what is missing.

    Nothing about the data layout is hard-coded here beyond the directory names: the
    channel count, trial count, window length and sampling rate are all READ from the
    arrays and the window definitions, then checked against each other. If the cached
    arrays are absent, 07_bands.py's own prepare() is invoked rather than reimplemented.
    """
    info = {"root": ROOT, "bands_dir": BANDS_D, "conditions": {}}
    need = [cond] if cond else CONDITIONS
    meta = [os.path.join(BANDS_D, f) for f in ("subject.npy", "label.npy")]
    missing = [c for c in need if not os.path.exists(os.path.join(BANDS_D, f"{c}.npy"))]
    missing += [] if all(os.path.exists(m) for m in meta) else ["<metadata>"]

    if missing:
        n_epo = len(glob.glob(os.path.join(EPOCHS_D, "*-epo.fif")))
        info["missing"] = missing
        info["n_epoch_files"] = n_epo
        if not n_epo:
            raise SystemExit(
                f"Cannot find {missing} in {BANDS_D}, and no epochs in {EPOCHS_D}.\n"
                f"Run the earlier stages first:\n"
                f"  python scripts/01_build_trials.py\n"
                f"  python scripts/02_preprocess.py")
        if not allow_build:
            raise SystemExit(f"missing {missing} - rerun without --inspect to build it")
        if verbose:
            print(f"cached arrays missing {missing}; building them from {n_epo} epoch "
                  f"files via 07_bands.prepare() ...", flush=True)
        _B7.prepare()

    subs = np.load(os.path.join(BANDS_D, "subject.npy"), allow_pickle=True).astype(str)
    y = np.load(os.path.join(BANDS_D, "label.npy")).astype(int)
    info["n_trials"] = int(len(y))
    info["n_subjects"] = int(len(np.unique(subs)))
    info["subjects"] = sorted(np.unique(subs).tolist())
    info["n_focus"] = int((y == 0).sum())
    info["n_distractor"] = int((y == 1).sum())
    per = pd.Series(subs).value_counts()
    info["trials_per_subject"] = dict(min=int(per.min()), median=float(per.median()),
                                      max=int(per.max()))

    for c in need:
        X = np.load(os.path.join(BANDS_D, f"{c}.npy"), mmap_mode="r")
        assert X.shape[0] == len(y), f"{c}.npy has {X.shape[0]} trials, labels have {len(y)}"
        w = WINDOWS[c]
        sfreq = X.shape[2] / (w[1] - w[0])
        info["conditions"][c] = dict(
            path=os.path.join(BANDS_D, f"{c}.npy"), shape=list(X.shape),
            dtype=str(X.dtype), window_s=list(w), sfreq_hz=round(float(sfreq), 3),
            mb=round(X.nbytes / 1e6, 1))
        del X

    # The normalisation is supposed to be per-subject z-scoring. Verify it rather than
    # trusting the filename: a silently un-normalised array would change every result.
    c0 = need[0]
    X0 = np.load(os.path.join(BANDS_D, f"{c0}.npy"), mmap_mode="r")
    s0 = info["subjects"][0]
    m = subs == s0
    chunk = np.asarray(X0[m])
    info["zscore_check"] = dict(subject=s0,
                                mean=round(float(chunk.mean()), 4),
                                std=round(float(chunk.std()), 4))
    del X0, chunk

    if verbose:
        print(f"\ndataset discovered under {ROOT}")
        print(f"  trials              {info['n_trials']:,} "
              f"({info['n_focus']:,} focus / {info['n_distractor']:,} distractor)")
        print(f"  subjects            {info['n_subjects']}  "
              f"({info['trials_per_subject']['min']}-{info['trials_per_subject']['max']} "
              f"trials each)")
        for c, d in info["conditions"].items():
            print(f"  {c:<8} {str(tuple(d['shape'])):<20} {d['dtype']}  "
                  f"{d['window_s'][0]}-{d['window_s'][1]} s @ {d['sfreq_hz']:g} Hz  "
                  f"{d['mb']:.0f} MB")
        z = info["zscore_check"]
        print(f"  per-subject z-score  '{z['subject']}': mean {z['mean']:+.4f}, "
              f"std {z['std']:.4f}  (expected ~0 / ~1)")
    return info


def load_condition(cond):
    """The same cached arrays every other network in this repo trained on."""
    X = np.load(os.path.join(BANDS_D, f"{cond}.npy"))          # ~100 MB, into RAM
    subs = np.load(os.path.join(BANDS_D, "subject.npy"), allow_pickle=True).astype(str)
    y = np.load(os.path.join(BANDS_D, "label.npy")).astype(int)
    return X, subs, y


def check_split(subs):
    """Independent re-derivation of the folds plus the leakage assertions, every run.

    folds_for() is already the shared function, so this cannot 'fix' a mismatch - it
    exists to fail loudly if 07_bands.py is ever edited in a way that would change the
    split out from under results this script has already written.
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


# ========================================================== context building
def make_contexts(idx, subs, S, rng, shuffle=True):
    """Group trial indices into within-subject contexts of EXACTLY S trials.

    A context never mixes subjects: the whole premise is that attention calibrates to
    one person, and a context spanning two people would be calibrating to an average
    of them.

    Every trial is SCORED exactly once. When a subject's trial count is not a multiple
    of S, the final context is topped up with other trials from that same subject,
    flagged as unscored filler. That is deliberate and not a leak: filler trials
    contribute their EEG but never their label, so a short tail still gets a genuine
    S-trial context instead of a runt of six that would be quietly evaluated under a
    much smaller context than the one being reported. A subject with fewer than S
    trials in total is padded for real and masked out of the attention.

    Returns (ctx (K,S) int64, scored (K,S) bool, valid (K,S) bool).
    """
    C_idx, C_scored, C_valid = [], [], []
    for s in sorted(np.unique(subs[idx])):
        pool = idx[subs[idx] == s]
        if shuffle:
            pool = pool[rng.permutation(len(pool))]
        n = len(pool)
        for k in range(max(1, math.ceil(n / S))):
            chunk = pool[k * S:(k + 1) * S]
            m = len(chunk)
            ci = np.zeros(S, dtype=np.int64)
            cs = np.zeros(S, dtype=bool)
            cv = np.zeros(S, dtype=bool)
            ci[:m], cs[:m], cv[:m] = chunk, True, True
            if m < S:
                others = pool[~np.isin(pool, chunk)]
                need = S - m
                if len(others):
                    fill = rng.choice(others, size=need,
                                      replace=len(others) < need)
                    ci[m:], cv[m:] = fill, True      # context only, never scored
            C_idx.append(ci)
            C_scored.append(cs)
            C_valid.append(cv)
    return np.stack(C_idx), np.stack(C_scored), np.stack(C_valid)


def make_skew_contexts(idx, subs, y, S, ratio, rng):
    """Contexts built to a TARGET class composition, for the shortcut stress test.

    `ratio` is the fraction of the context that is Focus. At 0.5 this is the natural
    composition; at 1.0 every trial in the context is Focus, which is where a model
    that ranks the context and splits at the median must fail - it would be forced to
    call half of a pure-Focus set 'distractor'.

    Coverage: within each subject the two class pools are consumed in order, so every
    trial of a class that has slots is scored exactly once. When one pool runs out
    before the other, the composition is held exact by topping the context up with
    unscored filler drawn from that class. At ratio 1.0 the distractor trials get no
    slots at all and are simply not scored - the returned mask says so.
    """
    n_f = int(round(ratio * S))
    n_d = S - n_f
    C_idx, C_scored, C_valid = [], [], []
    for s in sorted(np.unique(subs[idx])):
        pool = idx[subs[idx] == s]
        F = pool[y[pool] == 0]
        D = pool[y[pool] == 1]
        F = F[rng.permutation(len(F))]
        D = D[rng.permutation(len(D))]
        K = max(math.ceil(len(F) / n_f) if n_f else 0,
                math.ceil(len(D) / n_d) if n_d else 0)
        for k in range(K):
            parts_i, parts_s = [], []
            for want, P in ((n_f, F), (n_d, D)):
                if want == 0:
                    continue
                take = P[k * want:(k + 1) * want]
                parts_i.append(take)
                parts_s.append(np.ones(len(take), bool))
                if len(take) < want and len(P):          # hold composition exact
                    extra = rng.choice(P, size=want - len(take), replace=len(P) < want - len(take))
                    parts_i.append(extra)
                    parts_s.append(np.zeros(len(extra), bool))
            ci = np.concatenate(parts_i)
            cs = np.concatenate(parts_s)
            order = rng.permutation(len(ci))             # do not hand it a sorted set
            ci, cs = ci[order], cs[order]
            cv = np.ones(len(ci), bool)
            if len(ci) < S:                              # subject too small for the ratio
                pad = S - len(ci)
                ci = np.concatenate([ci, np.zeros(pad, np.int64)])
                cs = np.concatenate([cs, np.zeros(pad, bool)])
                cv = np.concatenate([cv, np.zeros(pad, bool)])
            C_idx.append(ci[:S])
            C_scored.append(cs[:S])
            C_valid.append(cv[:S])
    if not C_idx:
        return (np.zeros((0, S), np.int64), np.zeros((0, S), bool), np.zeros((0, S), bool))
    return np.stack(C_idx), np.stack(C_scored), np.stack(C_valid)


# ================================================================== training
def _batched(n, size):
    for a in range(0, n, size):
        yield a, min(a + size, n)


def _epoch(model, X, y, ctx, device, crit, opt=None):
    """One pass over a set of contexts. opt=None means evaluate.

    Loss and accuracy are computed over SCORED slots only, so filler and padding never
    influence the number that early stopping reads.
    """
    import torch
    C_idx, C_scored, C_valid = ctx
    train = opt is not None
    model.train(train)
    S = C_idx.shape[1]
    batch_ctx = max(1, TRIAL_BATCH // S)
    tot = correct = 0
    loss_sum = 0.0
    for a, b in _batched(len(C_idx), batch_ctx):
        ii, ss, vv = C_idx[a:b], C_scored[a:b], C_valid[a:b]
        xb = torch.from_numpy(X[ii]).unsqueeze(2).to(device, non_blocking=True)
        tgt = torch.from_numpy(y[ii]).to(device, non_blocking=True)
        msk = torch.from_numpy(ss).to(device, non_blocking=True)
        pad = torch.from_numpy(~vv).to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(xb, pad_mask=pad)                 # (b, S, 2)
            loss = crit(out[msk], tgt[msk])
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.apply_max_norm()                        # EEGNet's own regulariser
        k = int(msk.sum())
        loss_sum += float(loss) * k
        correct += int((out[msk].argmax(1) == tgt[msk]).sum())
        tot += k
    return loss_sum / max(tot, 1), correct / max(tot, 1)


@np.errstate(all="ignore")
def predict(model, X, y, ctx, device):
    """Score every SCORED slot exactly once. Returns (trial_index, p_distractor)."""
    import torch
    C_idx, C_scored, C_valid = ctx
    model.eval()
    S = C_idx.shape[1]
    batch_ctx = max(1, TRIAL_BATCH // S)
    oi, op = [], []
    with torch.no_grad():
        for a, b in _batched(len(C_idx), batch_ctx):
            ii, ss, vv = C_idx[a:b], C_scored[a:b], C_valid[a:b]
            xb = torch.from_numpy(X[ii]).unsqueeze(2).to(device)
            pad = torch.from_numpy(~vv).to(device)
            p = torch.softmax(model(xb, pad_mask=pad), -1)[..., 1].cpu().numpy()
            oi.append(ii[ss])
            op.append(p[ss])
    return np.concatenate(oi), np.concatenate(op)


def train_context_model(X, y, subs, itr, iva, S, seed, device, ckpt_path=None,
                        max_epochs=None, patience=None):
    """The shared protocol, adapted to contexts. Returns (model, history, best, stop).

    Two details that keep this comparable to the per-trial models:
      * batch size is fixed in TRIALS, so S=1 and S=64 see the same number of trials
        per optimiser step and take the same number of steps per epoch;
      * training contexts are RESAMPLED every epoch (a different partition of each
        subject's trials), which is set-composition augmentation and stops the model
        memorising any particular grouping. Validation contexts are FIXED, so the
        early-stopping signal is a property of the model and not of a fresh shuffle.
    """
    import torch
    import torch.nn as nn
    from context_transformer import ContextTransformer

    max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
    patience = PATIENCE if patience is None else patience
    torch.manual_seed(seed)
    n_ch, n_t = X.shape[1], X.shape[2]
    model = ContextTransformer(n_ch, n_t).to(device)   # built on CPU, then moved

    ytr = y[itr]
    cw = torch.tensor([len(ytr) / (2 * max((ytr == c).sum(), 1)) for c in (0, 1)],
                      dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    va_ctx = make_contexts(np.where(iva)[0], subs, S, np.random.default_rng(seed + 777))
    tr_idx = np.where(itr)[0]

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, max_epochs + 1):
        tr_ctx = make_contexts(tr_idx, subs, S, np.random.default_rng(seed + 10_000 + ep))
        trl, tra = _epoch(model, X, y, tr_ctx, device, crit, opt)
        val, vaa = _epoch(model, X, y, va_ctx, device, crit)
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=val,
                         train_acc=tra, val_acc=vaa))
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt_path:                      # checkpoint the best model, not the last
                torch.save(dict(state_dict=best_state, epoch=ep, val_loss=val,
                                context_size=S, n_ch=n_ch, n_times=n_t, seed=seed),
                           ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    stopped = ep
    best_ep = int(np.argmin([h["val_loss"] for h in hist])) + 1
    for h in hist:
        h["best_epoch"] = best_ep
        h["stopped_epoch"] = stopped
    return model, hist, best_ep, stopped


# =================================================================== metrics
def score(y_true, p):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                                 precision_recall_fscore_support, cohen_kappa_score,
                                 confusion_matrix, average_precision_score, log_loss,
                                 brier_score_loss)
    yh = (p >= 0.5).astype(int)
    both = len(np.unique(y_true)) > 1
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, yh, average=None,
                                                    labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, yh, labels=[0, 1]).ravel()
    return dict(
        accuracy=accuracy_score(y_true, yh),
        balanced_accuracy=balanced_accuracy_score(y_true, yh),
        auc=roc_auc_score(y_true, p) if both else float("nan"),
        average_precision=average_precision_score(y_true, p) if both else float("nan"),
        log_loss=log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        brier=brier_score_loss(y_true, p),
        precision_focus=pr[0], recall_focus=rc[0], f1_focus=f1[0],
        precision_distr=pr[1], recall_distr=rc[1], f1_distr=f1[1],
        f1_macro=(f1[0] + f1[1]) / 2,
        kappa=cohen_kappa_score(y_true, yh) if both else float("nan"),
        mcc=_mcc(tn, fp, fn, tp),
        sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(y_true)))


def _mcc(tn, fp, fn, tp):
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / d) if d > 0 else 0.0


def skew_test(model, X, y, subs, ite, S, device, seed):
    """Re-score the test set on contexts built to fixed class compositions.

    The decisive column is recall on a PURE context (ratio 1.0 and 0.0). A model that
    ranks the context and splits at the median scores ~0.5 there by construction, no
    matter how good the headline AUC looked. A genuine per-trial discriminant is
    roughly flat across the whole row.
    """
    out = []
    idx = np.where(ite)[0]
    for r in SKEW_RATIOS:
        ctx = make_skew_contexts(idx, subs, y, S, r, np.random.default_rng(seed + 31))
        if not len(ctx[0]) or not ctx[1].any():
            continue
        ii, p = predict(model, X, y, ctx, device)
        yt = y[ii]
        yh = (p >= 0.5).astype(int)
        rec_f = float((yh[yt == 0] == 0).mean()) if (yt == 0).any() else float("nan")
        rec_d = float((yh[yt == 1] == 1).mean()) if (yt == 1).any() else float("nan")
        both = len(np.unique(yt)) > 1
        from sklearn.metrics import roc_auc_score
        out.append(dict(focus_fraction=r, n_scored=int(len(ii)),
                        n_focus=int((yt == 0).sum()), n_distr=int((yt == 1).sum()),
                        recall_focus=rec_f, recall_distr=rec_d,
                        balanced_accuracy=float(np.nanmean([rec_f, rec_d])),
                        auc=float(roc_auc_score(yt, p)) if both else float("nan"),
                        mean_p=float(p.mean())))
    return out


def stability_test(model, X, y, subs, ite, S, device, seed, n_rep=N_STABILITY):
    """How much does a trial's answer depend on WHICH neighbours it happened to get?

    Re-scores the whole test set under n_rep independent random context assignments.
    A large spread would mean the headline number is partly an accident of grouping.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    idx = np.where(ite)[0]
    aucs, accs, per_trial = [], [], {}
    for r in range(n_rep):
        ctx = make_contexts(idx, subs, S, np.random.default_rng(seed + 5000 + r))
        ii, p = predict(model, X, y, ctx, device)
        aucs.append(float(roc_auc_score(y[ii], p)) if len(np.unique(y[ii])) > 1 else np.nan)
        accs.append(float(accuracy_score(y[ii], (p >= 0.5).astype(int))))
        for t, v in zip(ii, p):
            per_trial.setdefault(int(t), []).append(float(v))
    spread = [np.std(v) for v in per_trial.values() if len(v) > 1]
    return dict(n_repeats=n_rep, auc_mean=float(np.nanmean(aucs)),
                auc_sd=float(np.nanstd(aucs)), acc_mean=float(np.mean(accs)),
                acc_sd=float(np.std(accs)),
                per_trial_prob_sd=float(np.mean(spread)) if spread else 0.0)


# ================================================================== one job
def job_path(cond, S, fold):
    return os.path.join(JOBS_D, f"{SLUG}__{cond}__s{S}__f{fold}.json")


def run_job(cond, S, fold, quick=False, device=None):
    """Train and evaluate one (condition, context size, fold). Writes one JSON."""
    import torch
    from dl_models import resolve_device
    from context_transformer import ContextTransformer

    device = resolve_device(device)
    X, subs, y = load_condition(cond)
    check_split(subs)
    all_subs = np.array(sorted(np.unique(subs)))
    tr_s, va_s, te_s = folds_for(all_subs)[fold]
    itr, iva, ite = (np.isin(subs, s) for s in (tr_s, va_s, te_s))
    assert not (itr & ite).any() and not (iva & ite).any() and not (itr & iva).any()

    seed = job_seed(cond, S, fold)
    ckpt = os.path.join(MODELS_D, f"{SLUG}_{cond}_s{S}_fold{fold}.pt")
    kw = dict(max_epochs=3, patience=2) if quick else {}

    t0 = time.time()
    model, hist, best_ep, stopped = train_context_model(
        X, y, subs, itr, iva, S, seed, device,
        ckpt_path=(None if quick else ckpt), **kw)
    dt = time.time() - t0

    # Headline: one fixed context assignment, seeded, so the number is reproducible.
    te_ctx = make_contexts(np.where(ite)[0], subs, S, np.random.default_rng(seed + 999))
    ii, p = predict(model, X, y, te_ctx, device)
    assert len(ii) == int(ite.sum()) and len(set(ii.tolist())) == len(ii), \
        "every test trial must be scored exactly once"
    row = dict(model=MODEL, condition=cond, context_size=S, fold=fold, **score(y[ii], p))

    pg = model.param_groups()
    row.update(
        n_params=pg["total"], p_encoder=pg["encoder"], p_attention=pg["attention"],
        n_train=int(itr.sum()), n_val=int(iva.sum()), n_test=int(ite.sum()),
        n_test_contexts=int(len(te_ctx[0])),
        n_filler=int((te_ctx[2] & ~te_ctx[1]).sum()),
        train_subjects="|".join(map(str, tr_s)), val_subjects="|".join(map(str, va_s)),
        test_subjects="|".join(map(str, te_s)),
        best_epoch=int(best_ep), stopped_epoch=int(stopped), train_seconds=round(dt, 1),
        n_channels=int(X.shape[1]), n_times=int(X.shape[2]),
        emb_dim=model.dim, depth=model.depth, heads=model.heads,
        equivariance=model.verify_equivariance(), device=str(device), quick=bool(quick))

    payload = dict(row=row, history=hist,
                   preds=dict(subject=list(map(str, subs[ii])),
                              y_true=[int(v) for v in y[ii]],
                              y_prob=[float(v) for v in p]))
    # The two controls, on the main window only - the control window is already a
    # guardrail and does not need a guardrail of its own.
    if cond == "full" and not quick:
        payload["stability"] = stability_test(model, X, y, subs, ite, S, device, seed)
        if S > 1:
            payload["skew"] = skew_test(model, X, y, subs, ite, S, device, seed)

    tmp = job_path(cond, S, fold) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, job_path(cond, S, fold))    # atomic: a half-written job never counts

    print(f"[done] {cond:<8} S={S:<3} fold{fold}  acc {row['accuracy']:.3f}  "
          f"auc {row['auc']:.3f}  {dt/60:.1f} min  (ep {stopped}, best {best_ep})",
          flush=True)


# =============================================================== orchestration
def sizes_for(cond, sizes):
    return [s for s in sizes if cond == "full" or s in CONTROL_SIZES]


def all_jobs(conds, sizes, n_folds):
    return [(c, s, f) for c in conds for s in sizes_for(c, sizes) for f in range(n_folds)]


def pending(jobs):
    return [j for j in jobs if not os.path.exists(job_path(*j))]


def run_inline(conds, sizes, n_folds, quick, device, threads):
    """Every pending job sequentially in THIS process. The mode for a GPU: one CUDA
    context, not N, and the jobs serialise on the single device regardless."""
    import torch
    from dl_models import resolve_device, device_name
    os.environ["EEG_DEVICE"] = device
    torch.set_num_threads(threads)
    jobs = all_jobs(conds, sizes, n_folds)
    todo = sorted(pending(jobs), key=lambda j: (j[0] != "full", j[0], -j[1], j[2]))
    print(f"\n{len(jobs)} jobs, {len(jobs)-len(todo)} already done, {len(todo)} to run "
          f"INLINE on {device_name(resolve_device(device))}\n", flush=True)
    t0, failed = time.time(), []
    for k, spec in enumerate(todo, 1):
        try:
            run_job(*spec, quick=quick, device=device)
        except Exception as e:
            failed.append((*spec, f"{type(e).__name__}: {e}"))
            print(f"  !! FAILED {spec} - excluded, NOT scored as 0.5 "
                  f"({type(e).__name__}: {str(e)[:140]})", flush=True)
        print(f"  [{k}/{len(todo)}] {(time.time()-t0)/60:.0f} min elapsed", flush=True)
    if failed:
        print(f"\n{len(failed)} job(s) failed and are excluded from every mean: {failed}")
    print(f"\nall jobs finished in {(time.time()-t0)/60:.0f} min")


def orchestrate(conds, sizes, workers, threads, n_folds, quick, device="cpu"):
    jobs = all_jobs(conds, sizes, n_folds)
    todo = sorted(pending(jobs), key=lambda j: (j[0] != "full", j[0], -j[1], j[2]))
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
            c, s, f = todo.pop(0)
            cmd = [sys.executable, me, "--job", c, str(s), str(f)] + \
                  (["--quick"] if quick else [])
            running.append((subprocess.Popen(cmd, env=env), (c, s, f)))
            print(f"  -> launched {c} S={s} fold{f}   ({len(todo)} queued)", flush=True)
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


# =================================================================== reading
def read_jobs():
    rows, preds, hists, skews, stabs = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(JOBS_D, f"{SLUG}__*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        r = d["row"]
        rows.append(r)
        preds.append(pd.DataFrame(dict(condition=r["condition"], context_size=r["context_size"],
                                       fold=r["fold"], **d["preds"])))
        for h in d["history"]:
            hists.append(dict(condition=r["condition"], context_size=r["context_size"],
                              fold=r["fold"], **h))
        for s in d.get("skew", []):
            skews.append(dict(context_size=r["context_size"], fold=r["fold"], **s))
        if "stability" in d:
            stabs.append(dict(context_size=r["context_size"], fold=r["fold"],
                              **d["stability"]))
    if not rows:
        return (None,) * 5
    return (pd.DataFrame(rows), pd.concat(preds, ignore_index=True),
            pd.DataFrame(hists), pd.DataFrame(skews), pd.DataFrame(stabs))


def peer_results():
    """The other models' main-window numbers, read straight from their result files."""
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
    tj = sorted(glob.glob(os.path.join(DERIV, "transformer_jobs", "*__full__*.json")))
    if tj:
        rr = []
        for f in tj:
            with open(f) as fh:
                d = json.load(fh)["row"]
            rr.append(dict(model=d["model"], fold=d["fold"], accuracy=d["accuracy"],
                           auc=d["auc"], kappa=d["kappa"]))
        frames.append(pd.DataFrame(rr))
    if not frames:
        return pd.DataFrame()
    P = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "fold"])
    return P[P.model != MODEL]


# =================================================================== figures
def _png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    return base64.b64encode(buf.getvalue()).decode()


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor("#fcfcfb")
    ax.tick_params(labelsize=8)


def _sweep_png(R):
    """The headline figure: does attending across trials actually help, and by how much."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for ax, metric, label in zip(axes, ("auc", "accuracy"), ("ROC-AUC", "Accuracy")):
        for cond, colour in (("full", "#2a78d6"), ("control", "#b8531f")):
            d = R[R.condition == cond]
            if not len(d):
                continue
            g = d.groupby("context_size")[metric].agg(["mean", "std", "count"]).reset_index()
            g = g.sort_values("context_size")
            ax.errorbar(g.context_size, g["mean"], yerr=g["std"], marker="o", ms=5,
                        lw=1.8, capsize=3, color=colour,
                        label=f"{PRETTY[cond].split(' ')[0].lower()} window")
        ax.axhline(0.5, color="#8a8981", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(R.context_size.unique()))
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("context size (trials attended together)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="best")
        _style(ax)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    out = _png(fig)
    plt.close(fig)
    return out


def _curves_png(H, S):
    plt = _mpl()
    h = H[(H.condition == "full") & (H.context_size == S)]
    if not len(h):
        return None
    folds = sorted(h.fold.unique())
    fig, axes = plt.subplots(1, len(folds), figsize=(3.0 * len(folds), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fd in zip(axes, folds):
        g = h[h.fold == fd]
        ax.plot(g.epoch, g.train_loss, color="#2a78d6", lw=1.6, label="train")
        ax.plot(g.epoch, g.val_loss, color="#eb6834", lw=1.6, label="validation")
        be = int(g.best_epoch.iloc[0])
        ax.axvline(be, color="#8a8981", ls="--", lw=1)
        ax.set_title(f"fold {fd}  (best {be})", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("loss", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    out = _png(fig)
    plt.close(fig)
    return out


def _skew_png(K, S):
    """The shortcut stress test, drawn. Flat lines = a real per-trial discriminant."""
    d = K[K.context_size == S]
    if not len(d):
        return None
    plt = _mpl()
    g = d.groupby("focus_fraction").agg(
        rf=("recall_focus", "mean"), rf_sd=("recall_focus", "std"),
        rd=("recall_distr", "mean"), rd_sd=("recall_distr", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.errorbar(g.focus_fraction, g.rf, yerr=g.rf_sd, marker="o", ms=5, lw=1.8,
                capsize=3, color="#2a78d6", label="recall, Focus trials")
    ax.errorbar(g.focus_fraction, g.rd, yerr=g.rd_sd, marker="s", ms=5, lw=1.8,
                capsize=3, color="#eb6834", label="recall, Distractor trials")
    ax.axhline(0.5, color="#8a8981", ls=":", lw=1)
    ax.annotate("chance", (0.02, 0.505), fontsize=7.5, color="#8a8981")
    ax.axvline(0.5, color="#e5e4e0", lw=8, zorder=0)
    ax.annotate("natural\ncomposition", (0.5, 0.02), fontsize=7.5, color="#8a8981",
                ha="center")
    ax.set_xlabel("fraction of the context that is Focus", fontsize=9)
    ax.set_ylabel("recall", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    ax.set_title(f"context size {S}: does performance survive a skewed context?",
                 fontsize=9.5)
    _style(ax)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    out = _png(fig)
    plt.close(fig)
    return out


# ==================================================================== report
def _tbl(df, fmt=None, right=None, highlight=None):
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
        cls = ' class="me"' if highlight is not None and highlight(r) else ""
        trs.append(f"<tr{cls}>" + "".join(tds) + "</tr>")
    return (f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


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
.chk s{color:var(--warn);font-weight:600;text-decoration:none}
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


def build_report(R, P, H, K, ST, info=None, quick=False):
    """One self-contained HTML file. Every number is embedded as JSON as well, because
    a rendered page cannot be re-parsed and a result you cannot check is not a result."""
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    f0 = lambda v: "" if pd.isna(v) else f"{v:,.0f}"
    full = R[R.condition == "full"]
    ctrl = R[R.condition == "control"]
    sizes = sorted(R.context_size.unique())
    S_main = MAIN_CONTEXT if MAIN_CONTEXT in sizes else max(sizes)
    main = full[full.context_size == S_main]
    solo = full[full.context_size == 1]
    meta = (main if len(main) else full if len(full) else R).iloc[0]

    # ---------------------------------------------------------------- cards
    cards = []
    if len(main):
        cards += [(f"ROC-AUC · ctx {S_main}", f"{main.auc.mean():.3f}",
                   f"± {main.auc.std():.3f} over {main.fold.nunique()} folds"),
                  (f"Accuracy · ctx {S_main}", f"{main.accuracy.mean():.3f}",
                   f"± {main.accuracy.std():.3f}")]
    if len(solo):
        cards.append(("ROC-AUC · ctx 1 (ablation)", f"{solo.auc.mean():.3f}",
                      "cross-trial attention removed"))
    if len(main) and len(solo):
        d = main.auc.mean() - solo.auc.mean()
        cards.append(("Effect of cross-trial attention", f"{d:+.3f}", "AUC, ctx 64 − ctx 1"))
    if len(ctrl):
        cm = ctrl[ctrl.context_size == S_main]
        cm = cm if len(cm) else ctrl
        flag = "at chance ✓" if abs(cm.auc.mean() - 0.5) < 0.06 else "CHECK — off chance"
        cards.append(("Control AUC", f"{cm.auc.mean():.3f}", flag))
    if K is not None and len(K):
        pure = K[(K.context_size == S_main) & (K.focus_fraction == 1.0)]
        if len(pure):
            v = pure.recall_focus.mean()
            cards.append(("Recall on a pure-Focus context", f"{v:.3f}",
                          "0.50 would mean a median split"))
    cards.append(("Parameters", f"{int(meta.n_params):,}",
                  f"{int(meta.p_encoder):,} encoder + {int(meta.p_attention):,} attention"))
    cards_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in cards)

    # --------------------------------------------------------- architecture
    from context_transformer import ContextTransformer
    net = ContextTransformer(int(meta.n_channels), int(meta.n_times))
    trace = "\n".join(f"  {name:<46} {shp}"
                      for name, shp in net.shape_trace(context_size=S_main, batch=2))
    eqv = float(R.equivariance.max()) if "equivariance" in R else net.verify_equivariance()
    eq_ok = eqv < 1e-4
    eq_html = (f'<div class="note {"ok" if eq_ok else "warn"}"><b>Permutation '
               f'equivariance: max |Δ| = {eqv:.1e}.</b> '
               + ("Shuffling the trials in a context permutes the predictions and changes "
                  "nothing else — the model treats the context as a set, which is what "
                  "having no positional embedding is supposed to buy."
                  if eq_ok else
                  "TOO LARGE — the model is sensitive to trial order. Something is acting "
                  "as a positional embedding; the set framing below is not valid.")
               + "</div>")

    # -------------------------------------------------------- the sweep table
    sweep_html = "<p>No results yet.</p>"
    if len(full):
        g = (full.groupby("context_size")
             .agg(accuracy=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                  auc=("auc", "mean"), auc_sd=("auc", "std"),
                  bal=("balanced_accuracy", "mean"), f1=("f1_macro", "mean"),
                  kappa=("kappa", "mean"), mcc=("mcc", "mean"),
                  folds=("fold", "nunique"), secs=("train_seconds", "mean"))
             .reset_index().sort_values("context_size"))
        base = g[g.context_size == 1]
        base_auc = float(base.auc.iloc[0]) if len(base) else np.nan
        show = pd.DataFrame({
            "context size": g.context_size,
            "Accuracy": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.accuracy, g.acc_sd)],
            "ROC-AUC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.auc, g.auc_sd)],
            "Δ AUC vs ctx 1": [("—" if pd.isna(base_auc) or cs == 1 else f"{a - base_auc:+.3f}")
                               for cs, a in zip(g.context_size, g.auc)],
            "Bal. acc": g.bal.map(f3), "F1 macro": g.f1.map(f3),
            "Cohen's κ": g.kappa.map(f3), "MCC": g.mcc.map(f3),
            "folds": g.folds, "min/fold": (g.secs / 60).map(lambda v: f"{v:.1f}")})
        sweep_html = _tbl(show, right=[c for c in show.columns if c != "context size"],
                          highlight=lambda r: r["context size"] == S_main)

    # ------------------------------------------------------------- per fold
    cols = ["fold", "accuracy", "balanced_accuracy", "auc", "average_precision",
            "f1_macro", "kappa", "mcc", "sensitivity", "specificity", "brier",
            "n_test", "n_test_contexts", "best_epoch", "stopped_epoch", "test_subjects"]
    fold_html = "<p>No results yet.</p>"
    if len(main):
        pf = main[cols].copy()
        pf["test_subjects"] = pf.test_subjects.str.replace("|", ", ", regex=False)
        fold_html = _tbl(pf, fmt={c: f3 for c in cols[1:11]}, right=cols[1:15])

    # ------------------------------------------------------- skew stress test
    skew_html = "<p>Skew stress test not run.</p>"
    skew_verdict = ""
    if K is not None and len(K):
        d = K[K.context_size == S_main]
        if len(d):
            g = (d.groupby("focus_fraction")
                 .agg(recall_focus=("recall_focus", "mean"),
                      recall_distr=("recall_distr", "mean"),
                      balanced_accuracy=("balanced_accuracy", "mean"),
                      auc=("auc", "mean"), mean_p=("mean_p", "mean"),
                      n_scored=("n_scored", "mean")).reset_index()
                 .sort_values("focus_fraction", ascending=False))
            g["composition"] = g.focus_fraction.map(
                lambda r: f"{int(round(r*S_main))} focus / {S_main-int(round(r*S_main))} distr")
            show = g[["focus_fraction", "composition", "recall_focus", "recall_distr",
                      "balanced_accuracy", "auc", "mean_p", "n_scored"]]
            skew_html = _tbl(show, fmt={"focus_fraction": lambda v: f"{v:.2f}",
                                        "recall_focus": f3, "recall_distr": f3,
                                        "balanced_accuracy": f3, "auc": f3,
                                        "mean_p": f3, "n_scored": f0},
                             right=[c for c in show.columns if c != "composition"],
                             highlight=lambda r: r["focus_fraction"] == 0.5)
            # The DECISIVE statistic is whether the ranking survives, i.e. whether AUC
            # is flat across compositions. Recall alone conflates two different things:
            # a genuine ranking collapse, and a harmless slide of the 0.5 threshold. A
            # median-splitter's AUC would swing wildly with composition; a real
            # per-trial discriminant's does not move at all.
            au = g.dropna(subset=["auc"])
            nat = g[g.focus_fraction == 0.5]
            pf_ = g[g.focus_fraction == 1.0]
            pd_ = g[g.focus_fraction == 0.0]
            spread = float(au.auc.max() - au.auc.min()) if len(au) > 1 else float("nan")
            lo, hi = (float(au.auc.min()), float(au.auc.max())) if len(au) > 1 else (np.nan,)*2
            base = pure = drop = float("nan")
            if len(nat) and len(pf_) and len(pd_):
                base = float(np.nanmean([nat.recall_focus.iloc[0], nat.recall_distr.iloc[0]]))
                pure = float(np.nanmean([pf_.recall_focus.iloc[0], pd_.recall_distr.iloc[0]]))
                drop = base - pure
            tail = (f' Recall on a context where every trial shares one label is '
                    f'{pure:.3f}, against {base:.3f} at the natural composition — a '
                    f'{drop:.3f} slide of the 0.50 threshold, not of the ranking.'
                    if not np.isnan(pure) else "")
            if not np.isnan(spread) and spread < 0.02:
                skew_verdict = (
                    f'<div class="note ok"><b>Passes — decisively.</b> AUC stays within '
                    f'[{lo:.3f}, {hi:.3f}] — a spread of {spread:.3f} — across every '
                    f'composition from 90/10 to 10/90. A model that ranked the context '
                    f'and split at the median would have its ordering dictated by the '
                    f'composition and could not hold one AUC across all of them. The '
                    f'ordering here is a property of the individual trials.{tail}</div>')
            elif not np.isnan(spread) and spread < 0.05:
                skew_verdict = (
                    f'<div class="note"><b>Mostly holds.</b> AUC moves over '
                    f'[{lo:.3f}, {hi:.3f}] (spread {spread:.3f}) as the composition '
                    f'changes — small, but not nothing. Most of the signal is per-trial; '
                    f'a little of it tracks the set.{tail}</div>')
            else:
                skew_verdict = (
                    f'<div class="note warn"><b>FAILS — the shortcut is real.</b> AUC '
                    f'swings over [{lo:.3f}, {hi:.3f}] (spread {spread:.3f}) with the '
                    f'context composition, which is the signature of a set-level median '
                    f'split rather than a per-trial decision. Read the context-1 '
                    f'ablation as the honest per-trial number.{tail}</div>')

    # -------------------------------------------------------- stability test
    stab_html = ""
    if ST is not None and len(ST):
        s = ST[ST.context_size == S_main]
        if len(s):
            stab_html = (
                f'<p>Re-scoring the whole test set under {int(s.n_repeats.iloc[0])} '
                f'independent random context assignments: AUC '
                f'{s.auc_mean.mean():.3f} ± {s.auc_sd.mean():.3f} within a fold, and an '
                f'individual trial\'s predicted probability moves by '
                f'{s.per_trial_prob_sd.mean():.3f} on average depending on which '
                f'neighbours it was grouped with. The headline table uses one fixed, '
                f'seeded assignment.</p>')

    # ----------------------------------------------------------- comparison
    peers = peer_results()
    comp_html = "<p>No peer results found — run 04_train.py / 07_bands.py first.</p>"
    if len(peers) and len(main):
        mine = [main[["fold", "accuracy", "auc", "kappa"]].assign(
            model=f"{MODEL} (ctx {S_main})")]
        if len(solo):
            mine.append(solo[["fold", "accuracy", "auc", "kappa"]].assign(
                model=f"{MODEL} (ctx 1)"))
        allm = pd.concat([peers] + mine, ignore_index=True)
        g = (allm.groupby("model")
             .agg(accuracy=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                  auc=("auc", "mean"), auc_sd=("auc", "std"),
                  kappa=("kappa", "mean"), folds=("fold", "nunique"))
             .sort_values("auc", ascending=False).reset_index())
        show = pd.DataFrame({
            "model": g.model,
            "Accuracy": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.accuracy, g.acc_sd)],
            "ROC-AUC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(g.auc, g.auc_sd)],
            "Cohen's κ": g.kappa.map(f3), "folds": g.folds})
        comp_html = _tbl(show, right=["Accuracy", "ROC-AUC", "Cohen's κ", "folds"],
                         highlight=lambda r: str(r["model"]).startswith(MODEL))

    # ------------------------------------------------------ confusion matrix
    conf_html = ""
    if len(main):
        s = main[["tn", "fp", "fn", "tp"]].sum()
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
<p class="sub">Summed over the {main.fold.nunique()} folds — every one of the {tot:,}
trials appears exactly once, each scored by a model that never saw that person.</p>"""

    # ---------------------------------------------------------- control window
    ctrl_html = "<p>Control window not run.</p>"
    if len(ctrl):
        c = (ctrl.groupby("context_size")
             .agg(accuracy=("accuracy", "mean"), auc=("auc", "mean"),
                  auc_sd=("auc", "std"), kappa=("kappa", "mean"),
                  folds=("fold", "nunique")).reset_index())
        worst = float(np.nanmax(np.abs(c.auc - 0.5)))
        klass, verdict = (("note ok", "At chance at every context size, as required — "
                           "nothing leaks, and in particular the context itself carries "
                           "no label information.")
                          if worst < 0.06 else
                          ("note warn", "OFF CHANCE — investigate before trusting the "
                           "main number. A context that scores above chance in a window "
                           "where the conditions are physically identical means the "
                           "grouping is leaking, not the EEG."))
        ctrl_html = (_tbl(c, fmt={"accuracy": f3, "auc": f3, "auc_sd": f3, "kappa": f3})
                     + f'<div class="{klass}"><b>Worst |AUC − 0.5| = {worst:.3f}.</b> '
                       f'{verdict}</div>')

    # ------------------------------------------------------------ per subject
    subj_html = ""
    if P is not None and len(P):
        from sklearn.metrics import roc_auc_score
        recs = []
        for S_ in (S_main, 1):
            pf = P[(P.condition == "full") & (P.context_size == S_)]
            for s_, g_ in pf.groupby("subject"):
                recs.append(dict(subject=s_, context_size=S_, n=len(g_),
                                 accuracy=((g_.y_prob >= 0.5).astype(int) == g_.y_true).mean(),
                                 auc=roc_auc_score(g_.y_true, g_.y_prob)
                                 if g_.y_true.nunique() > 1 else np.nan))
        if recs:
            Sdf = pd.DataFrame(recs)
            wide = Sdf.pivot_table(index="subject", columns="context_size",
                                   values=["accuracy", "auc"])
            wide.columns = [f"{a} · ctx {b}" for a, b in wide.columns]
            wide = wide.reset_index().sort_values(f"auc · ctx {S_main}", ascending=False)
            above = int((wide[f"auc · ctx {S_main}"] > 0.5).sum())
            subj_html = (f'<p>{above} / {len(wide)} subjects above chance at context '
                         f'{S_main}.</p>' + _tbl(wide, fmt={c: f3 for c in wide.columns
                                                            if c != "subject"}))

    # ------------------------------------------------------------- the split
    subs_all = np.load(os.path.join(BANDS_D, "subject.npy"), allow_pickle=True).astype(str)
    split_tbl = _tbl(pd.DataFrame(check_split(subs_all)))

    # ------------------------------------------------------------ discovery
    disc_html = ""
    if info:
        rows = []
        for c, d in info["conditions"].items():
            rows.append(dict(condition=c, file=os.path.relpath(d["path"], ROOT),
                             shape=" × ".join(map(str, d["shape"])), dtype=d["dtype"],
                             window=f"{d['window_s'][0]}–{d['window_s'][1]} s",
                             sfreq=f"{d['sfreq_hz']:g} Hz", size=f"{d['mb']:.0f} MB"))
        z = info["zscore_check"]
        disc_html = (
            _tbl(pd.DataFrame(rows), right=[])
            + f'<p>{info["n_trials"]:,} trials · {info["n_subjects"]} subjects · '
              f'{info["n_focus"]:,} focus / {info["n_distractor"]:,} distractor · '
              f'{info["trials_per_subject"]["min"]}–{info["trials_per_subject"]["max"]} '
              f'trials per subject. Per-subject z-scoring verified on '
              f'<span class="mono">{z["subject"]}</span>: mean {z["mean"]:+.4f}, '
              f'std {z["std"]:.4f}.</p>')

    # --------------------------------------------- what the sweep actually showed
    verdict_html = ""
    if len(main) and len(solo):
        d = main.auc.mean() - solo.auc.mean()
        pooled = float(np.sqrt((main.auc.std() ** 2 + solo.auc.std() ** 2) / 2))
        stab = ""
        if ST is not None and len(ST):
            s64 = ST[ST.context_size == S_main]
            if len(s64):
                stab = (f" Re-grouping the same test trials into different contexts moves "
                        f"an individual prediction by only {s64.per_trial_prob_sd.mean():.3f} "
                        f"in probability, so the model is very nearly ignoring its context "
                        f"outright.")
        if abs(d) < 0.5 * pooled:
            verdict_html = (
                f'<div class="note warn"><b>Cross-trial attention does not help on this '
                f'dataset: ΔAUC = {d:+.3f}, against a between-fold spread of '
                f'±{pooled:.3f}.</b> The curve is flat from context 1 to context '
                f'{S_main}, and the difference is far inside the noise.{stab}</div>'
                f'<p><b>Why, most likely.</b> The calibration this architecture was built '
                f'to perform has already been done to the data before the model sees it. '
                f'<span class="mono">07_bands.py</span> z-scores every channel '
                f'<em>per subject</em> — verified in the discovery table at the bottom of '
                f'this page — so each person\'s amplitude scale and electrode impedances '
                f'are already divided out. That per-subject nuisance variance was exactly '
                f'the quantity attention across a person\'s trials could have estimated '
                f'and removed, and it is gone by the time the encoder runs. What is left '
                f'is a time-locked evoked response inside each individual 3-second trial, '
                f'and neighbouring trials carry no extra evidence about it.</p>'
                f'<p>The result is worth having as a measurement rather than a '
                f'disappointment: it says the per-subject normalisation already in this '
                f'pipeline is doing the work, and that a heavier set-level model is not '
                f'where the remaining headroom is.</p>')
        elif d > 0:
            verdict_html = (
                f'<div class="note ok"><b>Cross-trial attention helps: ΔAUC = {d:+.3f} '
                f'(context {S_main} over context 1), against a between-fold spread of '
                f'±{pooled:.3f}.</b> Read this together with the stress test below — a '
                f'gain here is only meaningful if it survives a skewed context.{stab}</div>')
        else:
            verdict_html = (
                f'<div class="note warn"><b>Cross-trial attention <em>hurts</em>: '
                f'ΔAUC = {d:+.3f}.</b> The context adds parameters and gradient noise '
                f'without adding evidence; context 1 is the configuration to use.{stab}</div>')

    figs = {}
    if len(R):
        figs["sweep"] = _sweep_png(R)
    c1 = _curves_png(H, S_main) if H is not None and len(H) else None
    c2 = _curves_png(H, 1) if H is not None and len(H) else None
    sk = _skew_png(K, S_main) if K is not None and len(K) else None
    img = lambda b: f'<img src="data:image/png;base64,{b}">' if b else "<p>—</p>"

    warn = ('<div class="note warn"><b>QUICK SMOKE TEST.</b> Shortened schedule — '
            'these numbers are not results.</div>') if quick else ""

    blob = json.dumps(dict(
        generated=time.strftime("%Y-%m-%d %H:%M"), model=MODEL, main_context=S_main,
        context_sizes=sizes, discovery=info,
        results=R.to_dict("records"),
        skew=(K.to_dict("records") if K is not None and len(K) else []),
        stability=(ST.to_dict("records") if ST is not None and len(ST) else []),
        split=check_split(subs_all)), default=str)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Context Transformer — Focus vs. Distractor</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Context Transformer</h1>
<p class="sub">EEGNet encoder → 64-d trial embedding → self-attention <b>across trials</b> ·
subject-independent 5-fold · {PRETTY['full']} · 125 Hz ·
generated {time.strftime('%Y-%m-%d %H:%M')}</p>
{warn}
<div class="cards">{cards_html}</div>

<h2>What this model does differently</h2>
<p>Every other model in this repository classifies one 3-second trial in isolation.
This one encodes each trial to a single 64-dimensional vector with an EEGNet-style
convolutional front end, then runs transformer self-attention <b>over a group of
{S_main} trials from the same person</b> and emits a Focus/Distractor label for every
trial in the group at once.</p>
<p>The reason to expect a gain is that a subject-independent split hands the network a
person it has never seen, whose amplitude scale and electrode impedances are unknown.
A single trial gives it no way to separate the effect from the person. {S_main} trials
seen together let attention estimate the common part and subtract it — an unsupervised,
test-time, per-subject calibration that a per-trial model structurally cannot perform.</p>
<div class="note"><b>No positional embedding, and no CLS token.</b> The trials in a
context are a set, not a sequence: they are drawn at random from one person's session,
so slot 12 sitting next to slot 40 means nothing. Every token is an answer, so there is
nothing to summarise.</div>
<div class="note warn"><b>This is transductive inference — read the number with that in
mind.</b> To classify one trial at context {S_main}, the model needs {S_main - 1} other
trials from the same person available at the same time. It never sees their labels, so
there is no leakage, but it is not a drop-in replacement for a per-trial classifier: it
cannot label the very first trial of a session in isolation. It fits a workflow where a
block of trials is scored after the fact, and does not fit one that must emit a verdict
the instant each trial ends. <b>Context 1 is the number to quote for that second
case</b>, and it is measured here on exactly the same folds.</div>
{eq_html}

<h3>Tensor shapes — traced through the real model</h3>
<pre class="trace">{trace}</pre>
<div class="note">{meta.depth} pre-norm encoder blocks over the trial axis ·
{meta.heads} heads · embedding {meta.emb_dim}-d · dropout 0.5 ·
{int(meta.n_params):,} trainable parameters
({int(meta.p_encoder):,} in the EEGNet encoder, {int(meta.p_attention):,} in the
cross-trial attention).</div>

<h2>Does attending across trials help?</h2>
{img(figs.get("sweep"))}
<p>Context size 1 is the ablation: the identical architecture with a context of one,
where self-attention over a single token is the identity and the model collapses to a
plain per-trial classifier. The difference between the two ends of this curve is the
entire effect of cross-trial attention, measured with everything else held fixed —
same folds, same encoder, same optimiser, same number of trials per optimiser step.</p>
{sweep_html}
{verdict_html}

<h2>The shortcut stress test</h2>
<p>A set model handed {S_main} trials that are roughly half Focus can cheat: rank them
by some internal score and call the top half Distractor. That posts a good AUC while
learning nothing about any individual trial. This table rebuilds the test contexts at
<em>fixed</em> compositions, including contexts where every single trial carries the
same label — where a median split is guaranteed to be half wrong.</p>
{img(sk)}
{skew_html}
{skew_verdict}

<h3>How much does the answer depend on which neighbours a trial got?</h3>
{stab_html}

<h2>Result — {PRETTY['full']}, context {S_main}</h2>
{fold_html}

<h3>Training curves — context {S_main}</h3>
{img(c1)}
<h3>Training curves — context 1 (ablation)</h3>
{img(c2)}

<h2>Comparison with every other model in the repo</h2>
<p>Same folds, same subjects, same 125 Hz input, same per-subject normalisation, same
optimiser and early-stopping rule. Read straight from the existing result files.</p>
{comp_html}

<h2>Confusion matrix — context {S_main}</h2>
{conf_html}

<h2>Control window — the leakage guardrail</h2>
<p>0.0–2.0 s, before the notification. The two conditions are <em>physically identical</em>
here, so any real skill in this window would be a bug. For this model the control does
double duty: it also checks that the act of grouping trials into contexts does not by
itself carry label information.</p>
{ctrl_html}

<h2>Subject-independent split</h2>
<p>Produced by <span class="mono">folds_for()</span> imported from
<span class="mono">07_bands.py</span> — the same function that split the data for EEGNet,
CTNet, LMDA-Net, EEG Conformer and the stage-9 transformer, not a copy of it.
Re-derived and asserted on every run.</p>
{split_tbl}
<ul class="chk">
<li><b>✓</b> test subjects never appear in training or validation</li>
<li><b>✓</b> a context never mixes subjects, so grouping cannot bridge the split</li>
<li><b>✓</b> model selection (early stopping) uses validation subjects only</li>
<li><b>✓</b> normalisation is per-subject, label-free, fitted on each person's own trials</li>
<li><b>✓</b> class weights computed from the training labels only</li>
<li><b>✓</b> contexts carry EEG from their filler trials but never a label</li>
<li><b>✓</b> every test trial is scored exactly once, asserted in code</li>
<li><b>✓</b> best-validation checkpoint restored before the test set is touched</li>
</ul>

<h2>Per-subject</h2>
{subj_html}

<h2>Dataset as discovered on disk</h2>
{disc_html}

<footer>Every number above is embedded below as JSON — a rendered page cannot be
re-parsed, so the data travels with it.
<script type="application/json" id="report-data">{blob}</script>
</footer>
</div></body></html>"""

    out = os.path.join(OUT_D, "context_transformer_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out


def read_data(path=None):
    """Recover the embedded numbers from a rendered report."""
    path = path or os.path.join(OUT_D, "context_transformer_report.html")
    txt = open(path, encoding="utf-8").read()
    marker = '<script type="application/json" id="report-data">'
    a = txt.index(marker) + len(marker)
    return json.loads(txt[a:txt.index("</script>", a)])


def report(conds=None, sizes=None, quick=False, info=None):
    R, P, H, K, ST = read_jobs()
    if R is None:
        print("no finished jobs - nothing to report")
        return
    out = build_report(R, P, H, K, ST, info=info, quick=quick)
    full = R[R.condition == "full"]
    sizes_done = sorted(R.context_size.unique())
    S_main = MAIN_CONTEXT if MAIN_CONTEXT in sizes_done else max(sizes_done)
    print("\n" + "=" * 74)
    print(f"{MODEL}  -  subject-independent {full.fold.nunique() if len(full) else 0}-fold")
    print("=" * 74)
    for S in sizes_done:
        d = full[full.context_size == S]
        if len(d):
            print(f"  context {S:<4} acc {d.accuracy.mean():.3f} +- {d.accuracy.std():.3f}"
                  f"   auc {d.auc.mean():.3f} +- {d.auc.std():.3f}"
                  f"   kappa {d.kappa.mean():.3f}")
    m, s = full[full.context_size == S_main], full[full.context_size == 1]
    if len(m) and len(s):
        print(f"\n  cross-trial attention effect: AUC {m.auc.mean()-s.auc.mean():+.3f} "
              f"(context {S_main} vs context 1)")
    ctrl = R[R.condition == "control"]
    if len(ctrl):
        w = float(np.nanmax(np.abs(ctrl.groupby('context_size').auc.mean() - 0.5)))
        print(f"  control window worst |auc-0.5| = {w:.3f}"
              f"{'   <-- CHECK' if w >= 0.06 else '   (at chance, ok)'}")
    if K is not None and len(K):
        d = K[(K.context_size == S_main)]
        nat = d[d.focus_fraction == 0.5]
        pf = d[d.focus_fraction == 1.0]
        pdd = d[d.focus_fraction == 0.0]
        au = d.groupby("focus_fraction").auc.mean().dropna()
        if len(au) > 1:
            sp = float(au.max() - au.min())
            print(f"  skew stress: auc {au.min():.3f}-{au.max():.3f} across "
                  f"compositions 90/10..10/90 (spread {sp:.3f})"
                  f"{'   <-- median-split shortcut' if sp >= 0.05 else '   (ranking holds)'}")
        if len(nat) and len(pf) and len(pdd):
            base = np.nanmean([nat.recall_focus.mean(), nat.recall_distr.mean()])
            pure = np.nanmean([pf.recall_focus.mean(), pdd.recall_distr.mean()])
            print(f"               recall {base:.3f} on natural contexts -> {pure:.3f} "
                  f"on pure single-label contexts (threshold slide)")
    miss = pending(all_jobs(conds or CONDITIONS, sizes or CONTEXT_SIZES,
                            full.fold.nunique() if len(full) else N_FOLDS))
    if miss:
        print(f"\nWARNING: {len(miss)} job(s) missing and EXCLUDED from the means above "
              f"(not scored as 0.5): {miss}")
    print(f"\nwritten -> {out}")


# ====================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--job", nargs=3, metavar=("COND", "CTX", "FOLD"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--context-sizes", default=",".join(map(str, CONTEXT_SIZES)))
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--inspect", action="store_true",
                    help="report what data is on disk and exit")
    ap.add_argument("--report", action="store_true", help="rebuild the report only")
    ap.add_argument("--quick", action="store_true",
                    help="1 fold, 3 epochs, 2 context sizes - a smoke test, not a result")
    a = ap.parse_args()

    if a.job:
        import torch
        torch.set_num_threads(int(os.environ.get("EEG_TORCH_THREADS", 4)))
        run_job(a.job[0], int(a.job[1]), int(a.job[2]), quick=a.quick)
        sys.exit(0)

    if a.inspect:
        discover(verbose=True, allow_build=False)
        sys.exit(0)

    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    sizes = sorted({int(s) for s in a.context_sizes.split(",") if s.strip()})
    if a.quick:
        sizes = [s for s in sizes if s in (1, MAIN_CONTEXT)] or [1, MAIN_CONTEXT]

    if a.report:
        report(conds, sizes, info=discover(verbose=False))
        sys.exit(0)

    info = discover(verbose=True)
    n_folds = 1 if a.quick else a.folds
    if a.quick:
        print("\n*** QUICK SMOKE TEST - 1 fold, 3 epochs, results are meaningless ***")
    from dl_models import resolve_device
    dev = resolve_device(a.device).type
    if dev == "cuda":            # one CUDA context, not N - see run_inline
        run_inline(conds, sizes, n_folds, a.quick, "cuda", a.threads)
    else:
        orchestrate(conds, sizes, a.workers, a.threads, n_folds, a.quick, dev)
    report(conds, sizes, quick=a.quick, info=info)
