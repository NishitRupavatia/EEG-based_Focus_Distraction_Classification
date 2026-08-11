"""
STEP 4 - Train and evaluate 6 models, SUBJECT-INDEPENDENT.

Subject-independent means: the people in the test set are never seen during training,
in any form. We split by PERSON, never by trial. This answers the only question that
matters for a real system - "will it work on a new user?" - and it is the one split
that cannot be faked by the model memorising individual brains.

THE SPLIT
  23 subjects
   |
   +- OUTER: subject-wise 5-fold        -> test = 4-5 subjects per fold
        |
        +- INNER: 4 more subjects held out as VALIDATION
              TRAIN      = 14-15 subjects   (~2,800 trials)
              VALIDATION = 4 subjects       (~790 trials)   EEGNet early stopping only
              TEST       = 4-5 subjects     (~790-990 trials)

  All 6 models train on the SAME train set so the comparison is fair. Only EEGNet
  uses the validation set (it needs to know when to stop); the classical models fit
  in one shot and ignore it.

THE MODELS
  1 LR-bandpower      logistic regression on log band power
  2 LDA-shrinkage     shrinkage LDA on band power + ERP bins
  3 CSP+LDA           common spatial patterns, fitted INSIDE each fold
  4 Riemann-TS        tangent space on covariances, with per-subject recentring
  5 Xdawn+TS          Xdawn covariances + tangent space, fitted INSIDE each fold
  6 EEGNet            compact CNN, PyTorch

  CSP and Xdawn use the labels to build their filters, so they are re-fitted inside
  every fold. Precomputing them once over the whole dataset would leak the test
  subjects' labels into training and inflate every score.

GUARDRAILS (run automatically)
  * Every model is also run on the 0-2 s "pre" window, where the two conditions are
    physically identical. Anything much above 0.50 there means a leak.
  * Label-shuffle control on the classical models -> must land at ~0.50.
"""
import os, glob, json, time, warnings
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             precision_recall_fscore_support, cohen_kappa_score,
                             confusion_matrix)
from mne.decoding import CSP
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.base import invsqrtm

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIV = os.path.join(ROOT, "derivatives")
MODELS_D = os.path.join(DERIV, "models")
os.makedirs(MODELS_D, exist_ok=True)

# ----------------------------------------------------------------- config
SEED        = 42
N_FOLDS     = 5
N_VAL_SUBS  = 4
SFREQ       = 250.0
WINDOWS     = {"post": (2.0, 5.0),      # main analysis window
               "pre":  (0.0, 2.0)}      # CONTROL - conditions identical here
ERP_BIN_S   = 0.100
EEGNET_DECIM = 2                        # 250 -> 125 Hz, EEGNet's design point
MAX_EPOCHS  = 300
PATIENCE    = 40
BATCH       = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT     = 0.5                       # 0.5 is the cross-subject recommendation

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED); np.random.seed(SEED)


# ================================================================= data
def load_data():
    """Returns raw time-series per window (250 Hz), band-power features, subjects, labels."""
    d = np.load(os.path.join(DERIV, "features.npz"), allow_pickle=True)
    files = sorted(glob.glob(os.path.join(DERIV, "epochs", "*-epo.fif")))
    raw, subs = {w: [] for w in WINDOWS}, []
    for f in files:
        ep = mne.read_epochs(f, preload=True, verbose="ERROR")
        t = ep.times
        X = (ep.get_data(copy=True) * 1e6).astype(np.float32)
        for w, (a, b) in WINDOWS.items():
            raw[w].append(X[:, :, (t >= a) & (t < b)])
        subs.append(ep.metadata["subject"].values)
    raw = {w: np.concatenate(v) for w, v in raw.items()}
    subs = np.concatenate(subs)
    assert (subs == d["subject"]).all(), "epoch order does not match features.npz"
    bp = {w: np.concatenate([d[f"bp_abs_{w}"].reshape(len(subs), -1),
                             d[f"bp_rel_{w}"].reshape(len(subs), -1)], 1).astype(np.float32)
          for w in WINDOWS}
    cov = {w: d[f"cov_bb_{w}"].astype(np.float64) for w in WINDOWS}
    return raw, bp, cov, subs, d["label"].astype(int)


def erp_bins(X, sf=SFREQ, bin_s=ERP_BIN_S):
    step = int(round(bin_s * sf)); n = X.shape[-1] // step
    return np.stack([X[:, :, i*step:(i+1)*step].mean(-1) for i in range(n)], -1)


def subject_zscore(X, subjects):
    """Standardise each person using ONLY their own data. No labels involved, so it is
    legal to apply to test subjects too - this is what makes cross-subject transfer work."""
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(subjects):
        m = subjects == s
        mu, sd = X[m].mean(0), X[m].std(0)
        out[m] = (X[m] - mu) / np.where(sd < 1e-12, 1.0, sd)
    return out


def subject_recenter(C, subjects):
    """Riemannian recentring: shift each person's covariances to their own centre.
    Unsupervised, so also legal on test subjects. Usually the biggest single win
    for cross-subject Riemannian pipelines."""
    out = np.empty_like(C)
    for s in np.unique(subjects):
        m = subjects == s
        W = invsqrtm(mean_covariance(C[m], metric="riemann"))
        out[m] = W @ C[m] @ W
    return out


def zscore_raw(X, subjects):
    """Per-subject, per-channel standardisation of the time series (for EEGNet)."""
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(subjects):
        m = subjects == s
        mu = X[m].mean(axis=(0, 2), keepdims=True)
        sd = X[m].std(axis=(0, 2), keepdims=True)
        out[m] = (X[m] - mu) / np.where(sd < 1e-12, 1.0, sd)
    return out


# ================================================================= EEGNet
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018)."""
    def __init__(self, n_ch, n_samples, n_classes=2, F1=8, D=2, F2=16,
                 kern_len=64, drop=DROPOUT):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding="same", bias=False),
            nn.BatchNorm2d(F1))
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():                      # work out the flatten size
            z = self.block3(self.block2(self.depthwise(
                self.block1(torch.zeros(1, 1, n_ch, n_samples)))))
            self.n_flat = int(np.prod(z.shape[1:]))
        self.fc = nn.Linear(self.n_flat, n_classes)

    def forward(self, x):
        z = self.block3(self.block2(self.depthwise(self.block1(x))))
        return self.fc(z.flatten(1))

    @torch.no_grad()
    def apply_max_norm(self):
        """EEGNet's regularisation: cap the norm of the spatial filters and the head."""
        w = self.depthwise.weight
        n = w.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        w.mul_(n.clamp(max=1.0) / n)
        w2 = self.fc.weight
        n2 = w2.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
        w2.mul_(n2.clamp(max=0.25) / n2)


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    tot = correct = 0; loss_sum = 0.0
    for xb, yb in loader:
        with torch.set_grad_enabled(train):
            out = model(xb); loss = crit(out, yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step(); model.apply_max_norm()
        loss_sum += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item(); tot += len(yb)
    return loss_sum / tot, correct / tot


def train_eegnet(Xtr, ytr, Xva, yva, Xte, tag):
    """Returns test probabilities and the per-epoch history."""
    n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
    model = EEGNet(n_ch, n_t)
    cw = torch.tensor([len(ytr) / (2 * (ytr == c).sum()) for c in (0, 1)], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def mk(X, y=None, shuf=False):
        t = torch.tensor(X[:, None], dtype=torch.float32)
        yy = torch.tensor(y if y is not None else np.zeros(len(X)), dtype=torch.long)
        return DataLoader(TensorDataset(t, yy), batch_size=BATCH, shuffle=shuf)

    tr_dl, va_dl = mk(Xtr, ytr, True), mk(Xva, yva)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        trl, tra = run_epoch(model, tr_dl, crit, opt)
        val, vaa = run_epoch(model, va_dl, crit)
        hist.append(dict(tag=tag, epoch=ep, train_loss=trl, val_loss=val,
                         train_acc=tra, val_acc=vaa))
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    stopped = ep; best_ep = int(np.argmin([h["val_loss"] for h in hist])) + 1
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(Xte[:, None], dtype=torch.float32)), 1)[:, 1].numpy()
    for h in hist:
        h["best_epoch"] = best_ep; h["stopped_epoch"] = stopped
    return p, hist, model, best_ep, stopped


# ================================================================= metrics
def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average=None,
                                                    labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(
        accuracy=accuracy_score(y, yh), balanced_accuracy=balanced_accuracy_score(y, yh),
        auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        precision_focus=pr[0], recall_focus=rc[0], f1_focus=f1[0],
        precision_distr=pr[1], recall_distr=rc[1], f1_distr=f1[1],
        f1_macro=(f1[0] + f1[1]) / 2, kappa=cohen_kappa_score(y, yh),
        sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
        tn=tn, fp=fp, fn=fn, tp=tp, n=len(y))


# ================================================================= main
def main(quick=False):
    """quick=True runs one fold with 3 EEGNet epochs - a smoke test, not a result."""
    global MAX_EPOCHS, PATIENCE
    if quick:
        MAX_EPOCHS, PATIENCE = 3, 2
        print("*** QUICK SMOKE TEST - 1 fold, 3 epochs, results are meaningless ***\n")
    t0 = time.time()
    print("Loading data ...", flush=True)
    raw, bp, cov, subjects, y = load_data()
    subs = np.array(sorted(np.unique(subjects)))
    print(f"  {len(y)} trials, {len(subs)} subjects, focus={np.sum(y==0)} distractor={np.sum(y==1)}")

    # --- precompute per-window inputs (all unsupervised transforms) ---
    prep = {}
    for w in WINDOWS:
        prep[w] = dict(
            bp=subject_zscore(bp[w], subjects),
            erp=subject_zscore(erp_bins(raw[w]).reshape(len(y), -1), subjects),
            raw=raw[w],
            raw_z=zscore_raw(raw[w][:, :, ::EEGNET_DECIM], subjects),
            cov=subject_recenter(cov[w], subjects))
        print(f"  window '{w}': raw {raw[w].shape}, EEGNet input {prep[w]['raw_z'].shape}")

    # --- subject-wise folds ---
    perm = rng.permutation(len(subs))
    folds = [subs[ix] for ix in np.array_split(perm, N_FOLDS)]
    print("\nSUBJECT-INDEPENDENT 5-FOLD SPLIT")
    for i, te in enumerate(folds):
        pool = np.array([s for s in subs if s not in te])
        va = pool[np.random.default_rng(SEED + i).permutation(len(pool))[:N_VAL_SUBS]]
        tr = np.array([s for s in pool if s not in va])
        print(f"  fold {i}: train {len(tr):>2} | val {len(va)} | test {len(te)}   test={list(te)}")

    # A model that throws is recorded here and EXCLUDED from the results table.
    # It is never silently replaced with 0.5 - a constant 0.5 scores AUC 0.500 and
    # looks exactly like a legitimate chance-level result, which would hide the bug.
    rows, preds, hists, failures = [], [], [], []
    for i, te_s in enumerate(folds):
        if quick and i > 0:
            break
        pool = np.array([s for s in subs if s not in te_s])
        va_s = pool[np.random.default_rng(SEED + i).permutation(len(pool))[:N_VAL_SUBS]]
        tr_s = np.array([s for s in pool if s not in va_s])
        itr = np.isin(subjects, tr_s); iva = np.isin(subjects, va_s); ite = np.isin(subjects, te_s)
        ytr, yva, yte = y[itr], y[iva], y[ite]

        for w in WINDOWS:
            P = prep[w]
            tag = f"fold{i}_{w}"
            ctrl = " [CONTROL]" if w == "pre" else ""
            print(f"\n--- fold {i}  window '{w}'{ctrl}  "
                  f"train {itr.sum()} / val {iva.sum()} / test {ite.sum()} ---", flush=True)
            out = {}

            # 1 logistic regression on band power
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
            m.fit(P["bp"][itr], ytr); out["LR-bandpower"] = m.predict_proba(P["bp"][ite])[:, 1]

            # 2 shrinkage LDA on band power + ERP bins
            Xc = np.concatenate([P["bp"], P["erp"]], 1)
            m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            m.fit(Xc[itr], ytr); out["LDA-shrinkage"] = m.predict_proba(Xc[ite])[:, 1]

            # 3 CSP + LDA  (CSP uses labels -> fitted inside the fold)
            # NOTE ON RANK: average referencing removes 1 degree of freedom and ICA
            # removed up to 3 more components, so the 15-channel covariance is only
            # rank ~11. Any method doing a generalised eigendecomposition needs a
            # SHRINKAGE estimator or it will fail on the singular matrix.
            try:
                csp = CSP(n_components=6, reg="oas", log=True)
                Ztr = csp.fit_transform(P["raw"][itr].astype(np.float64), ytr)
                Zte = csp.transform(P["raw"][ite].astype(np.float64))
                m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
                m.fit(Ztr, ytr); out["CSP+LDA"] = m.predict_proba(Zte)[:, 1]
            except Exception as e:
                failures.append(f"fold{i}/{w}/CSP+LDA: {type(e).__name__}: {str(e)[:90]}")
                print(f"    !! CSP+LDA FAILED -> excluded from results ({type(e).__name__})")

            # 4 Riemannian tangent space on recentred covariances
            ts = TangentSpace(metric="riemann")
            Ztr = ts.fit_transform(P["cov"][itr]); Zte = ts.transform(P["cov"][ite])
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
            m.fit(Ztr, ytr); out["Riemann-TS"] = m.predict_proba(Zte)[:, 1]

            # 5 Xdawn covariances + tangent space  (Xdawn uses labels -> inside the fold)
            # xdawn_estimator MUST be a shrinkage estimator ('lwf'), not 'scm':
            # the default 'scm' throws LinAlgError on our rank-deficient data.
            try:
                xd = XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="lwf")
                Ztr = xd.fit_transform(P["raw"][itr].astype(np.float64), ytr)
                Zte = xd.transform(P["raw"][ite].astype(np.float64))
                ts2 = TangentSpace(metric="riemann")
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
                m.fit(ts2.fit_transform(Ztr), ytr)
                out["Xdawn+TS"] = m.predict_proba(ts2.transform(Zte))[:, 1]
            except Exception as e:
                failures.append(f"fold{i}/{w}/Xdawn+TS: {type(e).__name__}: {str(e)[:90]}")
                print(f"    !! Xdawn+TS FAILED -> excluded from results ({type(e).__name__})")

            # 6 EEGNet
            te_p, hist, model, best_ep, stopped = train_eegnet(
                P["raw_z"][itr], ytr, P["raw_z"][iva], yva, P["raw_z"][ite], tag)
            out["EEGNet"] = te_p
            hists += hist
            torch.save(model.state_dict(), os.path.join(MODELS_D, f"eegnet_{tag}.pt"))
            print(f"    EEGNet stopped at epoch {stopped} (best {best_ep})")

            for name, p in out.items():
                r = metrics(yte, p)
                r.update(model=name, fold=i, window=w, n_train=int(itr.sum()),
                         n_val=int(iva.sum()), n_test=int(ite.sum()),
                         test_subjects="|".join(te_s))
                rows.append(r)
                preds.append(pd.DataFrame(dict(fold=i, window=w, model=name,
                                               subject=subjects[ite], y_true=yte, y_prob=p)))
                print(f"    {name:<14} acc {r['accuracy']:.3f}  auc {r['auc']:.3f}")

            # guardrail: shuffled labels (classical models only - cheap)
            ysh = ytr.copy(); np.random.default_rng(SEED + i).shuffle(ysh)
            ms = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
            ms.fit(P["bp"][itr], ysh)
            rsh = metrics(yte, ms.predict_proba(P["bp"][ite])[:, 1])
            rsh.update(model="SHUFFLED-control", fold=i, window=w, n_train=int(itr.sum()),
                       n_val=int(iva.sum()), n_test=int(ite.sum()), test_subjects="|".join(te_s))
            rows.append(rsh)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(DERIV, "results_cv5.csv"), index=False)
    pd.concat(preds, ignore_index=True).to_csv(os.path.join(DERIV, "predictions_cv5.csv"), index=False)
    pd.DataFrame(hists).to_csv(os.path.join(DERIV, "eegnet_history.csv"), index=False)

    # ---------------------------------------------------------- summary
    pd.set_option("display.width", 250, "display.max_columns", 40)
    num = ["accuracy", "balanced_accuracy", "auc", "f1_macro", "kappa",
           "sensitivity", "specificity"]
    for w in WINDOWS:
        sub = R[R.window == w]
        g = sub.groupby("model")[num].agg(["mean", "std"])
        g = g.sort_values(("auc", "mean"), ascending=False)
        head = "MAIN WINDOW 2.0-5.0 s" if w == "post" else \
               "CONTROL WINDOW 0.0-2.0 s  (conditions identical - everything must be ~0.50)"
        print("\n" + "=" * 104); print(head); print("=" * 104)
        print(f"{'model':<18}" + "".join(f"{c:>16}" for c in num))
        for mdl in g.index:
            print(f"{mdl:<18}" + "".join(
                f"{g.loc[mdl,(c,'mean')]:>9.3f}+-{g.loc[mdl,(c,'std')]:.3f}" for c in num))

    print("\n" + "=" * 104); print("CONFUSION MATRICES  (summed over the 5 folds, main window)")
    for mdl in R[R.window == "post"].model.unique():
        s = R[(R.window == "post") & (R.model == mdl)][["tn", "fp", "fn", "tp"]].sum()
        tot = s.sum()
        print(f"\n  {mdl}   (n={tot})")
        print(f"                 pred focus   pred distr")
        print(f"    true focus  {s.tn:>10}   {s.fp:>10}     ({100*s.tn/max(s.tn+s.fp,1):.1f}% correct)")
        print(f"    true distr  {s.fn:>10}   {s.tp:>10}     ({100*s.tp/max(s.fn+s.tp,1):.1f}% correct)")

    h = pd.DataFrame(hists)
    if len(h):
        e = h.groupby("tag").agg(best_epoch=("best_epoch", "first"),
                                 stopped=("stopped_epoch", "first"))
        print("\n" + "=" * 104); print("EEGNet TRAINING LENGTH")
        print(f"  max allowed {MAX_EPOCHS}, early-stopping patience {PATIENCE}")
        print(f"  stopped at  : mean {e.stopped.mean():.0f}  range {e.stopped.min()}-{e.stopped.max()}")
        print(f"  best epoch  : mean {e.best_epoch.mean():.0f}  range {e.best_epoch.min()}-{e.best_epoch.max()}")

    print("\n" + "=" * 104)
    if failures:
        print(f"MODEL FAILURES ({len(failures)}) - these were EXCLUDED, not scored as 0.5:")
        for f in failures:
            print(f"  {f}")
    else:
        print("MODEL FAILURES: none - every model fitted successfully in every fold.")

    print(f"\ntotal runtime {(time.time()-t0)/60:.1f} min")
    print("written -> results_cv5.csv, predictions_cv5.csv, eegnet_history.csv, models/")


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
