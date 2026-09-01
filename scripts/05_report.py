"""
STEP 5 - Figures.

Reads what 04_train.py saved and draws:
  fig1  model comparison        AUC + accuracy, main window vs control window
  fig2  confusion matrices      one panel per model
  fig3  deep-net training vs validation LOSS      (per fold + mean)
  fig4  deep-net training vs validation ACCURACY  (per fold + mean)
  fig5  ROC curves
  fig6  per-subject accuracy
  fig7  time-resolved decoding  <- the one that shows the confound

If 07_bands.py has run, this picks up results_all.csv / predictions_all.csv /
dl_history_all.csv instead, which are the same tables with CTNet and LMDA-Net
merged in. Falls back to the 04_train.py originals when the sweep has not run, so
this script works either way.

Kept separate from 04_train.py on purpose: training is expensive, plotting is cheap.
A bug in here costs seconds to fix, not an hour of retraining.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc as auc_fn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Both redirectable, so the figures can be regenerated from a trial set of results
# without overwriting the published ones. Unset, they are the normal project folders.
DERIV = os.environ.get("EEG_DERIV_DIR") or os.path.join(ROOT, "derivatives")
FIGD  = os.environ.get("EEG_FIG_DIR") or os.path.join(ROOT, "figures")
os.makedirs(FIGD, exist_ok=True)

# ---- validated palette (see references/palette.md; checked with validate_palette.js)
# Eight slots, in the documented order - re-validated as a set when CTNet and LMDA-Net
# were added: worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6, both
# above their gates. Three slots WARN on contrast, so every bar carries its value.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948", "#8c5a30"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8981"
SURFACE, GRID = "#fcfcfb", "#e5e4e0"
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

# Colour follows the ENTITY, never its rank - fixed model->hue mapping. The six
# original models keep the hue they have always had; the two new networks take the
# next free slots, so nothing in the old figures changes colour.
MODEL_ORDER = ["LR-bandpower", "LDA-shrinkage", "CSP+LDA", "Riemann-TS", "Xdawn+TS",
               "EEGNet", "CTNet", "LMDA-Net", "EEG Conformer"]
COLOR = dict(zip(MODEL_ORDER, SERIES))
DEEP = ["EEGNet", "CTNet", "LMDA-Net", "EEG Conformer"]   # the ones with training curves
CTRL_MODEL = "SHUFFLED-control"          # a reference, not a series -> neutral grey

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


def style(ax, ygrid=True):
    ax.grid(axis="y" if ygrid else "x", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def newest(preferred, fallback):
    """Prefer the merged table written by 07_bands.py, fall back to 04_train.py's."""
    p = os.path.join(DERIV, preferred)
    use = p if os.path.exists(p) else os.path.join(DERIV, fallback)
    print(f"  reading {os.path.basename(use)}")
    return pd.read_csv(use)


R = newest("results_all.csv", "results_cv5.csv")
P = newest("predictions_all.csv", "predictions_cv5.csv")
H = newest("dl_history_all.csv", "eegnet_history.csv")
if "model" not in H.columns:                 # the original eegnet_history.csv has no
    H["model"] = "EEGNet"                    # model column - it only ever held one
models = [m for m in MODEL_ORDER if m in set(R.model)]
deep = [m for m in DEEP if m in set(H.model)]

# ============================================================ fig 1
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
for ax, metric, title in zip(axes, ["auc", "accuracy"],
                             ["ROC-AUC", "Accuracy"]):
    x = np.arange(len(models)); w = 0.38; top = 0.0
    for k, (win, lbl) in enumerate([("post", "Main window 2.0-5.0 s"),
                                    ("pre", "Control 0.0-2.0 s")]):
        mu = [R[(R.model == m) & (R.window == win)][metric].mean() for m in models]
        sd = [R[(R.model == m) & (R.window == win)][metric].std() for m in models]
        cols = [COLOR[m] for m in models] if win == "post" else [MUTED] * len(models)
        ax.bar(x + (k - 0.5) * w, mu, w * 0.94, yerr=sd, color=cols,
               alpha=1.0 if win == "post" else 0.45, capsize=3, zorder=3,
               ecolor=INK2, error_kw=dict(lw=1),
               label=lbl if metric == "auc" else None)
        # relief rule: the palette WARNs on contrast, so every mark carries a value.
        # Sit it above the whisker, not the bar top, or the two overprint.
        for xi, v, s in zip(x + (k - 0.5) * w, mu, np.nan_to_num(sd)):
            ax.text(xi, v + s + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2 if win == "post" else MUTED, zorder=5)
            top = max(top, v + s)
    ax.axhline(0.5, color=INK2, ls="--", lw=1.2, zorder=2)
    ax.text(-0.75, 0.507, "chance", fontsize=8, color=INK2, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.35, min(1.0, top + 0.10)); ax.set_ylabel(title); ax.set_title(title)
    style(ax)
axes[0].legend(frameon=False, loc="upper left", fontsize=9, bbox_to_anchor=(0, 1.0))
fig.suptitle("Subject-independent 5-fold cross-validation  ·  23 subjects, 4,544 trials",
             fontsize=12, y=1.02)
fig.savefig(os.path.join(FIGD, "fig1_model_comparison.png")); plt.close(fig)

# ============================================================ fig 2
n = len(models); ncol = 3; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.7 * nrow))
for ax, m in zip(axes.ravel(), models):
    s = R[(R.window == "post") & (R.model == m)][["tn", "fp", "fn", "tp"]].sum()
    cm = np.array([[s.tn, s.fp], [s.fn, s.tp]], float)
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(pct, cmap=SEQ, vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j])}\n{pct[i,j]:.1f}%", ha="center", va="center",
                    fontsize=10, color="#ffffff" if pct[i, j] > 55 else INK)
    a = R[(R.window == "post") & (R.model == m)].accuracy.mean()
    ax.set_title(f"{m}\naccuracy {a:.3f}", color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred focus", "pred distr"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true focus", "true distr"])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle("Confusion matrices  ·  main window, summed over 5 folds", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig2_confusion_matrices.png")); plt.close(fig)

# ============================================================ fig 3 & 4  (deep nets)
# One row per architecture: left = all 5 folds thin, right = mean +- sd. Train and
# validation keep the SAME two hues in every row, because the comparison the reader
# makes here is train-vs-val within a panel, not model-vs-model across rows.
for metric, fname, title in [("loss", "fig3_deepnet_loss.png", "Loss"),
                             ("acc", "fig4_deepnet_accuracy.png", "Accuracy")]:
    fig, axes = plt.subplots(len(deep), 2, figsize=(12.5, 4.0 * len(deep)), squeeze=False)
    for r, mdl in enumerate(deep):
        hp = H[(H.model == mdl) & (H.tag.str.endswith("_post"))]
        # left: every fold, thin
        ax = axes[r][0]
        for tag, g in hp.groupby("tag"):
            ax.plot(g.epoch, g[f"train_{metric}"], color=SERIES[0], alpha=0.35, lw=1.2)
            ax.plot(g.epoch, g[f"val_{metric}"], color=SERIES[1], alpha=0.35, lw=1.2)
            ax.axvline(int(g.best_epoch.iloc[0]), color=MUTED, ls=":", lw=0.9)
        ax.plot([], [], color=SERIES[0], label="train")
        ax.plot([], [], color=SERIES[1], label="validation")
        ax.plot([], [], color=MUTED, ls=":", label="best epoch (early stop)")
        ax.set_title(f"{mdl} {title} — each of the 5 folds")
        ax.set_xlabel("epoch"); ax.set_ylabel(title.lower())
        ax.legend(frameon=False, fontsize=9)
        style(ax)
        # right: mean +- sd across folds, truncated to the shortest fold
        ax = axes[r][1]
        L = hp.groupby("tag").epoch.max().min()
        for col, c, lbl in [(f"train_{metric}", SERIES[0], "train"),
                            (f"val_{metric}", SERIES[1], "validation")]:
            M = np.stack([g.sort_values("epoch")[col].values[:L] for _, g in hp.groupby("tag")])
            e = np.arange(1, L + 1)
            ax.plot(e, M.mean(0), color=c, label=lbl)
            ax.fill_between(e, M.mean(0) - M.std(0), M.mean(0) + M.std(0),
                            color=c, alpha=0.16, lw=0)
        ax.set_title(f"{mdl} {title} — mean ± sd across folds")
        ax.set_xlabel("epoch"); ax.set_ylabel(title.lower())
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, fname)); plt.close(fig)

# ============================================================ fig 5  ROC
fig, ax = plt.subplots(figsize=(6.6, 6.0))
for m in models:
    d = P[(P.window == "post") & (P.model == m)]
    fpr, tpr, _ = roc_curve(d.y_true, d.y_prob)
    ax.plot(fpr, tpr, color=COLOR[m], label=f"{m}  (AUC {auc_fn(fpr,tpr):.3f})")
ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2, label="chance (AUC 0.500)")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("ROC — main window, all folds pooled")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
style(ax); ax.grid(alpha=0.7)
fig.savefig(os.path.join(FIGD, "fig5_roc.png")); plt.close(fig)

# ============================================================ fig 6  per subject
best = R[R.window == "post"].groupby("model").auc.mean().idxmax()
d = P[(P.window == "post") & (P.model == best)]
per = d.groupby("subject").apply(
    lambda g: pd.Series({"auc": roc_auc_score(g.y_true, g.y_prob) if g.y_true.nunique() > 1 else np.nan,
                         "acc": ((g.y_prob >= .5).astype(int) == g.y_true).mean()}),
    include_groups=False).sort_values("auc")
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.bar(per.index, per.auc, color=COLOR[best], zorder=3, width=0.72)
for i, v in enumerate(per.auc):
    ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
ax.axhline(0.5, color=INK2, ls="--", lw=1.2, zorder=4)
ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.3, 1.0)
ax.set_title(f"Per-subject performance — {best} (each subject scored when held out)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
style(ax); fig.savefig(os.path.join(FIGD, "fig6_per_subject.png")); plt.close(fig)

# ============================================================ fig 7  time-resolved
print("computing time-resolved decoding ...")
d = np.load(os.path.join(DERIV, "features.npz"), allow_pickle=True)
subj, y = d["subject"], d["label"].astype(int)
S = d["slide_bp"]                                   # (n, 15, 5, n_win)
starts, wlen = d["slide_starts"], float(d["slide_len"])
subs = np.array(sorted(np.unique(subj)))
folds = [subs[ix] for ix in np.array_split(np.random.default_rng(42).permutation(len(subs)), 5)]
aucs = np.full((len(starts), len(folds)), np.nan)
for wi in range(len(starts)):
    X = S[:, :, :, wi].reshape(len(y), -1)
    for s in subs:                                   # per-subject standardisation
        m = subj == s
        X[m] = (X[m] - X[m].mean(0)) / np.where(X[m].std(0) < 1e-12, 1, X[m].std(0))
    for fi, te in enumerate(folds):
        ite = np.isin(subj, te); itr = ~ite
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        mdl.fit(X[itr], y[itr])
        aucs[wi, fi] = roc_auc_score(y[ite], mdl.predict_proba(X[ite])[:, 1])
centres = starts + wlen / 2
mu, sd = aucs.mean(1), aucs.std(1)

fig, ax = plt.subplots(figsize=(11, 5.0))
for t, lab in [(2.0, "beep 1\n(both)"), (2.5, "beep 2\n(distractor)"), (4.0, "beep 3\n(distractor)")]:
    ax.axvline(t, color=MUTED, ls=":", lw=1.2)
    ax.text(t + 0.04, 0.735, lab, fontsize=8, color=INK2, va="top")
ax.axvspan(-0.5, 2.5, color=GRID, alpha=0.55, lw=0, zorder=0)
ax.text(1.0, 0.715, "conditions physically IDENTICAL here", fontsize=9,
        color=INK2, ha="center", style="italic")
ax.plot(centres, mu, color=SERIES[0], label="decoding AUC (5-fold, subject-independent)")
ax.fill_between(centres, mu - sd, mu + sd, color=SERIES[0], alpha=0.18, lw=0)
ax.axhline(0.5, color=INK2, ls="--", lw=1.2, label="chance")
ax.set_xlabel("time from trial start (s)   ·   500 ms sliding window")
ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.42, 0.75); ax.set_xlim(-0.55, 5.05)
ax.set_title("Time-resolved decoding — the class difference appears only after the extra beeps")
ax.legend(frameon=False, fontsize=9, loc="upper left")
style(ax)
fig.savefig(os.path.join(FIGD, "fig7_time_resolved.png")); plt.close(fig)
pd.DataFrame(dict(window_centre_s=centres, auc_mean=mu, auc_sd=sd)).to_csv(
    os.path.join(DERIV, "time_resolved_auc.csv"), index=False)

print("\ntime-resolved AUC (500 ms windows):")
for c, m_, s_ in zip(centres, mu, sd):
    bar = "#" * int(max(0, (m_ - 0.45)) * 100)
    print(f"  t={c:+5.2f}s  {m_:.3f} +- {s_:.3f}  {bar}")
print(f"\nfigures written to {FIGD}")
for f in sorted(os.listdir(FIGD)):
    print("  ", f)
