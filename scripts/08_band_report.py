"""
STEP 8 - Figures and tables for the band sweep.

Reads what 07_bands.py saved and draws:
  fig8   the band table          models x bands, accuracy, rendered as a figure
  fig9   band comparison bars    accuracy + AUC per band, grouped by model
  fig10  band heatmap            the same numbers as magnitude, models x bands
  fig11  control-window guardrail every model on the 0-2 s window, must be ~0.50

and writes derivatives/band_table.md for pasting into a report.

Why a table AND a heatmap AND bars: the table is the number of record, the bars
answer "which band wins", the heatmap answers "is the pattern the same for every
architecture". Three jobs, three forms - not the same chart three times.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Both redirectable, so the figures can be regenerated from a trial set of results
# without overwriting the published ones. Unset, they are the normal project folders.
DERIV = os.environ.get("EEG_DERIV_DIR") or os.path.join(ROOT, "derivatives")
FIGD  = os.environ.get("EEG_FIG_DIR") or os.path.join(ROOT, "figures")
os.makedirs(FIGD, exist_ok=True)

# ---- same validated palette as 05_report.py, same slots for the same models ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948", "#8c5a30"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8981"
SURFACE, GRID = "#fcfcfb", "#e5e4e0"
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

MODEL_ORDER = ["LR-bandpower", "LDA-shrinkage", "CSP+LDA", "Riemann-TS", "Xdawn+TS",
               "EEGNet", "CTNet", "LMDA-Net", "EEG Conformer"]
COLOR = dict(zip(MODEL_ORDER, SERIES))

BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma", "full"]
PRETTY = {"delta": "Delta", "theta": "Theta", "alpha": "Alpha", "beta": "Beta",
          "gamma": "Gamma", "full": "Full Band"}
RANGE = {"delta": "1–4 Hz", "theta": "4–8 Hz", "alpha": "8–13 Hz", "beta": "13–30 Hz",
         "gamma": "30–45 Hz", "full": "1–45 Hz"}

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


R = pd.read_csv(os.path.join(DERIV, "band_results.csv"))
B = R[R.condition != "control"]
models = [m for m in MODEL_ORDER if m in set(B.model)]
bands = [b for b in BAND_ORDER if b in set(B.condition)]


def grid_of(metric, how="mean"):
    return (B.pivot_table(index="model", columns="condition", values=metric, aggfunc=how)
             .reindex(index=models, columns=bands))


ACC, ACC_SD = grid_of("accuracy"), grid_of("accuracy", "std")
AUC, AUC_SD = grid_of("auc"), grid_of("auc", "std")

# ============================================================ fig 8  the table
# Deliberately a plain grid, because this is the number of record - no colour ranking,
# no bars behind the text. The only ink beyond the values marks each model's best band.
nrow, ncol = len(models), len(bands)
cw, ch = 1.0, 1.0
fig, ax = plt.subplots(figsize=(1.55 * (ncol + 1.6), 0.62 * (nrow + 1.9)))
ax.set_xlim(-1.6, ncol); ax.set_ylim(-nrow - 0.55, 1.15); ax.axis("off")

for j, b in enumerate(bands):                                     # header
    ax.add_patch(Rectangle((j, 0.02), cw, 1.0, facecolor="#f4f3f0",
                           edgecolor=INK, lw=1.0, zorder=1))
    ax.text(j + 0.5, 0.68, PRETTY[b], ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK, zorder=3)
    ax.text(j + 0.5, 0.28, RANGE[b], ha="center", va="center", fontsize=7.5,
            color=MUTED, zorder=3)
ax.add_patch(Rectangle((-1.6, 0.02), 1.6, 1.0, facecolor="#f4f3f0",
                       edgecolor=INK, lw=1.0, zorder=1))
ax.text(-0.8, 0.52, "Accuracy", ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=INK2, zorder=3)

for i, m in enumerate(models):
    y = -i - 1.0
    ax.add_patch(Rectangle((-1.6, y + 0.02), 1.6, 1.0, facecolor=SURFACE,
                           edgecolor=INK, lw=1.0, zorder=1))
    ax.text(-1.5, y + 0.52, m, ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=INK, zorder=3)
    best = ACC.loc[m].idxmax() if ACC.loc[m].notna().any() else None
    for j, b in enumerate(bands):
        v, sd = ACC.loc[m, b], ACC_SD.loc[m, b]
        hit = (b == best)
        ax.add_patch(Rectangle((j, y + 0.02), cw, 1.0,
                               facecolor="#eef4fd" if hit else SURFACE,
                               edgecolor=INK, lw=1.0, zorder=1))
        if pd.isna(v):
            ax.text(j + 0.5, y + 0.52, "—", ha="center", va="center",
                    fontsize=10, color=MUTED, zorder=3)
            continue
        ax.text(j + 0.5, y + 0.66, f"{100*v:.2f}%", ha="center", va="center",
                fontsize=10.5, fontweight="bold" if hit else "normal",
                color=SERIES[0] if hit else INK, zorder=3)
        ax.text(j + 0.5, y + 0.26, f"± {100*sd:.1f}", ha="center", va="center",
                fontsize=7.5, color=MUTED, zorder=3)

ax.text(-1.6, -nrow - 0.30,
        "Subject-independent 5-fold cross-validation · mean ± sd over folds · "
        "chance = 50.00% · shaded cell = that model's best band",
        ha="left", va="center", fontsize=8, color=MUTED)
fig.savefig(os.path.join(FIGD, "fig8_band_table.png")); plt.close(fig)

# ============================================================ fig 9  grouped bars
fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))
rot = 0 if len(models) <= 4 else 90          # keep labels horizontal while they fit
for ax, (G, Gsd, name) in zip(axes, [(ACC, ACC_SD, "Accuracy"), (AUC, AUC_SD, "ROC-AUC")]):
    x = np.arange(len(bands)); w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        off = (k - (len(models) - 1) / 2) * w
        mu, sd = G.loc[m].values, Gsd.loc[m].values
        ax.bar(x + off, mu, w * 0.86, yerr=sd, color=COLOR[m], capsize=2, zorder=3,
               ecolor=INK2, error_kw=dict(lw=0.8), label=m if name == "Accuracy" else None)
        # direct labels, clear of the error bar - the bars are thin and close, so a
        # value sitting on top of a whisker is worse than no value at all
        for xi, v, s in zip(x + off, mu, np.nan_to_num(sd)):
            if not np.isnan(v):
                ax.text(xi, v + s + 0.010, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7, color=INK2, rotation=rot, zorder=5)
    ax.axhline(0.5, color=INK2, ls="--", lw=1.2, zorder=4)
    ax.text(len(bands) - 0.45, 0.507, "chance", fontsize=8, color=INK2, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{PRETTY[b]}\n{RANGE[b]}" for b in bands])
    top = float(np.nanmax(G.values + np.nan_to_num(Gsd.values)))
    ax.set_ylim(0.45, min(1.0, top + 0.09)); ax.set_ylabel(name); ax.set_title(name)
    style(ax)
axes[0].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
fig.suptitle("Decoding by frequency band  ·  subject-independent 5-fold, 23 subjects",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig9_band_comparison.png")); plt.close(fig)

# ============================================================ fig 10  heatmap
# Magnitude, so one hue light->dark. Scaled from chance, not from 0: the interesting
# range is 0.50-0.85 and a 0-1 scale would flatten every difference in it.
fig, axes = plt.subplots(1, 2, figsize=(13.5, 1.05 * len(models) + 2.6))
for ax, (G, name) in zip(axes, [(ACC, "Accuracy"), (AUC, "ROC-AUC")]):
    M = G.values.astype(float)
    lo, hi = 0.50, np.nanmax(M)
    im = ax.imshow(M, cmap=SEQ, vmin=lo, vmax=hi, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED); continue
            frac = (M[i, j] - lo) / max(hi - lo, 1e-9)
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if frac > 0.55 else INK)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([PRETTY[b] for b in bands], rotation=15, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    ax.set_title(f"{name} — chance ({lo:.2f}) is the palest step")
    # 2 px surface gap between cells: adjacent fills of a similar step otherwise read
    # as one block, which is exactly what this figure is meant to let you compare
    ax.set_xticks(np.arange(-.5, len(bands), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.grid(False, which="major")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
fig.suptitle("Band × model decoding", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "fig10_band_heatmap.png")); plt.close(fig)

# ============================================================ fig 11  guardrail
C = R[R.condition == "control"]
if len(C):
    cm = [m for m in MODEL_ORDER if m in set(C.model)]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(cm)); w = 0.38
    for k, (src, lbl, alpha) in enumerate(
            [(B[B.condition == "full"], "Main window 2.0–5.0 s", 1.0),
             (C, "Control 0.0–2.0 s", 0.45)]):
        mu = [src[src.model == m].auc.mean() for m in cm]
        sd = [src[src.model == m].auc.std() for m in cm]
        cols = [COLOR[m] for m in cm] if k == 0 else [MUTED] * len(cm)
        ax.bar(x + (k - 0.5) * w, mu, w * 0.94, yerr=sd, color=cols, alpha=alpha,
               capsize=3, zorder=3, ecolor=INK2, error_kw=dict(lw=1), label=lbl)
        for xi, v, s in zip(x + (k - 0.5) * w, mu, np.nan_to_num(sd)):
            if not np.isnan(v):
                ax.text(xi, v + s + 0.012, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color=INK2 if k == 0 else MUTED, zorder=5)
    ax.axhline(0.5, color=INK2, ls="--", lw=1.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(cm, rotation=15, ha="right")
    ax.set_ylim(0.40, 0.90); ax.set_ylabel("ROC-AUC")
    ax.set_title("Guardrail — on the control window the conditions are physically\n"
                 "identical, so every network must sit at chance")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.savefig(os.path.join(FIGD, "fig11_band_control.png")); plt.close(fig)

# ================================================================ markdown table
lines = ["# Decoding accuracy by frequency band", "",
         "Subject-independent 5-fold cross-validation, 23 subjects, 4,544 trials.",
         "Mean ± sd over folds. Chance = 50.0%.", "",
         "| Model | " + " | ".join(f"{PRETTY[b]}<br><sub>{RANGE[b]}</sub>" for b in bands) + " |",
         "|---|" + "---|" * len(bands)]
for m in models:
    cells = []
    best = ACC.loc[m].idxmax() if ACC.loc[m].notna().any() else None
    for b in bands:
        v, sd = ACC.loc[m, b], ACC_SD.loc[m, b]
        cells.append("—" if pd.isna(v) else
                     (f"**{100*v:.2f}%** ± {100*sd:.1f}" if b == best
                      else f"{100*v:.2f}% ± {100*sd:.1f}"))
    lines.append(f"| **{m}** | " + " | ".join(cells) + " |")

lines += ["", "## ROC-AUC", "",
          "| Model | " + " | ".join(PRETTY[b] for b in bands) + " |",
          "|---|" + "---|" * len(bands)]
for m in models:
    lines.append(f"| **{m}** | " + " | ".join(
        "—" if pd.isna(AUC.loc[m, b]) else f"{AUC.loc[m,b]:.3f} ± {AUC_SD.loc[m,b]:.3f}"
        for b in bands) + " |")

if len(C):
    lines += ["", "## Control window 0.0–2.0 s (must be ~0.50)", "",
              "| Model | Accuracy | ROC-AUC |", "|---|---|---|"]
    for m in [x for x in MODEL_ORDER if x in set(C.model)]:
        g = C[C.model == m]
        lines.append(f"| **{m}** | {100*g.accuracy.mean():.2f}% | {g.auc.mean():.3f} |")

with open(os.path.join(DERIV, "band_table.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

# ===================================================================== console
print("\nACCURACY  (subject-independent 5-fold mean)")
print(f"{'model':<15}" + "".join(f"{PRETTY[b]:>12}" for b in bands))
for m in models:
    print(f"{m:<15}" + "".join(
        "           —" if pd.isna(ACC.loc[m, b]) else f"{100*ACC.loc[m,b]:>11.2f}%"
        for b in bands))
print("\nfigures written to", FIGD)
for f in ["fig8_band_table.png", "fig9_band_comparison.png",
          "fig10_band_heatmap.png", "fig11_band_control.png"]:
    if os.path.exists(os.path.join(FIGD, f)):
        print("  ", f)
print("table written to", os.path.join(DERIV, "band_table.md"))
