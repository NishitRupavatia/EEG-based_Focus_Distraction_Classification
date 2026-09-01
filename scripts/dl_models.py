"""
Deep models + the shared training loop.

Pulled out of 04_train.py so that the band sweep (07_bands.py) trains EXACTLY the
same networks with EXACTLY the same optimiser, early stopping and class weighting.
One definition, two callers - the two scripts cannot drift apart.

THE FOUR NETWORKS
  EEGNet        Lawhern et al. 2018   compact separable CNN               ~2.6 k params
  CTNet         Zhao et al. 2024      EEGNet patch-embed + transformer     ~25 k params
  LMDA-Net      Miao et al. 2023      channel/depth attention CNN           ~5 k params
  EEGConformer  Song et al. 2023      conv patch-embed + transformer      ~573 k params

A NOTE ON SAMPLING RATE
  Everything here is fed the same 125 Hz input that EEGNet has always used in this
  repo (250 Hz decimated by 2), because a fair comparison needs an identical input.
  The other three were published at 250 Hz, so every kernel and pooling factor
  below is specified in SECONDS and converted with SFREQ_DL - the layers keep the
  physical time scale of the papers rather than their raw sample counts.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

SFREQ_DL = 125.0          # rate the networks actually see (250 Hz decimated by 2)

# shared optimisation settings - identical for all three networks.
# The two schedule constants can be shortened from the environment so that a smoke
# test can drive the REAL orchestrator over the REAL grid in minutes instead of hours.
# Unset (the normal case) they are the published 300/40.
MAX_EPOCHS   = int(os.environ.get("EEG_MAX_EPOCHS", 300))
PATIENCE     = int(os.environ.get("EEG_PATIENCE", 40))
BATCH        = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.5


def _s2n(seconds, sfreq=SFREQ_DL, minimum=1):
    """Seconds -> odd-safe sample count, so a paper's kernel keeps its duration."""
    return max(minimum, int(round(seconds * sfreq)))


# ===================================================================== EEGNet
class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018). Unchanged from 04_train.py."""

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


# ====================================================================== CTNet
class _PatchEmbedding(nn.Module):
    """CTNet's front end: an EEGNet block whose output time steps become tokens.

    The two pooling stages compress 0.256 s of signal into one token, exactly as in
    the paper (8x8 at 250 Hz); at 125 Hz that is 8x4.
    """

    def __init__(self, n_ch, F1=8, D=2, F2=16, kern_s=0.256, pool1=8, pool2=4, drop=0.25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, _s2n(kern_s)), padding="same", bias=False),
            nn.BatchNorm2d(F1))
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, pool1)), nn.Dropout(drop))
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, pool2)), nn.Dropout(drop))

    def forward(self, x):                                  # (B, 1, C, T)
        z = self.separable(self.spatial(self.temporal(x)))  # (B, F2, 1, T')
        return z.squeeze(2).transpose(1, 2)                 # (B, T', F2) = tokens


class _TransformerBlock(nn.Module):
    """Pre-norm encoder block: MHSA + feed-forward, both residual."""

    def __init__(self, dim, heads, mlp_ratio=4, drop=0.5):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.d1 = nn.Dropout(drop)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.d1(self.attn(h, h, h, need_weights=False)[0])
        return x + self.mlp(self.n2(x))


class CTNet(nn.Module):
    """CTNet (Zhao et al. 2024) - convolutional patch embedding + transformer encoder.

    The convolution learns local time-frequency structure; the transformer relates
    tokens that are hundreds of milliseconds apart, which is what a 3 s window with
    beeps at +0.5 s and +2.0 s actually needs.
    """

    def __init__(self, n_ch, n_samples, n_classes=2, F1=8, D=2, F2=16,
                 depth=6, heads=2, drop=0.5, emb_drop=0.25):
        super().__init__()
        self.embed = _PatchEmbedding(n_ch, F1=F1, D=D, F2=F2, drop=emb_drop)
        with torch.no_grad():
            n_tok = self.embed(torch.zeros(1, 1, n_ch, n_samples)).shape[1]
        self.n_tokens = n_tok
        self.pos = nn.Parameter(torch.zeros(1, n_tok, F2))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.pos_drop = nn.Dropout(emb_drop)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(F2, heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(F2)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(n_tok * F2, n_classes))

    def forward(self, x):
        z = self.pos_drop(self.embed(x) + self.pos)
        for b in self.blocks:
            z = b(z)
        return self.head(self.norm(z).flatten(1))


# =================================================================== LMDA-Net
class _EEGDepthAttention(nn.Module):
    """Attention across the feature-map (depth) axis, Miao et al. 2023.

    Pools away the channel axis, scores each depth slice with a 1-D convolution over
    depth, softmaxes, and rescales. Cheap, and it is the part of LMDA-Net that makes
    it transfer across subjects.
    """

    def __init__(self, W, C, k=7):
        super().__init__()
        self.C = C
        self.pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, (k, 1), padding=(k // 2, 0), bias=True)

    def forward(self, x):                       # (B, C, H, W)
        y = self.pool(x).transpose(-2, -3)      # (B, 1, C, W)
        y = torch.softmax(self.conv(y), dim=-2).transpose(-2, -3)
        return y * self.C * x


class LMDANet(nn.Module):
    """LMDA-Net (Miao et al. 2023) - lightweight multi-dimensional attention.

    A learnable (depth x 1 x n_ch) tensor lifts the single-plane input into `depth`
    channel-weighted views before any convolution, so the network chooses its own
    montage instead of being handed one.
    """

    def __init__(self, n_ch, n_samples, n_classes=2, depth=9, kern_s=0.30,
                 ch_depth1=24, ch_depth2=9, avepool=3, drop=0.65):
        super().__init__()
        kernel = _s2n(kern_s)
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, n_ch))
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, ch_depth1, (1, 1), bias=False),
            nn.BatchNorm2d(ch_depth1),
            nn.Conv2d(ch_depth1, ch_depth1, (1, kernel), groups=ch_depth1, bias=False),
            nn.BatchNorm2d(ch_depth1), nn.GELU())
        self.channel_conv = nn.Sequential(
            nn.Conv2d(ch_depth1, ch_depth2, (1, 1), bias=False),
            nn.BatchNorm2d(ch_depth2),
            nn.Conv2d(ch_depth2, ch_depth2, (n_ch, 1), groups=ch_depth2, bias=False),
            nn.BatchNorm2d(ch_depth2), nn.GELU())
        self.norm = nn.Sequential(nn.AvgPool3d((1, 1, avepool)), nn.Dropout(drop))

        with torch.no_grad():
            z = torch.einsum("bdcw,hdc->bhcw",
                             torch.ones(1, 1, n_ch, n_samples), self.channel_weight)
            z = self.time_conv(z)
            self.depth_attention = _EEGDepthAttention(z.shape[-1], z.shape[1], k=7)
            z = self.norm(self.channel_conv(z))
            self.n_flat = int(np.prod(z.shape[1:]))
        self.classifier = nn.Linear(self.n_flat, n_classes)

    def forward(self, x):
        z = torch.einsum("bdcw,hdc->bhcw", x, self.channel_weight)
        z = self.depth_attention(self.time_conv(z))
        z = self.norm(self.channel_conv(z))
        return self.classifier(z.flatten(1))


# =============================================================== EEG Conformer
class EEGConformer(nn.Module):
    """EEG Conformer (Song et al. 2023) - conv patch embedding + transformer encoder.

    Same two-stage idea as CTNet but a much wider convolutional front end: 40 temporal
    filters and a single large average pool with a SHORT STRIDE, so the patches overlap
    rather than tile. That overlap is the point - it gives the encoder a finely sampled
    view of when things happen, which is what a transient buried in a 3 s window needs.

    Published at 250 Hz with a 25-sample kernel, 75-sample pool and 15-sample stride;
    expressed here in seconds (0.10 / 0.30 / 0.06 s) so the physical scale survives the
    move to 125 Hz. The encoder block is the same pre-norm MHSA+FFN used by CTNet, which
    is exactly what the paper's ResidualAdd(LayerNorm -> ...) pairs amount to.

    Far and away the largest model in the set: the classifier head alone dominates the
    parameter count. Left as published rather than trimmed to match the others, since
    shrinking it would no longer be the architecture being compared.
    """

    def __init__(self, n_ch, n_samples, n_classes=2, emb=40, depth=6, heads=10,
                 kern_s=0.10, pool_s=0.30, stride_s=0.06, drop=DROPOUT):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, emb, (1, _s2n(kern_s)), bias=False),
            nn.Conv2d(emb, emb, (n_ch, 1), bias=False),
            nn.BatchNorm2d(emb), nn.ELU(),
            nn.AvgPool2d((1, _s2n(pool_s)), stride=(1, _s2n(stride_s))),
            nn.Dropout(drop))
        self.proj = nn.Conv2d(emb, emb, (1, 1))          # token projection
        self.blocks = nn.ModuleList(
            [_TransformerBlock(emb, heads, drop=drop) for _ in range(depth)])
        with torch.no_grad():
            z = self.proj(self.embed(torch.zeros(1, 1, n_ch, n_samples)))
            self.n_tokens = z.shape[-1]
            self.n_flat = self.n_tokens * emb
        # the paper's tapered head, sized from the dry run rather than hard-coded
        self.head = nn.Sequential(
            nn.Linear(self.n_flat, 256), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes))

    def forward(self, x):
        z = self.proj(self.embed(x))            # (B, emb, 1, T')
        z = z.squeeze(2).transpose(1, 2)        # (B, T', emb) = tokens
        for b in self.blocks:
            z = b(z)
        return self.head(z.flatten(1))


# =============================================================== the registry
# Name -> constructor(n_ch, n_samples). 04_train.py and 07_bands.py both read this,
# so adding a network here adds it to both the main table and the band sweep.
DL_MODELS = {
    "EEGNet":        lambda n_ch, n_t: EEGNet(n_ch, n_t),
    "CTNet":         lambda n_ch, n_t: CTNet(n_ch, n_t),
    "LMDA-Net":      lambda n_ch, n_t: LMDANet(n_ch, n_t),
    "EEG Conformer": lambda n_ch, n_t: EEGConformer(n_ch, n_t),
}


# ============================================================== training loop
def _run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    tot = correct = 0
    loss_sum = 0.0
    for xb, yb in loader:
        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = crit(out, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
        loss_sum += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        tot += len(yb)
    return loss_sum / tot, correct / tot


def train_dl(name, Xtr, ytr, Xva, yva, Xte, tag, seed=42,
             max_epochs=None, patience=None):
    """Train one network with early stopping on validation loss.

    Returns (test probabilities, per-epoch history, fitted model, best epoch, stop epoch).
    Identical protocol for every architecture - class-weighted cross-entropy, Adam,
    restore the best-validation-loss weights before predicting.

    max_epochs/patience default to the module constants and are read at CALL time, not
    at import time, so a smoke test can shorten them by setting dl_models.MAX_EPOCHS.
    """
    max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
    patience = PATIENCE if patience is None else patience
    torch.manual_seed(seed)
    n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
    model = DL_MODELS[name](n_ch, n_t)

    cw = torch.tensor([len(ytr) / (2 * max((ytr == c).sum(), 1)) for c in (0, 1)],
                      dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def mk(X, y=None, shuf=False):
        # from_numpy VIEWS the array instead of copying it. torch.tensor(X[:, None])
        # duplicated every set in memory, which this 8 GB laptop does not have to spare.
        t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
        yy = torch.tensor(y if y is not None else np.zeros(len(X)), dtype=torch.long)
        return DataLoader(TensorDataset(t, yy), batch_size=BATCH, shuffle=shuf)

    tr_dl, va_dl = mk(Xtr, ytr, True), mk(Xva, yva)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, max_epochs + 1):
        trl, tra = _run_epoch(model, tr_dl, crit, opt)
        val, vaa = _run_epoch(model, va_dl, crit)
        hist.append(dict(model=name, tag=tag, epoch=ep, train_loss=trl, val_loss=val,
                         train_acc=tra, val_acc=vaa))
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    stopped = ep
    best_ep = int(np.argmin([h["val_loss"] for h in hist])) + 1
    # Predict in batches. One forward pass over ~1,000 test trials would allocate a
    # (1000, 24, 15, 375) activation inside LMDA-Net - over half a gigabyte for a
    # single intermediate - and that spike is enough to kill the process outright.
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(Xte), BATCH):
            xb = torch.from_numpy(
                np.ascontiguousarray(Xte[i:i + BATCH], dtype=np.float32)).unsqueeze(1)
            chunks.append(torch.softmax(model(xb), 1)[:, 1].numpy())
    p = np.concatenate(chunks)
    for h in hist:
        h["best_epoch"] = best_ep
        h["stopped_epoch"] = stopped
    return p, hist, model, best_ep, stopped


def n_params(name, n_ch, n_t):
    m = DL_MODELS[name](n_ch, n_t)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
