"""
ConformerTransformer - conv-conditioned local attention, then global attention.

WHY A SIXTH NETWORK
  The five networks already in this repo cover three of the four possible ways to
  combine convolution and attention:

    EEGNet, LMDA-Net    convolution only (attention over feature maps at most)
    CTNet, EEG Conformer  convolution THEN attention, in series
    EEG Transformer     attention only, no convolution at all  -> AUC 0.521, chance

  None of them puts a convolution INSIDE the attention block. That is what a Conformer
  block (Gulati et al. 2020, speech) does: every block is
  half-FFN -> multi-head self-attention -> depthwise convolution -> half-FFN, so each
  layer refines a global view and a local one together instead of doing all the local
  work up front and then throwing attention at the result.

  This file is that fourth cell of the table. The pipeline is

    trial -> conv tokenizer -> Conformer blocks -> reduce/project -> Transformer -> head

  with the Conformer stage running at fine temporal resolution (36 tokens, 80 ms apart)
  and the Transformer stage running on a reduced, narrower sequence (12 tokens, 240 ms
  apart) where whole-trial relations are cheap.

WHAT THE DATASET DICTATES, AND WHERE THE NUMBERS COME FROM
  * TOKENS ARE TIME, NEVER CHANNELS. Four independent lines of evidence in the README
    say the effect is a time-locked evoked response, not an oscillatory or spatial
    state change. The 15 electrodes are collapsed by a depthwise spatial convolution
    in the tokenizer, exactly as in EEGNet - the front end that already scores 0.798
    here. After average referencing the montage is rank 14 anyway, so 15 channel
    tokens would carry one redundant dimension.

  * PATCHES OVERLAP. Beeps sit at +0.0, +0.5 and +2.0 s inside the 2.0-5.0 s window and
    decoding peaks ~250 ms after each one. A 200 ms patch every 80 ms resolves a P2/P3
    across ~3 tokens and guarantees no transient falls between two patches.
    Non-overlapping 200 ms patches would give 15 tokens - which is the exact
    configuration EEG Transformer used to score chance.

  * CAPACITY IS THE ENEMY. With ~2,800 training trials this repo's own results are
    monotone the wrong way: 573 k params -> 0.771, 21 k -> 0.780, 1.7 k -> 0.798. So
    the FFN expansion is 2 (not the paper's 4), depth is 2+2 (not 6), and the readout
    is a CLS token whose head is 66 parameters. Total ~94 k, and the capacity sweep is
    the first ablation worth running.

  * PRE-NORM EVERYWHERE. The shared training loop is plain Adam at a fixed 1e-3 with no
    warmup (dl_models.train_dl). Post-norm transformers need that warmup to train
    stably; pre-norm ones do not. Changing the loop instead would make this network's
    number incomparable with the other five, which is the whole point of reusing it.

SHAPES  (main window: C=15, T=375, N=36, M=12)
  input                            (B, 1, 15, 375)  or  (B, 15, 375)
  temporal conv  k=13 (0.104 s)    (B, 16, 15, 375)
  spatial depthwise (15,1) x D=2   (B, 32,  1, 375)
  avg-pool 25 stride 10            (B, 32,  1,  36)
  token projection 32 -> 48        (B, 36, 48)        = (B, N, d_conf)
  Conformer blocks x2              (B, 36, 48)
  temporal reduction stride 3      (B, 12, 48)
  projection 48 -> 32              (B, 12, 32)        = (B, M, d_tf)
  prepend CLS + positional         (B, 13, 32)
  Transformer blocks x2            (B, 13, 32)
  CLS token                        (B, 32)
  classification head              (B, 2)             logits for CrossEntropyLoss

  Control window (T=250) gives N=23, M=7, 8 tokens. Because the readout is the CLS
  token and not a flatten, the PARAMETER COUNT IS THE SAME in both windows bar the
  positional table (160 values) - so the 0-2 s guardrail is run by the same model
  rather than by a differently-sized one, which is not true of a flatten head.

POSITIONAL ENCODING
  Learned absolute, and only at the Transformer stage. The Conformer stage needs none:
  its depthwise convolution supplies relative position implicitly, which is the entire
  premise of the block. Absolute rather than sinusoidal because every trial is the same
  length and time-locked to the same marker, so there is no length generalisation to
  preserve and position IS the signal - the opposite of context_transformer.py, where
  trials form a set and a positional embedding would be a bug.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SFREQ_DL   = 125.0        # rate the networks actually see (250 Hz decimated by 2)

# --- tokenizer ---------------------------------------------------------------
F1         = 16           # temporal filters
D          = 2            # spatial filters per temporal filter -> F1*D maps
KERN_S     = 0.104        # temporal kernel, seconds (13 samples at 125 Hz)
POOL_S     = 0.20         # patch length,   seconds (25 samples) - the "patch size"
HOP_S      = 0.08         # patch stride,   seconds (10 samples) - patches overlap 60%
MAX_NORM   = 1.0          # EEGNet's cap on the spatial filter norms

# --- Conformer stage ---------------------------------------------------------
D_CONF     = 48
CONF_DEPTH = 2
CONV_K     = 9            # depthwise kernel over tokens = 0.72 s of context
FFN_RATIO  = 2            # the paper uses 4; 2 halves the block and loses nothing here

# --- reduction + Transformer stage ------------------------------------------
REDUCE_R   = 3            # 36 tokens -> 12
D_TF       = 32
TF_DEPTH   = 2
MLP_RATIO  = 2

N_HEADS    = 4            # 48/4 = 12 and 32/4 = 8 dims per head
DROPOUT    = 0.3
HEAD_DROP  = 0.5
POOL       = "cls"


def _s2n(seconds, sfreq=SFREQ_DL, minimum=1):
    """Seconds -> sample count, so every kernel keeps its physical duration."""
    return max(minimum, int(round(seconds * sfreq)))


# =============================================================== tokenizer
class ConvTokenizer(nn.Module):
    """Turn (B, 1, C, T) into a sequence of (B, N, d) tokens with a conv front end.

    Three steps, each doing one job:
      1. temporal convolution   learns a filter bank - what shapes to look for
      2. spatial depthwise      learns a montage     - where to look for them
      3. overlapping avg-pool   makes the patches    - when each one happened

    Step 3 is the "patch creation" the architecture calls for, done with a pooling
    stride rather than a tiling reshape so that consecutive patches overlap. The
    alternative - reshaping T into (N, patch) - is what eeg_transformer.py does, and
    it scores at chance on this dataset.
    """

    def __init__(self, n_ch, dim=D_CONF, f1=F1, d=D, kern_s=KERN_S,
                 pool_s=POOL_S, hop_s=HOP_S, drop=DROPOUT):
        super().__init__()
        self.kern, self.patch, self.hop = _s2n(kern_s), _s2n(pool_s), _s2n(hop_s)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, self.kern), padding="same", bias=False),
            nn.BatchNorm2d(f1))
        self.spatial = nn.Conv2d(f1, f1 * d, (n_ch, 1), groups=f1, bias=False)
        self.post = nn.Sequential(
            nn.BatchNorm2d(f1 * d), nn.ELU(),
            nn.AvgPool2d((1, self.patch), stride=(1, self.hop)),
            nn.Dropout(drop))
        self.proj = nn.Linear(f1 * d, dim)
        self.norm = nn.LayerNorm(dim)

    def n_patches(self, n_samples):
        """How many tokens a window of n_samples produces. Pooling arithmetic, not a guess."""
        return (n_samples - self.patch) // self.hop + 1

    def forward(self, x):                                  # (B, 1, C, T)
        z = self.post(self.spatial(self.temporal(x)))       # (B, F1*D, 1, N)
        z = z.squeeze(2).transpose(1, 2)                    # (B, N, F1*D)
        return self.norm(self.proj(z))                      # (B, N, dim)

    @torch.no_grad()
    def apply_max_norm(self, max_norm=MAX_NORM):
        """Cap the L2 norm of each spatial filter, as EEGNet does.

        Without it the spatial convolution is free to grow one electrode's weight
        without bound, which is how a cross-subject model latches onto a single
        person's bad channel.
        """
        w = self.spatial.weight
        n = w.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        w.mul_(n.clamp(max=max_norm) / n)


# ========================================================== conformer block
class _FeedForward(nn.Module):
    """Pre-norm FFN. Used twice per Conformer block, each at half weight."""

    def __init__(self, dim, ratio=FFN_RATIO, drop=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * ratio), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(dim * ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class _ConvModule(nn.Module):
    """The Conformer block's local half: GLU -> depthwise conv -> BN -> SiLU -> pointwise.

    This is the module that distinguishes this network from CTNet and EEG Conformer.
    They convolve once, before any attention, and then attend; here a depthwise
    convolution over the token axis sits INSIDE every block, so local shape and global
    context are refined together at every depth. Ablation A2 - deleting exactly this
    module - is what measures whether that matters on this dataset.

    Dropout here is deliberately light (0.1): this path is already heavily constrained
    by being depthwise, and the two FFNs either side of it carry the full rate.
    """

    def __init__(self, dim, kernel=CONV_K, drop=0.1):
        super().__init__()
        assert kernel % 2 == 1, "odd kernel keeps the token sequence length unchanged"
        self.norm = nn.LayerNorm(dim)
        self.pw_in = nn.Conv1d(dim, 2 * dim, 1)            # 2x for the GLU gate
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw_out = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                                  # (B, N, dim)
        z = self.norm(x).transpose(1, 2)                    # (B, dim, N) for Conv1d
        z = F.glu(self.pw_in(z), dim=1)
        z = self.pw_out(F.silu(self.bn(self.depthwise(z))))
        return self.drop(z.transpose(1, 2))                 # (B, N, dim)


class ConformerBlock(nn.Module):
    """Macaron block: 1/2 FFN -> MHSA -> conv -> 1/2 FFN -> LayerNorm, all residual.

    The half-weight FFNs sandwiching the attention and convolution are Gulati's
    "macaron" arrangement, and they are not decoration: a full-weight FFN on each side
    would double this block's parameter count, which is the budget this dataset does
    not have.
    """

    def __init__(self, dim, heads=N_HEADS, ffn_ratio=FFN_RATIO, kernel=CONV_K,
                 drop=DROPOUT):
        super().__init__()
        self.ff1 = _FeedForward(dim, ffn_ratio, drop)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.attn_drop = nn.Dropout(drop)
        self.conv = _ConvModule(dim, kernel)
        self.ff2 = _FeedForward(dim, ffn_ratio, drop)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, need_weights=False):
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + self.attn_drop(a)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.out_norm(x)
        return (x, w) if need_weights else (x, None)


# ======================================================= reduction / projection
class Reduction(nn.Module):
    """Shorten the sequence and narrow the model dimension before global attention.

    Two reductions in one stage:
      TOKENS   a depthwise stride-r convolution, 36 -> 12. Learned rather than a plain
               average pool, for 144 parameters - a token is an ERP fragment and which
               part of the fragment matters is worth learning.
      WIDTH    Linear(d_conf -> d_tf) + LayerNorm.

    A trailing partial group is cropped rather than padded, so nothing invents signal
    at the end of the window. On the main window 36 divides by 3 exactly; on the
    control window 23 does not, and the last 2 tokens (160 ms) are dropped.
    """

    def __init__(self, d_in=D_CONF, d_out=D_TF, r=REDUCE_R):
        super().__init__()
        self.r = r
        self.pool = (nn.Conv1d(d_in, d_in, r, stride=r, groups=d_in) if r > 1
                     else nn.Identity())
        self.proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def n_tokens(self, n_in):
        return n_in // self.r

    def forward(self, x):                                  # (B, N, d_in)
        if self.r > 1:
            n = (x.shape[1] // self.r) * self.r             # crop a partial group
            x = self.pool(x[:, :n].transpose(1, 2)).transpose(1, 2)
        return self.norm(self.proj(x))                      # (B, M, d_out)


# ============================================================ transformer block
class TransformerBlock(nn.Module):
    """Plain pre-norm encoder block: MHSA + MLP, both residual. No convolution.

    Identical in structure to the blocks in dl_models.py and eeg_transformer.py, on
    purpose: the difference between this stage and the Conformer stage above should be
    the conv module and nothing else, or the ablation measures two things at once.
    """

    def __init__(self, dim, heads=N_HEADS, mlp_ratio=MLP_RATIO, drop=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + self.drop1(a)
        x = x + self.mlp(self.norm2(x))
        return (x, w) if need_weights else (x, None)


# ==================================================================== the model
class ConformerTransformer(nn.Module):
    """Conv tokenizer -> Conformer -> reduce -> Transformer -> head. Focus (0) vs. Distractor (1).

    Constructor signature matches every other network in this repo - (n_ch, n_samples) -
    so it drops straight into dl_models.DL_MODELS and the shared training loop with no
    special-casing. forward() accepts (B, 1, C, T), which is what the loop feeds, or a
    bare (B, C, T) so the model is usable on its own.

    `pool` selects the readout: 'cls' (default), 'mean', or 'flat'. The default is CLS
    because it keeps the head at 66 parameters and keeps the parameter count identical
    between the main and control windows; the other two exist so that choice is an
    ablation rather than an assumption.
    """

    def __init__(self, n_ch, n_samples, n_classes=2, f1=F1, d=D,
                 d_conf=D_CONF, conf_depth=CONF_DEPTH, conv_k=CONV_K,
                 ffn_ratio=FFN_RATIO, reduce_r=REDUCE_R, d_tf=D_TF,
                 tf_depth=TF_DEPTH, mlp_ratio=MLP_RATIO, heads=N_HEADS,
                 drop=DROPOUT, head_drop=HEAD_DROP, pool=POOL, pos_enc="learned"):
        super().__init__()
        assert pool in ("cls", "mean", "flat"), "pool must be 'cls', 'mean' or 'flat'"
        assert pos_enc in ("learned", "none"), "pos_enc must be 'learned' or 'none'"
        self.n_ch, self.n_samples = n_ch, n_samples
        self.pool, self.pos_enc = pool, pos_enc
        self.d_conf, self.d_tf = d_conf, d_tf
        self.heads = heads

        self.tokenizer = ConvTokenizer(n_ch, dim=d_conf, f1=f1, d=d, drop=drop)
        self.patch, self.hop = self.tokenizer.patch, self.tokenizer.hop
        assert n_samples >= self.patch, (
            f"window of {n_samples} samples is shorter than one {self.patch}-sample patch")

        self.conformer = nn.ModuleList(
            [ConformerBlock(d_conf, heads, ffn_ratio, conv_k, drop)
             for _ in range(conf_depth)])
        self.reduce = Reduction(d_conf, d_tf, reduce_r)

        # Token counts are computed from the layer arithmetic and then CHECKED against
        # a real forward pass in _verify_shapes(), so a mis-specified stride fails here
        # rather than silently producing a differently-sized head.
        self.n_patches = self.tokenizer.n_patches(n_samples)
        self.n_reduced = self.reduce.n_tokens(self.n_patches)
        assert self.n_reduced >= 1, (
            f"{self.n_patches} patches reduced by {reduce_r} leaves nothing")
        self.n_tokens = self.n_reduced + (1 if pool == "cls" else 0)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_tf)) if pool == "cls" else None
        if pos_enc == "learned":
            self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_tf))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.pos = None
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_tf, heads, mlp_ratio, drop) for _ in range(tf_depth)])
        self.norm = nn.LayerNorm(d_tf)
        n_in = self.n_tokens * d_tf if pool == "flat" else d_tf
        self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear(n_in, n_classes))
        self._verify_shapes()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _canon(x):
        """-> (B, 1, C, T). Accepts (B, 1, C, T) from the shared loop or a bare (B, C, T)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 4 and x.shape[1] == 1, \
            f"expected (B, C, T) or (B, 1, C, T), got {tuple(x.shape)}"
        return x

    def _tokens(self, x):
        """(B, [1,] C, T) -> (B, M(+1), d_tf), positional information added."""
        z = self.tokenizer(self._canon(x))                  # (B, N, d_conf)
        for blk in self.conformer:
            z, _ = blk(z)                                   # (B, N, d_conf)
        z = self.reduce(z)                                  # (B, M, d_tf)
        if self.cls is not None:
            z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], 1)
        if self.pos is not None:
            z = z + self.pos
        return self.pos_drop(z)

    def _readout(self, z):
        if self.pool == "flat":
            return z.flatten(1)
        return z[:, 0] if self.pool == "cls" else z.mean(1)

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        z = self._tokens(x)
        for blk in self.blocks:
            z, _ = blk(z)
        return self.head(self._readout(self.norm(z)))        # (B, 2) logits

    @torch.no_grad()
    def apply_max_norm(self):
        """Called by dl_models._run_epoch after every optimiser step, if present."""
        self.tokenizer.apply_max_norm()

    # ----------------------------------------------------------- verification
    @torch.no_grad()
    def _verify_shapes(self):
        """Run one real forward pass and assert the token counts were predicted right.

        Cheap insurance: every derived size in this class comes from pooling and stride
        arithmetic, and getting one of them wrong would build a head of the wrong width
        that still trains and still reports a plausible number.
        """
        was = self.training
        self.eval()
        x = torch.zeros(2, 1, self.n_ch, self.n_samples)
        n = self.tokenizer(x).shape[1]
        assert n == self.n_patches, f"predicted {self.n_patches} patches, forward gave {n}"
        z = self._tokens(x)
        assert z.shape[1:] == (self.n_tokens, self.d_tf), \
            f"predicted {(self.n_tokens, self.d_tf)} tokens, forward gave {tuple(z.shape[1:])}"
        assert self.forward(x).shape == (2, self.head[-1].out_features)
        self.train(was)

    @torch.no_grad()
    def attention_maps(self, x):
        """Per-stage attention. Returns dict of (depth, B, n, n) tensors. Diagnostics only.

        The two stages have different sequence lengths (N tokens vs M+1), so they cannot
        be stacked into one tensor - hence a dict rather than eeg_transformer.py's stack.
        """
        was = self.training
        self.eval()
        z = self.tokenizer(self._canon(x))
        conf = []
        for blk in self.conformer:
            z, w = blk(z, need_weights=True)
            conf.append(w)
        z = self.reduce(z)
        if self.cls is not None:
            z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], 1)
        if self.pos is not None:
            z = z + self.pos
        glob = []
        for blk in self.blocks:
            z, w = blk(z, need_weights=True)
            glob.append(w)
        self.train(was)
        return dict(conformer=torch.stack(conf) if conf else None,
                    transformer=torch.stack(glob) if glob else None)

    # ------------------------------------------------------------- description
    def shape_trace(self, batch=2):
        """Every intermediate shape at the real dimensions, for the report printout."""
        was = self.training
        self.eval()
        x = torch.zeros(batch, 1, self.n_ch, self.n_samples)
        rows = [("input                        [B, 1, C, T]", tuple(x.shape))]
        with torch.no_grad():
            t = self.tokenizer.temporal(x)
            rows.append((f"temporal conv k={self.tokenizer.kern:<3}     [B, F1, C, T]",
                         tuple(t.shape)))
            s = self.tokenizer.spatial(t)
            rows.append(("spatial depthwise          [B, F1*D, 1, T]", tuple(s.shape)))
            p = self.tokenizer.post(s)
            rows.append((f"patches {self.patch}/{self.hop}            [B, F1*D, 1, N]",
                         tuple(p.shape)))
            z = self.tokenizer(x)
            rows.append(("token projection          [B, N, d_conf]", tuple(z.shape)))
            for blk in self.conformer:
                z, _ = blk(z)
            rows.append((f"Conformer x{len(self.conformer)}              [B, N, d_conf]",
                         tuple(z.shape)))
            r = self.reduce(z)
            rows.append((f"reduce /{self.reduce.r} + project     [B, M, d_tf]",
                         tuple(r.shape)))
            f = self._tokens(x)
            tag = "+ CLS + positional" if self.cls is not None else "+ positional"
            rows.append((f"{tag:<26} [B, M+1, d_tf]", tuple(f.shape)))
            g = f
            for blk in self.blocks:
                g, _ = blk(g)
            rows.append((f"Transformer x{len(self.blocks)}            [B, M+1, d_tf]",
                         tuple(g.shape)))
            pooled = self._readout(self.norm(g))
            label = {"cls": "CLS token", "mean": "mean pool", "flat": "flatten"}[self.pool]
            rows.append((f"{label:<26} [B, d_tf]", tuple(pooled.shape)))
            rows.append(("classifier                      [B, 2]",
                         tuple(self.head(pooled).shape)))
        self.train(was)
        return rows

    def param_groups(self):
        n = lambda mod: sum(q.numel() for q in mod.parameters() if q.requires_grad)
        extra = sum(q.numel() for q in (self.cls, self.pos) if q is not None)
        g = dict(tokenizer=n(self.tokenizer), conformer=n(self.conformer),
                 reduce=n(self.reduce), transformer=n(self.blocks),
                 head=n(self.head) + n(self.norm) + extra)
        g["total"] = sum(g.values())
        assert g["total"] == sum(q.numel() for q in self.parameters() if q.requires_grad)
        return g

    def config(self):
        """Everything the job row needs to describe what was actually trained."""
        return dict(
            n_channels=self.n_ch, n_times=self.n_samples,
            patch_samples=self.patch, hop_samples=self.hop,
            patch_s=round(self.patch / SFREQ_DL, 4), hop_s=round(self.hop / SFREQ_DL, 4),
            n_patches=self.n_patches, n_reduced=self.n_reduced, n_tokens=self.n_tokens,
            d_conf=self.d_conf, d_tf=self.d_tf, conf_depth=len(self.conformer),
            tf_depth=len(self.blocks), heads=self.heads, conv_k=CONV_K,
            reduce_r=self.reduce.r, pool=self.pool, pos_enc=self.pos_enc,
            n_params=self.param_groups()["total"])


def build(n_ch, n_samples, **kw):
    """Registry-shaped constructor: build(n_ch, n_samples) -> ConformerTransformer."""
    return ConformerTransformer(n_ch, n_samples, **kw)


def count_params(n_ch=15, n_samples=375, **kw):
    return ConformerTransformer(n_ch, n_samples, **kw).param_groups()["total"]


if __name__ == "__main__":
    for n_ch, n_t, label in [(15, 375, "main window 2.0-5.0 s"),
                             (15, 250, "control window 0.0-2.0 s")]:
        m = ConformerTransformer(n_ch, n_t)
        print(f"\n=== {label}:  C={n_ch}, T={n_t} ===")
        for name, shp in m.shape_trace(batch=4):
            print(f"  {name:<44} {shp}")
        g = m.param_groups()
        print(f"  tokens: {m.n_patches} patches of {m.patch} samples "
              f"({m.patch / SFREQ_DL:.2f} s) every {m.hop} ({m.hop / SFREQ_DL:.2f} s)"
              f"  ->  {m.n_reduced} reduced  ->  {m.n_tokens} with CLS")
        print(f"  parameters: tokenizer {g['tokenizer']:,} + conformer {g['conformer']:,}"
              f" + reduce {g['reduce']:,} + transformer {g['transformer']:,}"
              f" + head {g['head']:,} = {g['total']:,}")

    # The readout and capacity variants the ablation plan calls for, so a change that
    # breaks one of them is caught here rather than five folds into a sweep.
    print("\n=== variants (main window) ===")
    for kw in [dict(), dict(pool="mean"), dict(pool="flat"), dict(pos_enc="none"),
               dict(conf_depth=0), dict(tf_depth=0), dict(reduce_r=1),
               dict(d_conf=32, d_tf=32, f1=8), dict(conf_depth=3, tf_depth=3),
               dict(ffn_ratio=4)]:
        m = ConformerTransformer(15, 375, **kw)
        desc = ", ".join(f"{k}={v}" for k, v in kw.items()) or "default"
        print(f"  {desc:<34} N={m.n_patches:>3} M={m.n_reduced:>3} "
              f"tok={m.n_tokens:>3}  {m.param_groups()['total']:>8,} params")
