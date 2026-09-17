"""
ContextTransformer - cross-TRIAL attention for Focus vs. Distractor.

THE IDEA, AND WHY IT IS DIFFERENT FROM EVERY OTHER MODEL IN THIS REPO
  Every network in dl_models.py and eeg_transformer.py classifies ONE trial in
  isolation: 3 s of 15-channel EEG in, one label out. Attention, where they have it,
  runs over slices of time WITHIN a trial.

  This model moves the attention up one level. An EEGNet-style convolutional encoder
  squeezes each complete trial into a single 64-d embedding, and the transformer then
  attends ACROSS a group of trials - a "context" of 64 of them - producing a label for
  every trial in the group at once.

  The bet is that a trial is easier to judge next to its neighbours than alone. A
  single 3 s epoch from an unseen person carries that person's amplitude scale, their
  electrode impedances and their arousal level baked into it, and the network has no
  way to know which part of what it sees is the effect and which is the person. Given
  64 trials from that person simultaneously, attention can subtract the common part.
  That is a per-subject calibration the model performs for itself at inference time,
  from unlabelled data, which is exactly the quantity a subject-independent split
  otherwise denies it.

NO POSITIONAL EMBEDDING
  Deliberate, and load-bearing. The trials in a context are a SET, not a sequence -
  they are drawn at random from one person's session, and trial 12 sitting next to
  trial 40 means nothing. Adding a positional embedding would invite the model to
  memorise slot identity, and would break the permutation equivariance that makes the
  output well defined: shuffle the input trials and the predictions follow them
  unchanged, which is the property a set model must have. There is no CLS token
  either, for the same reason - there is nothing to summarise, every token is an
  answer.

THE HONEST WORRY, AND THE ABLATION THAT MEASURES IT
  A set model handed 64 trials that happen to be ~50/50 can cheat: rank all 64 by some
  internal score and call the top half distractor. That would post a high AUC while
  learning nothing about any individual trial. Two things exist to expose it -
  contexts are composed at random so their class balance varies binomially rather than
  being pinned at 32/32, and 10_context_transformer.py re-scores the test set on
  deliberately skewed contexts (90/10, and pure single-class) where a median split is
  catastrophically wrong. context_size=1, which reduces this exact architecture to a
  plain per-trial classifier, is the other end of the same measurement.

SHAPES  (C=15, T=375, S=64, D=64)
  input                        (B, S, 1, C, T)     B contexts of S trials
  fold trials into the batch   (B*S, 1, C, T)
  EEGNet encoder               (B*S, D)
  unfold back to contexts      (B, S, D)           <- the transformer's sequence
  encoder blocks x2, no pos    (B, S, D)           attention is over TRIALS
  per-token head               (B, S, 2)           one logit pair per trial

  Padded slots in a short final context are excluded by a key-padding mask and never
  reach the loss, so a subject whose trial count is not a multiple of S is scored on
  exactly their real trials and nothing else.
"""
import numpy as np
import torch
import torch.nn as nn

SFREQ_DL   = 125.0        # rate the networks actually see (250 Hz decimated by 2)
EMB_DIM    = 64           # the trial embedding the task specifies
DEPTH      = 2            # encoder blocks over the trial axis
N_HEADS    = 4
MLP_RATIO  = 2
DROPOUT    = 0.5
KERN_LEN   = 64           # EEGNet temporal kernel, in samples - matches dl_models.EEGNet


# ================================================================= trial encoder
class EEGNetEncoder(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018) with the classifier replaced by a projection.

    Identical convolutional stack to dl_models.EEGNet - same F1/D/F2, same kernel
    length, same two pooling stages, same dropout - so the embedding is produced by
    the front end that already scores 0.794 on this dataset. The only change is the
    last layer: instead of Linear(n_flat, 2) it is Linear(n_flat, 64) + LayerNorm,
    turning a classifier into a feature extractor.

    LayerNorm on the output matters more than it looks. The transformer that consumes
    these embeddings is pre-norm, and feeding it vectors whose scale drifts per subject
    would hand it back the very nuisance variance the context is supposed to remove.
    """

    def __init__(self, n_ch, n_samples, dim=EMB_DIM, F1=8, D=2, F2=16,
                 kern_len=KERN_LEN, drop=DROPOUT):
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
        with torch.no_grad():                       # work out the flatten size for real
            z = self.block3(self.block2(self.depthwise(
                self.block1(torch.zeros(1, 1, n_ch, n_samples)))))
            self.n_flat = int(np.prod(z.shape[1:]))
        self.proj = nn.Linear(self.n_flat, dim)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x):                           # (N, 1, C, T)
        z = self.block3(self.block2(self.depthwise(self.block1(x))))
        return self.norm(self.proj(z.flatten(1)))   # (N, D)

    @torch.no_grad()
    def apply_max_norm(self):
        """EEGNet's own regulariser: cap the L2 norm of each spatial filter at 1."""
        w = self.depthwise.weight
        n = w.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        w.mul_(n.clamp(max=1.0) / n)


# ============================================================== attention blocks
class ContextBlock(nn.Module):
    """Pre-norm encoder block whose sequence axis is TRIALS, not time.

    Pre-norm rather than post-norm because the shared protocol is plain Adam at a fixed
    1e-3 with no warmup, and post-norm transformers need the warmup to train stably.

    `pad_mask` is True at slots that are padding. It is passed to the attention as
    key_padding_mask, so a real trial never attends to a filler slot. A padded QUERY
    still attends over the real keys and produces some vector, which is harmless - the
    loss and every metric drop those rows.
    """

    def __init__(self, dim, heads, mlp_ratio=MLP_RATIO, drop=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x, pad_mask=None, need_weights=False):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, key_padding_mask=pad_mask,
                         need_weights=need_weights, average_attn_weights=True)
        x = x + self.drop1(a)
        x = x + self.mlp(self.norm2(x))
        return (x, w) if need_weights else (x, None)


# ================================================================== the model
class ContextTransformer(nn.Module):
    """EEGNet encoder + self-attention across trials. One label per trial in the set.

    forward() accepts (B, S, C, T) or (B, S, 1, C, T) and returns (B, S, 2) logits.
    A bare (B, C, T) is also accepted and read as one context of B trials, so the model
    can be probed without constructing a context axis by hand.

    Permutation equivariance is a property of the architecture, not a claim: there is
    no positional embedding and no CLS token, attention is symmetric in its keys, and
    the head is applied per token. verify_equivariance() checks it numerically.
    """

    def __init__(self, n_ch, n_samples, n_classes=2, dim=EMB_DIM, depth=DEPTH,
                 heads=N_HEADS, mlp_ratio=MLP_RATIO, drop=DROPOUT, kern_len=KERN_LEN):
        super().__init__()
        self.n_ch, self.n_samples, self.dim, self.depth = n_ch, n_samples, dim, depth
        self.heads = heads
        self.encoder = EEGNetEncoder(n_ch, n_samples, dim=dim, kern_len=kern_len, drop=drop)
        self.blocks = nn.ModuleList(
            [ContextBlock(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, n_classes))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _canon(x):
        """-> (B, S, 1, C, T). Accepts (B,S,C,T), (B,S,1,C,T) or a bare (B,C,T)."""
        if x.dim() == 3:                    # (B, C, T): one context of B trials
            x = x.unsqueeze(0)
        if x.dim() == 4:                    # (B, S, C, T)
            x = x.unsqueeze(2)
        assert x.dim() == 5, f"expected (B,S,C,T) or (B,S,1,C,T), got {tuple(x.shape)}"
        return x

    def embed_trials(self, x):
        """(B, S, [1,] C, T) -> (B, S, D). The encoder never sees the context axis."""
        x = self._canon(x)
        B, S = x.shape[:2]
        z = self.encoder(x.reshape(B * S, *x.shape[2:]))    # fold S into the batch
        return z.view(B, S, self.dim)                        # and unfold it again

    # ------------------------------------------------------------------ forward
    def forward(self, x, pad_mask=None):
        """x: (B, S, [1,] C, T); pad_mask: (B, S) bool, True where the slot is filler."""
        z = self.embed_trials(x)                             # (B, S, D)
        for blk in self.blocks:
            z, _ = blk(z, pad_mask=pad_mask)                 # attention across trials
        return self.head(self.norm(z))                       # (B, S, 2)

    @torch.no_grad()
    def attention_maps(self, x, pad_mask=None):
        """Per-layer trial-to-trial attention, (depth, B, S, S). Diagnostics only."""
        was = self.training
        self.eval()
        z, maps = self.embed_trials(x), []
        for blk in self.blocks:
            z, w = blk(z, pad_mask=pad_mask, need_weights=True)
            maps.append(w)
        self.train(was)
        return torch.stack(maps)

    @torch.no_grad()
    def apply_max_norm(self):
        self.encoder.apply_max_norm()

    # ------------------------------------------------------------ verification
    @torch.no_grad()
    def verify_equivariance(self, n_trials=8, seed=0):
        """Shuffle the trials in a context; the predictions must follow, unchanged.

        This is the numerical stand-in for 'there is no positional embedding'. If
        anyone ever adds one, this returns a max deviation far above float noise and
        the report says so, instead of quietly presenting a sequence model as a set
        model.
        """
        was = self.training
        self.eval()
        # Probe tensors are drawn on the CPU (so the seed gives the same probe whether
        # or not there is a GPU) and then moved to wherever the weights actually live.
        dev = next(self.parameters()).device
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(2, n_trials, 1, self.n_ch, self.n_samples, generator=g).to(dev)
        perm = torch.randperm(n_trials, generator=g).to(dev)
        a = self.forward(x)[:, perm]
        b = self.forward(x[:, perm])
        self.train(was)
        return float((a - b).abs().max())

    @torch.no_grad()
    def verify_padding(self, n_real=4, n_pad=2, seed=0):
        """A short final context must give its real trials the same answers."""
        was = self.training
        self.eval()
        dev = next(self.parameters()).device
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(1, n_real + n_pad, 1, self.n_ch, self.n_samples, generator=g).to(dev)
        pm = torch.zeros(1, n_real + n_pad, dtype=torch.bool, device=dev)
        pm[:, n_real:] = True
        d = (self.forward(x[:, :n_real]) - self.forward(x, pad_mask=pm)[:, :n_real])
        self.train(was)
        return float(d.abs().max())

    def shape_trace(self, context_size=64, batch=2):
        """Every intermediate shape at the real dimensions, for the report."""
        was = self.training
        self.eval()
        S = context_size
        x = torch.zeros(batch, S, 1, self.n_ch, self.n_samples)
        with torch.no_grad():
            flat = x.reshape(batch * S, 1, self.n_ch, self.n_samples)
            emb = self.encoder(flat)
            z = emb.view(batch, S, self.dim)
            for blk in self.blocks:
                z, _ = blk(z)
            out = self.head(self.norm(z))
        self.train(was)
        return [("input                          [B, S, 1, C, T]", tuple(x.shape)),
                ("trials folded into batch         [B*S, 1, C, T]", tuple(flat.shape)),
                ("EEGNet encoder                          [B*S, D]", tuple(emb.shape)),
                ("unfolded to contexts                   [B, S, D]", tuple(z.shape)),
                (f"cross-trial attention x{self.depth} (no pos)      [B, S, D]", tuple(z.shape)),
                ("per-trial head                         [B, S, 2]", tuple(out.shape))]

    def param_groups(self):
        n = lambda mod: sum(p.numel() for p in mod.parameters() if p.requires_grad)
        enc, att, hd = n(self.encoder), n(self.blocks), n(self.head) + n(self.norm)
        return dict(encoder=enc, attention=att, head=hd, total=enc + att + hd)


def build(n_ch, n_samples, **kw):
    return ContextTransformer(n_ch, n_samples, **kw)


if __name__ == "__main__":
    for n_ch, n_t, label in [(15, 375, "main window 2.0-5.0 s"),
                             (15, 250, "control window 0.0-2.0 s")]:
        m = ContextTransformer(n_ch, n_t)
        print(f"\n=== {label}:  C={n_ch}, T={n_t} ===")
        for name, shp in m.shape_trace(context_size=64, batch=2):
            print(f"  {name:<46} {shp}")
        g = m.param_groups()
        print(f"  parameters: encoder {g['encoder']:,} + attention {g['attention']:,} "
              f"+ head {g['head']:,} = {g['total']:,}")
        print(f"  permutation equivariance, max |deviation|: {m.verify_equivariance():.2e}")
        print(f"  padding is inert,          max |deviation|: {m.verify_padding():.2e}")
