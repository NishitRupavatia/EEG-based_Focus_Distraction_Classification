"""
EEGTransformer - a standalone temporal-token transformer for Focus vs. Distractor.

WHY A FIFTH NETWORK
  The four networks in dl_models.py all put a CONVOLUTION in front of the attention:
  EEGNet is pure conv, LMDA-Net is conv + attention over feature maps, and both CTNet
  and EEG Conformer learn their tokens with an EEGNet-style / 40-filter conv stack.
  None of them answers the simplest question - what does plain attention over raw
  temporal segments get you, with no learned convolutional front end at all?

  This file is that experiment. It shares NOTHING with CTNet: no inheritance, no
  imported layers, no shared patch embedding. The only thing it borrows is the
  training protocol, which is the whole point of a controlled comparison.

THE DESIGN
  A trial is 3.0 s at 125 Hz. Cut it into 0.2 s slices, flatten each slice across all
  15 electrodes, and project that to a token. Fifteen tokens, one per 0.2 s of trial,
  plus a CLS token. Attention then relates "what happened at +0.4 s" to "what happened
  at +2.2 s" directly, in one hop, which is what a window holding beeps at +0.5 s and
  +2.0 s actually asks for.

  TOKENS ARE TIME, NOT CHANNELS. Making the 15 electrodes the sequence would throw
  away the temporal ordering that carries the entire effect - the README's own
  evidence is that this is a time-locked evoked response, not a spatial pattern.

SHAPES  (main window: C=15, T=375, patch=25 -> N=15 patches, D=64)
  input                       (B, 1, 15, 375)  or  (B, 15, 375)
  temporal patches            (B, 15, 375)        = (B, N, C*patch)
  linear projection           (B, 15, 64)         = (B, N, D)
  prepend CLS                 (B, 16, 64)
  + positional embedding      (B, 16, 64)
  transformer encoder x3      (B, 16, 64)
  CLS token / mean pool       (B, 64)
  classification head         (B, 2)              logits for CrossEntropyLoss

  Control window (T=250) gives N=10 patches, 11 tokens. Both divide by 25 exactly, so
  nothing is ever cropped - but the crop is implemented anyway rather than silently
  producing a wrong-length sequence if the window ever changes.

SIZING
  ~2,800 training trials is small, so this stays deliberately light: D=64, 3 layers,
  4 heads, MLP 4xD, dropout 0.5. That lands between CTNet (~25 k) and EEG Conformer
  (~573 k) in capacity, and every regulariser the other nets use is kept.
"""
import torch
import torch.nn as nn

SFREQ_DL  = 125.0         # rate the networks actually see (250 Hz decimated by 2)
PATCH_S   = 0.20          # one token per 0.2 s of EEG
EMB_DIM   = 64
DEPTH     = 3
N_HEADS   = 4
MLP_RATIO = 4
DROPOUT   = 0.5


class PatchEmbedding(nn.Module):
    """Cut [B, C, T] into non-overlapping temporal patches and project them to tokens.

    (B, C, T) -> (B, N, C*patch) -> (B, N, D).  The flatten is done with `unfold`, so a
    patch holds every electrode's samples for that slice of time: the projection sees
    the full 15-channel topography of a 0.2 s moment and compresses it into one token.
    """

    def __init__(self, n_ch, n_samples, patch, dim):
        super().__init__()
        self.patch = patch
        self.n_patches = n_samples // patch          # remainder is cropped, see forward
        self.patch_dim = n_ch * patch
        self.proj = nn.Linear(self.patch_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                            # (B, C, T)
        T = self.n_patches * self.patch
        x = x[:, :, :T]                              # drop a partial trailing patch
        x = x.unfold(-1, self.patch, self.patch)     # (B, C, N, patch)
        x = x.permute(0, 2, 1, 3).flatten(2)         # (B, N, C*patch)
        return self.norm(self.proj(x))               # (B, N, D)


class EncoderBlock(nn.Module):
    """Pre-norm encoder block: multi-head self-attention + MLP, both residual.

    Pre-norm (LayerNorm before the sublayer, not after) because post-norm needs a
    warmup schedule to train stably and the shared training loop has none - it is
    plain Adam at a fixed 1e-3 for every network in this repo.
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

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + self.drop1(a)
        x = x + self.mlp(self.norm2(x))
        return (x, w) if need_weights else (x, None)


class EEGTransformer(nn.Module):
    """Temporal-token transformer over preprocessed EEG. Focus (0) vs. Distractor (1).

    Constructor signature matches the other networks - (n_ch, n_samples) - so it drops
    straight into DL_MODELS and the shared training loop with no special-casing.

    forward() accepts (B, 1, C, T) - what the shared loop feeds every network - or a
    bare (B, C, T), so the model is usable on its own without the loop's unsqueeze.
    """

    def __init__(self, n_ch, n_samples, n_classes=2, dim=EMB_DIM, depth=DEPTH,
                 heads=N_HEADS, mlp_ratio=MLP_RATIO, patch=None, drop=DROPOUT,
                 pool="cls"):
        super().__init__()
        assert pool in ("cls", "mean"), "pool must be 'cls' or 'mean'"
        if patch is None:
            patch = max(1, int(round(PATCH_S * SFREQ_DL)))
        assert n_samples >= patch, f"window of {n_samples} samples is shorter than one patch"

        self.pool = pool
        self.patch = patch
        self.embed = PatchEmbedding(n_ch, n_samples, patch, dim)
        self.n_patches = self.embed.n_patches
        self.n_tokens = self.n_patches + (1 if pool == "cls" else 0)

        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, n_classes))

    # ---------------------------------------------------------------- forward
    def _tokens(self, x):
        if x.dim() == 4:                             # (B, 1, C, T) from the shared loop
            x = x.squeeze(1)
        assert x.dim() == 3, f"expected (B, C, T) or (B, 1, C, T), got {tuple(x.shape)}"
        z = self.embed(x)                            # (B, N, D)
        if self.cls is not None:
            z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], 1)
        return self.pos_drop(z + self.pos)           # positional info actually added

    def forward(self, x):
        z = self._tokens(x)
        for b in self.blocks:
            z, _ = b(z)
        z = self.norm(z)
        z = z[:, 0] if self.pool == "cls" else z.mean(1)
        return self.head(z)                          # (B, 2) logits

    @torch.no_grad()
    def attention_maps(self, x):
        """Per-layer attention, (depth, B, n_tokens, n_tokens). Diagnostics only."""
        was = self.training
        self.eval()
        z, maps = self._tokens(x), []
        for b in self.blocks:
            z, w = b(z, need_weights=True)
            maps.append(w)
        self.train(was)
        return torch.stack(maps)

    def shape_trace(self, n_ch, n_samples, batch=2):
        """Every intermediate shape, for the verification printout."""
        was = self.training
        self.eval()
        x = torch.zeros(batch, 1, n_ch, n_samples)
        with torch.no_grad():
            b3 = x.squeeze(1)
            T = self.embed.n_patches * self.patch
            patches = (b3[:, :, :T].unfold(-1, self.patch, self.patch)
                       .permute(0, 2, 1, 3).flatten(2))
            tok = self.embed(b3)
            full = self._tokens(x)
            z = full
            for b in self.blocks:
                z, _ = b(z)
            z = self.norm(z)
            pooled = z[:, 0] if self.pool == "cls" else z.mean(1)
            out = self.head(pooled)
        self.train(was)
        pool_label = "CLS token" if self.pool == "cls" else "mean pool"
        return [("input                        [B, C, T]", tuple(b3.shape)),
                ("patches       [B, N, C x patch_size]", tuple(patches.shape)),
                ("patch projection             [B, N, D]", tuple(tok.shape)),
                ("+ CLS + positional         [B, N+1, D]", tuple(full.shape)),
                ("transformer encoder        [B, N+1, D]", tuple(z.shape)),
                (f"{pool_label:<28} [B, D]", tuple(pooled.shape)),
                ("classifier                      [B, 2]", tuple(out.shape))]


def build(n_ch, n_samples, **kw):
    """Registry-shaped constructor: build(n_ch, n_samples) -> EEGTransformer."""
    return EEGTransformer(n_ch, n_samples, **kw)


def count_params(n_ch=15, n_samples=375, **kw):
    m = EEGTransformer(n_ch, n_samples, **kw)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n_ch, n_t, label in [(15, 375, "main window 2.0-5.0 s"),
                             (15, 250, "control window 0.0-2.0 s")]:
        m = EEGTransformer(n_ch, n_t)
        print(f"\n=== {label}:  C={n_ch}, T={n_t} ===")
        for name, shp in m.shape_trace(n_ch, n_t, batch=4):
            print(f"  {name:<40} {shp}")
        print(f"  tokens {m.n_tokens} ({m.n_patches} patches + CLS), "
              f"patch {m.patch} samples = {m.patch / SFREQ_DL:.2f} s")
        print(f"  trainable parameters: "
              f"{sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
