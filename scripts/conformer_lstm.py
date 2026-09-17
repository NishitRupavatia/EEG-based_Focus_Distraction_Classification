"""
ConformerLSTM - conv-conditioned local attention, then a recurrent sequence model.

THE ONE-LINE DIFFERENCE FROM conformer_transformer.py
  The tokenizer and the Conformer block are IMPORTED from that module, not copied. The
  only thing this network changes is what consumes the token sequence: a bidirectional
  LSTM instead of a reduction stage plus transformer blocks. That is deliberate - it is
  the only way the comparison between the two means anything, and it is the same
  discipline 09_transformer.py applies to folds_for().

WHY RECURRENCE IS THE RIGHT TOOL AT THIS SEQUENCE LENGTH
  1. PARAMETER EFFICIENCY. Self-attention costs ~4d^2 per block no matter how long the
     sequence is. The transformer stage in the sibling model spent 17,088 parameters
     attending over 13 tokens. A BiLSTM does the sequence job for 9,600, and its cost
     grows with LENGTH - which is 36 here, i.e. nothing - rather than with WIDTH.

  2. ORDER IS FREE. The sibling model needed a learned absolute positional table. An
     LSTM is order-aware by construction, so there is NO positional encoding here at
     all. That also makes the parameter count EXACTLY identical for the 375-sample main
     window and the 250-sample control window, so the leakage guardrail is run by a
     byte-for-byte equally sized model.

  3. IT MATCHES THE TASK. Classifying a fixed-length, time-locked window is an
     integrate-evidence-in-order problem. Recurrence encodes that prior directly;
     attention has to learn it from 2,789 training trials.

  Division of labour: the Conformer stage says what happened inside each 200 ms patch
  and how patches relate locally; the LSTM accumulates it across the trial.

WHAT THE 94 k SIBLING TAUGHT US, AND WHAT IT CHANGED HERE
  Conformer-Transformer scored AUC 0.663 +- 0.080 against EEGNet's 0.794, and the logs
  said exactly why: train loss fell 0.53 -> 0.22 while validation loss climbed
  0.56 -> 1.05, best epochs were 11-33 against EEGNet's 133, and two of five folds never
  left val loss ln(2) = 0.693 - they never learned at all. So:

    * ~17.8 k parameters, not 94 k. Every model in this repo above ~20 k has scored
      WORSE than one below it: 1.7 k -> 0.798, 3.8 k -> 0.799, 21 k -> 0.780,
      94 k -> 0.663, 175 k -> 0.521, 573 k -> 0.771.
    * ONE Conformer block at FFN ratio 1, not two at ratio 2 (7.1 k, not 72.5 k).
    * No reduction stage - the LSTM is linear in sequence length, so it can afford the
      full 80 ms resolution the transformer could not.
    * Orthogonal recurrent init and forget-gate bias 1.0, so the LSTM starts willing to
      remember. A forget gate initialised near 0.5 is one plausible reason a model sits
      at chance for a while, and two folds previously never got started at all.

  A 40-epoch probe on the two folds that stalled reached AUC 0.669 and 0.702 with
  validation loss 0.658 and 0.650 - well clear of ln(2). Those folds were never the
  problem; the capacity was.

SHAPES  (main window: C=15, T=375, N=36, H=24)
  input                            (B, 1, 15, 375)  or  (B, 15, 375)
  temporal conv  k=13 (0.104 s)    (B,  8, 15, 375)
  spatial depthwise (15,1) x D=2   (B, 16,  1, 375)
  avg-pool 25 stride 10            (B, 16,  1,  36)
  token projection 16 -> 24        (B, 36, 24)        = (B, N, dim)
  Conformer block x1               (B, 36, 24)
  BiLSTM(24 -> 24)                 (B, 36, 48)  + final states (2, B, 24)
  readout                          (B, 48)
  classification head              (B, 2)             logits for CrossEntropyLoss

  Control window (T=250) gives N=23 and the SAME parameter count - no positional table,
  no flatten, so nothing in this model depends on the window length.
"""
import torch
import torch.nn as nn

# The tokenizer and the Conformer block are shared with the sibling model on purpose:
# one definition, so "Conformer-LSTM vs Conformer-Transformer" isolates the sequence
# stage and nothing else. A change to either layer moves both networks together.
from conformer_transformer import ConvTokenizer, ConformerBlock, SFREQ_DL  # noqa: F401

# --- tokenizer (narrower than the sibling's: 8x2 maps into 24 dims, not 16x2 into 48) --
F1         = 8
D          = 2

# --- Conformer stage ---------------------------------------------------------
DIM        = 24
CONF_DEPTH = 1
N_HEADS    = 4            # 24/4 = 6 dims per head
FFN_RATIO  = 1            # the paper uses 4, the sibling used 2; 1 is what fits here
CONV_K     = 9            # depthwise kernel over tokens = 0.72 s of context

# --- recurrent stage ---------------------------------------------------------
HIDDEN      = 24
LSTM_LAYERS = 1
BIDIR       = True        # the trial is classified offline, all 3 s are available
REDUCE_R    = 1           # no token reduction - the LSTM is linear in sequence length

# Chosen by mean VALIDATION loss over all five folds under the real protocol, with NO
# test prediction computed during the selection. The numbers, so the choice is auditable
# rather than a preference:
#
#   readout    val_loss           val_auc          best_ep
#   last       0.6232 +- 0.0200   0.718 +- 0.031   31      <- selected
#   mean       0.6340 +- 0.0173   0.705 +- 0.030   32
#   attn       0.6397 +- 0.0226   0.700 +- 0.025   35
#   meanmax    0.6460 +- 0.0148   0.691 +- 0.037   38
#
# 'last' wins on validation loss and on validation AUC, with the same ordering on both.
# Be honest about the size of it: the 0.011 margin over 'mean' is SMALLER than the
# +-0.020 between-fold spread, so this is a consistent preference, not a decisive one.
# That is why B4 re-reports all four on the test folds afterwards - the selection is
# what must stay clean, not the curiosity.
READOUT    = "last"

DROPOUT    = 0.3
HEAD_DROP  = 0.5
READOUTS   = ("last", "attn", "mean", "meanmax")


class AttentionPool(nn.Module):
    """One learned score per time step -> softmax -> weighted sum. 49 parameters.

    Only used when readout='attn'. Worth keeping as an option beyond its accuracy: the
    weights are a per-token relevance curve that can be plotted against the beep times,
    which is an independent check on the same claim figure 7 makes from decoding.
    """

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, return_weights=False):
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)      # (B, S)
        out = torch.einsum("bs,bsd->bd", w, x)                   # (B, d)
        return (out, w) if return_weights else out


class ConformerLSTM(nn.Module):
    """Conv tokenizer -> Conformer -> BiLSTM -> readout -> head. Focus (0) vs. Distractor (1).

    Constructor signature matches every other network in this repo - (n_ch, n_samples) -
    so it drops straight into dl_models.DL_MODELS and the shared training loop with no
    special-casing. forward() accepts (B, 1, C, T), which is what the loop feeds, or a
    bare (B, C, T).

    `readout` selects how the sequence becomes one vector:
      'last'     final forward + backward hidden states, concatenated (the default)
      'attn'     attention pooling over the LSTM outputs
      'mean'     mean over time steps
      'meanmax'  mean and max concatenated
    Setting hidden=0 removes the LSTM entirely and setting conf_depth=0 removes the
    Conformer stage, which are the two ablations that decide whether either earns its
    parameters. Both are exercised by the __main__ self-test.
    """

    def __init__(self, n_ch, n_samples, n_classes=2, f1=F1, d=D, dim=DIM,
                 conf_depth=CONF_DEPTH, heads=N_HEADS, ffn_ratio=FFN_RATIO,
                 conv_k=CONV_K, hidden=HIDDEN, lstm_layers=LSTM_LAYERS, bidir=BIDIR,
                 reduce_r=REDUCE_R, readout=READOUT, rnn="lstm",
                 drop=DROPOUT, head_drop=HEAD_DROP):
        super().__init__()
        assert readout in READOUTS, f"readout must be one of {READOUTS}"
        assert rnn in ("lstm", "gru"), "rnn must be 'lstm' or 'gru'"
        self.n_ch, self.n_samples = n_ch, n_samples
        self.readout, self.bidir, self.rnn_kind = readout, bidir, rnn
        self.dim, self.hidden, self.heads = dim, hidden, heads
        self.reduce_r = reduce_r

        self.tokenizer = ConvTokenizer(n_ch, dim=dim, f1=f1, d=d, drop=drop)
        self.patch, self.hop = self.tokenizer.patch, self.tokenizer.hop
        assert n_samples >= self.patch, (
            f"window of {n_samples} samples is shorter than one {self.patch}-sample patch")

        self.conformer = nn.ModuleList(
            [ConformerBlock(dim, heads, ffn_ratio, conv_k, drop) for _ in range(conf_depth)])

        # Optional token reduction. Off by default - it exists so "does the LSTM want
        # fine temporal resolution" is a switch rather than a rewrite.
        self.reduce = (nn.Conv1d(dim, dim, reduce_r, stride=reduce_r, groups=dim)
                       if reduce_r > 1 else None)

        rnn_cls = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = (rnn_cls(dim, hidden, lstm_layers, batch_first=True,
                            bidirectional=bidir,
                            dropout=drop if lstm_layers > 1 else 0.0)
                    if hidden > 0 else None)
        self.n_dirs = 2 if bidir else 1
        seq_dim = hidden * self.n_dirs if hidden > 0 else dim

        self.pool = AttentionPool(seq_dim) if readout == "attn" else None
        head_in = seq_dim * (2 if readout == "meanmax" else 1)
        self.norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear(head_in, n_classes))

        self.n_patches = self.tokenizer.n_patches(n_samples)
        self.n_steps = self.n_patches // reduce_r
        assert self.n_steps >= 1, f"{self.n_patches} patches reduced by {reduce_r} leaves nothing"
        self._init_rnn()
        self._verify_shapes()

    # ------------------------------------------------------------------- init
    def _init_rnn(self):
        """Orthogonal recurrent weights, Xavier input weights, forget-gate bias 1.0.

        All three are standard, and the third is the one that matters here: PyTorch
        initialises every gate bias uniformly around zero, which puts the forget gate at
        sigmoid(0) = 0.5 and makes the cell leak half its state per step before training
        has done anything. The sibling model had two folds sit at chance for good; this
        removes one candidate reason. These are INITIALISATION choices, so no part of
        the shared training protocol changes.

        GRUs have three gates in the order r, z, n and no forget gate, so the bias
        fill is skipped for them rather than applied to the wrong slice.
        """
        if self.rnn is None:
            return
        for name, p in self.rnn.named_parameters():
            if "weight_hh" in name:
                # (gates*hidden, hidden) - orthogonalise each gate's block separately
                h = p.shape[1]
                for i in range(0, p.shape[0], h):
                    nn.init.orthogonal_(p.data[i:i + h])
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "bias" in name:
                p.data.zero_()
                if self.rnn_kind == "lstm":
                    n = p.shape[0] // 4            # i, f, g, o - f is the second block
                    p.data[n:2 * n].fill_(1.0)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _canon(x):
        """-> (B, 1, C, T). Accepts (B, 1, C, T) from the shared loop or a bare (B, C, T)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 4 and x.shape[1] == 1, \
            f"expected (B, C, T) or (B, 1, C, T), got {tuple(x.shape)}"
        return x

    def _sequence(self, x):
        """(B, [1,] C, T) -> (sequence (B, S, seq_dim), final states or None)."""
        z = self.tokenizer(self._canon(x))                   # (B, N, dim)
        for blk in self.conformer:
            z, _ = blk(z)                                    # (B, N, dim)
        if self.reduce is not None:
            n = (z.shape[1] // self.reduce_r) * self.reduce_r
            z = self.reduce(z[:, :n].transpose(1, 2)).transpose(1, 2)
        if self.rnn is None:
            return z, None
        seq, state = self.rnn(z)                             # (B, S, H*dirs)
        h = state[0] if self.rnn_kind == "lstm" else state   # LSTM returns (h, c)
        return seq, h

    def _readout(self, seq, h):
        if self.readout == "attn":
            return self.pool(seq)
        if self.readout == "mean":
            return seq.mean(1)
        if self.readout == "meanmax":
            return torch.cat([seq.mean(1), seq.max(1).values], -1)
        # 'last': the final state of each direction. For a BiLSTM that is the END of the
        # trial from the forward pass and the START of it from the backward pass, so the
        # pair summarises the window from both ends. Falls back to the last time step
        # when there is no recurrent layer at all (the hidden=0 ablation).
        if h is None:
            return seq[:, -1]
        return torch.cat([h[-2], h[-1]], -1) if self.bidir else h[-1]

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        seq, h = self._sequence(x)
        return self.head(self.norm(self._readout(seq, h)))    # (B, 2) logits

    @torch.no_grad()
    def apply_max_norm(self):
        """Called by dl_models._run_epoch after every optimiser step, if present."""
        self.tokenizer.apply_max_norm()

    # ----------------------------------------------------------- verification
    @torch.no_grad()
    def _verify_shapes(self):
        """One real forward pass, asserting the derived sizes. Cheap insurance: a wrong
        stride would still train and still report a plausible number."""
        was = self.training
        self.eval()
        x = torch.zeros(2, 1, self.n_ch, self.n_samples)
        n = self.tokenizer(x).shape[1]
        assert n == self.n_patches, f"predicted {self.n_patches} patches, forward gave {n}"
        seq, h = self._sequence(x)
        assert seq.shape[1] == self.n_steps, \
            f"predicted {self.n_steps} steps, forward gave {seq.shape[1]}"
        assert self.forward(x).shape == (2, self.head[-1].out_features)
        self.train(was)

    @torch.no_grad()
    def pooling_weights(self, x):
        """Per-token relevance curve, (B, S). Only defined for readout='attn'."""
        assert self.readout == "attn", "pooling weights exist only for readout='attn'"
        was = self.training
        self.eval()
        seq, _ = self._sequence(x)
        _, w = self.pool(seq, return_weights=True)
        self.train(was)
        return w

    @torch.no_grad()
    def attention_maps(self, x):
        """Conformer-stage attention, (depth, B, N, N). Diagnostics only."""
        was = self.training
        self.eval()
        z, maps = self.tokenizer(self._canon(x)), []
        for blk in self.conformer:
            z, w = blk(z, need_weights=True)
            maps.append(w)
        self.train(was)
        return torch.stack(maps) if maps else None

    # ------------------------------------------------------------- description
    def token_times(self):
        """Centre of each token in seconds from the start of the window.

        The reason this exists: a relevance curve or an attention map is only readable
        if you know which millisecond each token covers.
        """
        step = self.hop * self.reduce_r
        return [round((i * step + self.patch / 2) / SFREQ_DL, 4) for i in range(self.n_steps)]

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
            rows.append(("token projection             [B, N, dim]", tuple(z.shape)))
            for blk in self.conformer:
                z, _ = blk(z)
            rows.append((f"Conformer x{len(self.conformer)}                 [B, N, dim]",
                         tuple(z.shape)))
            seq, h = self._sequence(x)
            tag = f"Bi{self.rnn_kind.upper()}" if self.bidir else self.rnn_kind.upper()
            rows.append((f"{tag}({self.dim}->{self.hidden})          [B, S, H*dirs]"
                         if self.rnn is not None else "no recurrent layer",
                         tuple(seq.shape)))
            if h is not None:
                rows.append(("final states              [dirs, B, H]", tuple(h.shape)))
            r = self._readout(seq, h)
            rows.append((f"readout ({self.readout})", tuple(r.shape)))
            rows.append(("classifier                      [B, 2]",
                         tuple(self.head(self.norm(r)).shape)))
        self.train(was)
        return rows

    def param_groups(self):
        n = lambda m: 0 if m is None else sum(q.numel() for q in m.parameters()
                                              if q.requires_grad)
        g = dict(tokenizer=n(self.tokenizer), conformer=n(self.conformer),
                 reduce=n(self.reduce), recurrent=n(self.rnn),
                 head=n(self.pool) + n(self.norm) + n(self.head))
        g["total"] = sum(g.values())
        assert g["total"] == sum(q.numel() for q in self.parameters() if q.requires_grad)
        return g

    def config(self):
        """Everything the job row needs to describe what was actually trained."""
        return dict(
            n_channels=self.n_ch, n_times=self.n_samples,
            patch_samples=self.patch, hop_samples=self.hop,
            patch_s=round(self.patch / SFREQ_DL, 4), hop_s=round(self.hop / SFREQ_DL, 4),
            n_patches=self.n_patches, n_steps=self.n_steps,
            dim=self.dim, conf_depth=len(self.conformer), heads=self.heads,
            conv_k=CONV_K, reduce_r=self.reduce_r,
            rnn=self.rnn_kind, hidden=self.hidden,
            lstm_layers=(self.rnn.num_layers if self.rnn is not None else 0),
            bidirectional=bool(self.bidir), readout=self.readout, pos_enc="none",
            n_params=self.param_groups()["total"])


def build(n_ch, n_samples, **kw):
    """Registry-shaped constructor: build(n_ch, n_samples) -> ConformerLSTM."""
    return ConformerLSTM(n_ch, n_samples, **kw)


def count_params(n_ch=15, n_samples=375, **kw):
    return ConformerLSTM(n_ch, n_samples, **kw).param_groups()["total"]


if __name__ == "__main__":
    for n_ch, n_t, label in [(15, 375, "main window 2.0-5.0 s"),
                             (15, 250, "control window 0.0-2.0 s")]:
        m = ConformerLSTM(n_ch, n_t)
        print(f"\n=== {label}:  C={n_ch}, T={n_t} ===")
        for name, shp in m.shape_trace(batch=4):
            print(f"  {name:<44} {shp}")
        g = m.param_groups()
        print(f"  tokens: {m.n_patches} patches of {m.patch} samples "
              f"({m.patch / SFREQ_DL:.2f} s) every {m.hop} ({m.hop / SFREQ_DL:.2f} s)"
              f"  ->  {m.n_steps} recurrent steps, no positional encoding")
        print(f"  parameters: tokenizer {g['tokenizer']:,} + conformer {g['conformer']:,}"
              f" + recurrent {g['recurrent']:,} + head {g['head']:,} = {g['total']:,}")
        print(f"  token centres (s): {m.token_times()[:5]} ... {m.token_times()[-1]}")

    print("\n=== variants / ablations (main window) ===")
    for kw in [dict(), dict(readout="attn"), dict(readout="mean"), dict(readout="meanmax"),
               dict(bidir=False), dict(conf_depth=0), dict(hidden=0), dict(conf_depth=2),
               dict(lstm_layers=2), dict(dim=32, hidden=32), dict(dim=16, hidden=16),
               dict(ffn_ratio=2), dict(reduce_r=3), dict(rnn="gru")]:
        m = ConformerLSTM(15, 375, **kw)
        desc = ", ".join(f"{k}={v}" for k, v in kw.items()) or "default"
        g = m.param_groups()
        print(f"  {desc:<30} N={m.n_patches:>3} S={m.n_steps:>3}  "
              f"tok {g['tokenizer']:>5} conf {g['conformer']:>6} rnn {g['recurrent']:>6} "
              f"head {g['head']:>5}  TOTAL {g['total']:>7,}")
