"""
Baseline models for wind power forecasting.

Three classic time-series architectures with a unified interface:
    input:  [B, seq_len, input_dim]
    output: [B, pred_len]

Models:
    WindPowerMLP              — flatten → MLP → predict
    WindPowerLSTM             — stacked LSTM → last hidden → predict
    WindPowerSimpleTransformer — TransformerEncoder → mean pool → predict

Classic long-sequence forecasting baselines added for the Location1 study:
    WindPowerDLinear          — DLinear        (Zeng et al., AAAI 2023)
    WindPowerInformer         — Informer       (Zhou et al., AAAI 2021)
    WindPowerTimesNet         — TimesNet       (Wu et al., ICLR 2023)

All of them consume the exact same scaled windows / splits / scalers as the
FAM and QFAM models, so the comparison isolates the architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MLP Baseline — persistence-residual architecture (TiDE-style)
# =============================================================================
class WindPowerMLP(nn.Module):
    """
    TiDE-inspired residual MLP for wind power forecasting.

    Core idea: predict the *delta* from the last observed power.
      pred = last_power + MLP(all features)

    This persistence connection provides a strong baseline (R²~0.7 alone)
    and lets the MLP focus on learning the deviation, which is a much
    easier optimization problem.

    Architecture:
      - Feature mixing: shared Linear per time step  (28 → 64)
      - Temporal mixing: shared Linear per feature   (96 → 32)
      - Flatten + narrow deep MLP with residual blocks
      - Persistence skip connection
    """

    def __init__(
        self,
        input_dim: int = 28,
        d_model: int = 256,
        pred_len: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        seq_len: int = 96,
    ):
        super().__init__()
        self.power_idx = -1  # power is the last feature column

        # Stage 1: per-timestep feature mixing (shared over time)
        self.feature_mix = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stage 2: per-feature temporal mixing (shared over features)
        # Transpose: [B, T, D] → [B, D, T], then Linear(T, T//3)
        reduced_t = seq_len // 3  # 96 → 32
        self.temporal_mix = nn.Sequential(
            nn.Linear(seq_len, reduced_t),
            nn.LayerNorm(reduced_t),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        flat_dim = d_model * reduced_t  # e.g. 256 * 32 = 8192

        # Stage 3: deep narrow MLP with residual blocks
        hidden = [flat_dim, 1024, 512, 256, 128]
        self.mlp = nn.ModuleList()
        for i in range(len(hidden) - 1):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(hidden[i], hidden[i+1]),
                    nn.BatchNorm1d(hidden[i+1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        # Additional residual blocks at the narrowest layer
        self.res_blocks = nn.ModuleList([
            ResMLPBlock(hidden[-1], dropout) for _ in range(num_layers)
        ])

        # Output: predict delta from last power
        self.head = nn.Linear(hidden[-1], pred_len)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Persistence anchor: last known power value
        last_power = x[:, -1, self.power_idx].unsqueeze(-1)  # [B, 1]

        # Feature mixing: [B, T, D] → [B, T, d_model]
        h = self.feature_mix(x)

        # Temporal mixing: [B, T, d_model] → [B, d_model, T] → [B, d_model, T//3]
        h = h.transpose(1, 2)                     # [B, d_model, T]
        h = self.temporal_mix(h)                  # [B, d_model, T//3]
        h = h.flatten(1)                          # [B, d_model * T//3]

        # MLP
        for block in self.mlp:
            h = block(h)
        for res in self.res_blocks:
            h = res(h)

        delta = self.head(h)                      # [B, pred_len]
        return last_power + delta


class ResMLPBlock(nn.Module):
    """MLP residual block with pre-normalization."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


# =============================================================================
# LSTM Baseline
# =============================================================================
class WindPowerLSTM(nn.Module):
    """Stacked (bi)directional LSTM → last hidden → MLP head.

    bidirectional=True (default) keeps the historical behaviour: hidden=d_model/2
    per direction, final forward+backward states concatenated to d_model.
    bidirectional=False runs a unidirectional stack with hidden=d_model.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        pred_len: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        lstm_hidden = d_model // 2 if bidirectional else d_model

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)  # == d_model

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            # Concatenate final forward + backward hidden states of the last layer
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, d_model]
        else:
            last_hidden = h_n[-1]  # [B, d_model] (unidirectional)
        return self.head(last_hidden)  # [B, pred_len]


# =============================================================================
# Simple Transformer Baseline
# =============================================================================
class WindPowerSimpleTransformer(nn.Module):
    """Standard TransformerEncoder + positional encoding + mean pool."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        pred_len: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
        max_len: int = 500,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (fixed, not learnable)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project + positional encoding
        x = self.input_proj(x)                        # [B, T, d_model]
        x = x + self.pe[:, :T, :]                     # add position
        x = self.encoder(x)                           # [B, T, d_model]

        # Mean pooling over time
        x = x.mean(dim=1)                             # [B, d_model]

        return self.head(x)                           # [B, pred_len]


# =============================================================================
# DLinear  (Zeng et al., "Are Transformers Effective for Time Series
#           Forecasting?", AAAI 2023)
# =============================================================================
class _SeriesDecomp(nn.Module):
    """Moving-average trend/seasonal decomposition (`series_decomp` in DLinear)."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size % 2 == 0:  # keep the moving average centred
            kernel_size += 1
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        """x: [B, L, F] -> (seasonal, trend), both [B, L, F]."""
        front = x[:, 0:1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        padded = torch.cat([front, x, end], dim=1)                        # [B, L+2p, F]
        trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)        # [B, L, F]
        return x - trend, trend


class WindPowerDLinear(nn.Module):
    """
    DLinear: series decomposition + a *single linear layer* over the time axis.

    Kept faithful to the paper: decompose into trend + seasonal with a moving
    average, project each component with ``Linear(seq_len, pred_len)`` and add
    the two. Weights are shared across variates (the paper's default
    ``individual=False``). Because this task is many-to-one, the horizon of the
    ``power`` variate is selected as the forecast.

    This is the deliberately simple control: no attention, no recurrence, and
    only ``input_dim * pred_len * seq_len`` multiply-adds.
    """

    def __init__(self, input_dim: int, seq_len: int = 96, pred_len: int = 1,
                 kernel_size: int = 25, power_idx: int = -1, dropout: float = 0.0):
        super().__init__()
        self.power_idx = power_idx if power_idx is not None else -1
        self.decomp = _SeriesDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)                        # [B, L, F]
        s = self.linear_seasonal(seasonal.permute(0, 2, 1))     # [B, F, P]
        t = self.linear_trend(trend.permute(0, 2, 1))           # [B, F, P]
        out = s + t                                             # [B, F, P]
        return self.dropout(out[:, self.power_idx, :])          # [B, P]


# =============================================================================
# Informer  (Zhou et al., AAAI 2021)
# =============================================================================
class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Informer's encoder/decoder input)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class _MHSA(nn.Module):
    """Multi-head scaled dot-product attention, optionally cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 cross: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.h = n_heads
        self.dk = d_model // n_heads
        self.cross = cross
        self.q = nn.Linear(d_model, d_model)
        if cross:
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
        else:
            self.kv = nn.Linear(d_model, d_model * 2)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, Lq, D = x.shape
        src = x if ctx is None else ctx
        Lk = src.shape[1]

        q = self.q(x).view(B, Lq, self.h, self.dk).transpose(1, 2)      # [B,H,Lq,dk]
        if self.cross:
            k = self.k(src).view(B, Lk, self.h, self.dk).transpose(1, 2)
            v = self.v(src).view(B, Lk, self.h, self.dk).transpose(1, 2)
        else:
            kv = self.kv(src).view(B, Lk, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)   # [B,H,Lq,Lk]
        if mask is not None:
            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                                       # [B,H,Lq,dk]
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.proj(out)


class _InformerEncoderLayer(nn.Module):
    """Self-attention + position-wise FFN, with optional distilling."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 distil: bool = True):
        super().__init__()
        self.attn = _MHSA(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.distil = distil
        if distil:  # "distilling" operation of the Informer encoder
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        if self.distil and x.size(1) > 4:
            x = F.gelu(self.conv(x.transpose(1, 2)))
            x = self.pool(x).transpose(1, 2)
        return x


class _InformerDecoderLayer(nn.Module):
    """Masked self-attention + cross-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = _MHSA(d_model, n_heads, dropout)
        self.cross_attn = _MHSA(d_model, n_heads, dropout, cross=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, mask=mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, ctx=memory)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class WindPowerInformer(nn.Module):
    """
    Informer-style encoder-decoder for many-to-one forecasting.

    Follows the paper's architecture: token embedding + sinusoidal positional
    encoding, an encoder with self-attention and **distilling** (3x1 conv +
    max-pool halving the sequence), a decoder with masked self-attention and
    cross-attention over the encoder memory, and a generative-style head that
    emits the whole horizon at once.

    Deviation, stated explicitly: attention is the dense O(L^2) scaled
    dot-product form, *not* ProbSparse sampling. At seq_len=96 the sparse
    approximation has negligible accuracy impact while its sampling bookkeeping
    dominates on CPU, so the dense form is used. Parameter count is therefore
    very close to the reference implementation.
    """

    def __init__(self, input_dim: int, seq_len: int = 96, pred_len: int = 3,
                 d_model: int = 128, n_heads: int = 8, num_layers: int = 2,
                 d_ff: int | None = None, dropout: float = 0.1,
                 label_len: int | None = None, distil: bool = True):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.pred_len = pred_len
        self.label_len = label_len if label_len is not None else max(1, min(seq_len // 4, seq_len - 1))

        self.enc_embed = nn.Linear(input_dim, d_model)
        self.dec_embed = nn.Linear(input_dim, d_model)
        self.pe = _SinusoidalPE(d_model)

        self.encoder = nn.ModuleList([
            _InformerEncoderLayer(d_model, n_heads, d_ff, dropout, distil)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            _InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Fdim = x.shape

        # ---- encoder (with distilling) ----
        enc = self.pe(self.enc_embed(x))
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)

        # ---- decoder input: known tail tokens + zero placeholders ----
        label = x[:, -self.label_len:, :]
        zeros = torch.zeros(B, self.pred_len, Fdim, device=x.device, dtype=x.dtype)
        dec = self.pe(self.dec_embed(torch.cat([label, zeros], dim=1)))

        Ld = dec.size(1)
        causal = torch.triu(torch.ones(Ld, Ld, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.decoder:
            dec = layer(dec, enc, mask=causal)
        dec = self.dec_norm(dec)

        # ---- generative head: all horizon steps at once ----
        out = self.proj(dec[:, -self.pred_len:, :]).squeeze(-1)   # [B, P]
        return out


# =============================================================================
# TimesNet  (Wu et al., ICLR 2023)
# =============================================================================
class _InceptionBlock2D(nn.Module):
    """Inception block over the (period, frequency) 2-D view (TimesNet paper)."""

    def __init__(self, in_ch: int, out_ch: int, num_kernels: int = 6, dropout: float = 0.0):
        super().__init__()
        self.kernels = nn.ModuleList()
        k = 2
        for i in range(num_kernels):
            if i % 2 == 0:
                ks = (1, 2 ** k + 1)
            else:
                ks = (2 ** k + 1, 1)
                k += 1
            self.kernels.append(nn.Conv2d(in_ch, out_ch, ks,
                                          padding=(ks[0] // 2, ks[1] // 2)))
        self.dconv = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dconv(x)
        for conv in self.kernels:
            out = out + conv(x)
        return self.drop(F.gelu(out))


class _TimesBlock(nn.Module):
    """
    One TimesNet block: FFT -> top-k periods -> 2-D convolution -> amplitude
    weighted aggregation -> residual.

    Periods are derived from the batch-averaged amplitude spectrum, exactly as
    in the reference implementation, so every sample in a batch shares them.
    """

    def __init__(self, seq_len: int, d_model: int, top_k: int = 3,
                 num_kernels: int = 6, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.inception = _InceptionBlock2D(d_model, d_model, num_kernels, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # amplitude spectrum averaged over batch and channels
        amp = torch.fft.rfft(x.float(), dim=1).abs().mean(dim=(0, 2))   # [L//2+1]
        amp = amp.clone()
        amp[0] = 0.0                                                    # drop DC
        k = max(1, min(self.top_k, amp.numel() - 1))
        top = torch.topk(amp, k)
        freqs, weights = top.indices, top.values

        outs = []
        for f in freqs.tolist():
            period = max(2, int(round(L / float(f))))
            length = int(math.ceil(L / period) * period)
            xx = F.pad(x, (0, 0, 0, length - L)) if length != L else x
            rows = length // period
            # [B, L, D] -> [B, D, rows, period]
            xx = xx.reshape(B, rows, period, D).permute(0, 3, 1, 2)
            y = self.inception(xx)
            y = y.permute(0, 2, 3, 1).reshape(B, length, D)[:, :L, :]
            outs.append(y)

        w = torch.softmax(weights, dim=0)
        agg = torch.stack(outs, dim=0)                    # [k, B, L, D]
        y = (agg * w.view(-1, 1, 1, 1)).sum(dim=0)
        return self.norm(self.drop(y) + x)


class WindPowerTimesNet(nn.Module):
    """
    TimesNet for many-to-one forecasting.

    Embeds the multivariate window, applies stacked TimesBlocks (FFT period
    detection -> 2-D Inception -> amplitude-weighted fusion), projects the
    sequence onto the horizon with a linear map over time, and reads out the
    prediction with a linear head.
    """

    def __init__(self, input_dim: int, seq_len: int = 96, pred_len: int = 3,
                 d_model: int = 128, num_layers: int = 2, top_k: int = 3,
                 num_kernels: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            _TimesBlock(seq_len, d_model, top_k, num_kernels, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(seq_len, pred_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)                                   # [B, L, D]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = self.predict_linear(h.permute(0, 2, 1)).permute(0, 2, 1)   # [B, P, D]
        return self.head(h).squeeze(-1)                     # [B, P]

