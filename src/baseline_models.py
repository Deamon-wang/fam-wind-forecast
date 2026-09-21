"""
Baseline models for wind power forecasting.

Three classic time-series architectures with a unified interface:
    input:  [B, seq_len, input_dim]
    output: [B, pred_len]

Models:
    WindPowerMLP              — flatten → MLP → predict
    WindPowerLSTM             — stacked LSTM → last hidden → predict
    WindPowerSimpleTransformer — TransformerEncoder → mean pool → predict
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
    """Stacked bidirectional LSTM → last hidden → MLP head."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        pred_len: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        lstm_hidden = d_model // 2  # halved because bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out_dim = d_model  # lstm_hidden * 2 (bidirectional)

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
        # Concatenate final forward + backward hidden states
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, d_model]
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
