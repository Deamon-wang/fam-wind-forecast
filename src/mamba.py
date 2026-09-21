
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import pyvqnet
from src.VQC_cir import Angle_VQC_cir
import tednet.tnn.tensor_train.base as tt
from pyvqnet.nn.torch import TorchModule,Linear
from pyvqnet.tensor import tensor
pyvqnet.backends.set_backend("torch")

########################################################################
# 3.20
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]


class BetterMambaBlock(nn.Module):
    """
    改进点：
    1. 先做通道映射 in_proj
    2. 在时间维上做 depthwise causal conv
    3. 再做门控融合
    """
    def __init__(self, d_model, kernel_size=5, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)

        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,T,D]
        u, v = self.in_proj(x).chunk(2, dim=-1)

        v = v.transpose(1, 2)          # [B,D,T]
        v = self.dw_conv(v)            # [B,D,T+k-1]
        v = v[:, :, :x.size(1)]        # causal trim
        v = self.pw_conv(v)
        v = v.transpose(1, 2)          # [B,T,D]

        v = torch.sigmoid(self.norm(v))
        out = self.out_proj(u * v)
        return self.dropout(out) # self.dropout(out)

 
class FlowAttention(nn.Module):
    """
    更稳一点的 Flow-like attention：
    - head 内做 RMSNorm 风格约束   - 正域映射
    - 双边归一    - 适合 encoder 历史窗口建模
    """
    def __init__(self, d_model, n_heads, dropout=0.1, eps=1e-6):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()  # [B,H,T,Dh]

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, x):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = F.softplus(q * self.scale) + self.eps
        k = F.softplus(k) + self.eps

        # 计算每个时刻的前缀 KV 和前缀 K
        # kv_t = sum_{s<=t} k_s v_s^T
        kv = torch.einsum("bhtd,bhte->bhtde", k, v)   # [B,H,T,Dh,Dh]
        kv_prefix = kv.cumsum(dim=2)                  # [B,H,T,Dh,Dh]
         # k_prefix_t = sum_{s<=t} k_s
        k_prefix = k.cumsum(dim=2)                    # [B,H,T,Dh]

        # 分子: q_t @ kv_prefix_t
        out = torch.einsum("bhtd,bhtde->bhte", q, kv_prefix)   # [B,H,T,Dh]

        # 分母: q_t @ k_prefix_t
        z = torch.einsum("bhtd,bhtd->bht", q, k_prefix) + self.eps   # [B,H,T]

        out = out / z.unsqueeze(-1)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out
        # row_norm = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k_sum) + self.eps)
        # col_norm = 1.0 / (torch.einsum("bhtd,bhd->bht", k, q_sum) + self.eps)

        # weighted_k = k * col_norm.unsqueeze(-1)
        # kv = torch.einsum("bhtd,bhte->bhde", weighted_k, v)   # [B,H,Dh,Dh]
        # out = torch.einsum("bhtd,bhde->bhte", q, kv)          # [B,H,T,Dh]
        # out = out * row_norm.unsqueeze(-1)

        # out = self._merge_heads(out)
        # out = self.out_proj(out)
        # out = self.dropout(out)
        # return out


# Mamba 用于局部动态建模；
# GRU 用于顺序状态编码；
# FlowAttention 用于全局关键历史重分配。
class WindPowerMambaFlowGRU(nn.Module):
    def __init__(self, input_dim,d_model=64,n_heads=4,num_layers=2,gru_hidden=64,
        pred_len=6,dropout=0.1,power_idx=None,recent_window=12,use_persistence=True,local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,):
        super().__init__()

        self.input_dim = input_dim
        self.pred_len = pred_len
        self.power_idx = power_idx
        self.recent_window = recent_window
        self.use_persistence = use_persistence

        self.embedding = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_enc = PositionalEncoding(d_model)

        k1, k2 = local_kernel_sizes
        self.local_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k1, padding=k1 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=k2, padding=k2 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mamba_blocks = nn.ModuleList([
            BetterMambaBlock(d_model, dropout=dropout, kernel_size=mamba_kernel_size)
            for _ in range(num_layers)
        ])
        self.mamba_norm = nn.LayerNorm(d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.gru_proj = nn.Linear(gru_hidden, d_model) if gru_hidden != d_model else nn.Identity()

        self.flow_norm = nn.LayerNorm(d_model)
        self.flow_attn = FlowAttention(d_model, n_heads, dropout)
        # self.flow_attn = CausalFlowAttention(d_model, n_heads, dropout)
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

        self.shortcut = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        x_raw = x
        x = self.embedding(x)
        x = self.input_dropout(x)
        x = self.pos_enc(x)
        x = x.transpose(1, 2)
        x = self.local_stem(x)
        x = x.transpose(1, 2)

        for blk in self.mamba_blocks:
            x = x + blk(x)
        x = self.mamba_norm(x)

        gru_out, _ = self.gru(x)
        x = self.gru_proj(gru_out)

        x = x + self.flow_attn(self.flow_norm(x))
        x = self.final_norm(x)

        last_token = x[:, -1, :]
        rw = min(self.recent_window, x.size(1))
        recent_pool = x[:, -rw:, :].mean(dim=1)
        last_raw = x_raw[:, -1, :]

        feat = torch.cat([last_token, recent_pool, last_raw], dim=-1)
        delta = self.head(feat)

        base = self.shortcut(last_raw)
        if self.use_persistence and (self.power_idx is not None):
            last_power = x_raw[:, -1, self.power_idx].unsqueeze(-1)
            if self.pred_len > 1:
                last_power = last_power.expand(-1, self.pred_len)
            base = base + last_power

        return base + delta

##############################################################

class WindPowerGRUFlow(nn.Module):
    """
    版本 B：GRU + FlowAttention + head

    结构：
        x -> embedding -> GRU -> FlowAttention -> final_norm
          -> [last_token, recent_pool, last_raw] -> head -> pred

    说明：
    1. 不使用 local_stem
    2. 不使用 Mamba blocks
    3. 不使用 persistence
    4. shortcut 默认关闭，避免和输入中的 power 重复建模
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        gru_hidden=64,
        gru_layers=1,
        pred_len=6,
        dropout=0.2,
        recent_window=12,
        use_pos_enc=False,
        use_shortcut=False,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")

        self.input_dim = input_dim
        self.d_model = d_model
        self.gru_hidden = gru_hidden
        self.pred_len = pred_len
        self.recent_window = recent_window
        self.use_shortcut = use_shortcut

        self.embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.gru_proj = nn.Identity() if gru_hidden == d_model else nn.Linear(gru_hidden, d_model)

        self.flow_norm = nn.LayerNorm(d_model)
        self.flow_attn = FlowAttention(d_model, n_heads, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        if use_shortcut:
            self.shortcut = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        # x: [B, T, input_dim]
        x_raw = x

        x = self.embedding(x)          # [B, T, d_model]
        x = self.input_norm(x)
        if self.use_pos_enc:
            x = self.pos_enc(x)
        x = self.input_dropout(x)

        x, _ = self.gru(x)             # [B, T, gru_hidden]
        x = self.gru_proj(x)           # [B, T, d_model]

        x = x + self.flow_attn(self.flow_norm(x))
        x = self.final_norm(x)

        last_token = x[:, -1, :]       # [B, d_model]
        rw = min(self.recent_window, x.size(1))
        recent_pool = x[:, -rw:, :].mean(dim=1)   # [B, d_model]
        last_raw = x_raw[:, -1, :]     # [B, input_dim]

        feat = torch.cat([last_token, recent_pool, last_raw], dim=-1)
        out = self.head(feat)          # [B, pred_len]

        if self.use_shortcut:
            out = out + self.shortcut(last_raw)

        return out



class WindPowerExogFusion(nn.Module):
    """
    1. exog_branch: 气象+时间 -> Conv + FlowAttention
    2. ar_branch: 最近功率 -> 线性基线
    3. gate_fusion: 学习融合两条分支
    """

    def __init__(self, input_dim, d_model=64, pred_len=6, num_layers=1,
                 n_heads=4, dropout=0.2, recent_window=12, power_idx=-1):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.recent_window = recent_window
        self.power_idx = power_idx

        self.embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        conv_blocks = []
        for _ in range(num_layers):
            conv_blocks.extend([
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.temporal_conv = nn.Sequential(*conv_blocks)

        self.flow_norm = nn.LayerNorm(d_model)
        self.flow_attn = FlowAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

        self.exog_head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

        self.ar_head = nn.Sequential(
            nn.Linear(recent_window, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, input_dim]
        x_raw = x

        z = self.embedding(x)
        z = self.input_norm(z)
        z = self.input_dropout(z)

        z = z.transpose(1, 2)
        z = self.temporal_conv(z)
        z = z.transpose(1, 2)

        z = z + self.flow_attn(self.flow_norm(z))
        z = self.final_norm(z)

        last_token = z[:, -1, :]
        rw = min(self.recent_window, z.size(1))
        recent_pool = z[:, -rw:, :].mean(dim=1)
        last_raw = x_raw[:, -1, :]

        fusion_feat = torch.cat([last_token, recent_pool, last_raw], dim=-1)
        exog_pred = self.exog_head(fusion_feat)

        power_hist = x_raw[:, -rw:, self.power_idx]
        if rw < self.recent_window:
            pad = self.recent_window - rw
            power_hist = F.pad(power_hist, (pad, 0))
        ar_pred = self.ar_head(power_hist)

        gate = self.gate(fusion_feat)
        out = gate * exog_pred + (1.0 - gate) * ar_pred
        return out

################################
# 0324 

class WindPowerMambaFlow(nn.Module):
    """
   风电预测混合模型：
        x
        -> embedding + norm + pos_enc
        -> local_stem
        -> Mamba branch
        -> FlowAttention branch
        -> gate fusion
        -> [last_token, recent_pool, last_raw] -> delta head
        -> shortcut(last_raw)
        -> out = base + delta

    说明：
    2. Mamba 分支负责局部/中程动态建模
    3. FlowAttention 分支负责全局关键历史信息重分配
    4. 用 gate 做双分支融合
    5. 用 shortcut，避免 X / y 分别标准化造成的尺度错位
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_mamba_layers=2,
        pred_len=6,
        dropout=0.1,
        recent_window=12,
        local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,
        use_pos_enc=True,
        use_shortcut=True,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.recent_window = recent_window
        self.use_pos_enc = use_pos_enc
        self.use_shortcut = use_shortcut

        # 共享输入层
        self.embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model)

        # 局部卷积 stem
        k1, k2 = local_kernel_sizes
        self.local_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k1, padding=k1 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=k2, padding=k2 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mamba 分支
        self.mamba_blocks = nn.ModuleList([
            BetterMambaBlock(
                d_model=d_model,
                kernel_size=mamba_kernel_size,
                dropout=dropout
            )
            for _ in range(num_mamba_layers)
        ])
        self.mamba_norm = nn.LayerNorm(d_model)

        # FlowAttention 分支
        self.flow_in_norm = nn.LayerNorm(d_model)
        self.flow_attn = FlowAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.flow_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.flow_out_norm = nn.LayerNorm(d_model)

        # 门控融合
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # 预测头：学习残差 delta
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        # shortcut 基线
        if self.use_shortcut:
            self.shortcut = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, pred_len]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        x_raw = x

        # shared encoder
        z = self.embedding(x)
        z = self.input_norm(z)
        if self.use_pos_enc:
            z = self.pos_enc(z)
        z = self.input_dropout(z)

        # local stem
        z = z.transpose(1, 2)          # [B, D, T]
        z = self.local_stem(z)
        z = z.transpose(1, 2)          # [B, T, D]

        # branch 1: Mamba
        z_m = z
        for blk in self.mamba_blocks:
            z_m = z_m + blk(z_m)
        z_m = self.mamba_norm(z_m)

        # branch 2: FlowAttention
        z_f = z + self.flow_attn(self.flow_in_norm(z))
        z_f = z_f + self.flow_ffn(z_f)
        z_f = self.flow_out_norm(z_f)

        # gate fusion
        m_last = z_m[:, -1, :]
        f_last = z_f[:, -1, :]
        last_raw = x_raw[:, -1, :]

        gate_in = torch.cat([m_last, f_last, last_raw], dim=-1)
        gate = self.gate_net(gate_in).unsqueeze(1)   # [B,1,D]

        z = gate * z_m + (1.0 - gate) * z_f

        # readout
        last_token = z[:, -1, :]
        rw = min(self.recent_window, z.size(1))
        recent_pool = z[:, -rw:, :].mean(dim=1)

        feat = torch.cat([last_token, recent_pool, last_raw], dim=-1)
        delta = self.head(feat)

        if self.use_shortcut:
            base = self.shortcut(last_raw)
            out = base + delta
        else:
            out = delta

        return out




###################################

# 2026 3 24
class WindPowerMambaFlowLite(nn.Module):
    """
    方案 A：
    Mamba 主干 + FlowAttention 轻量残差修正

    结构：
        x
        -> embedding + norm + pos_enc
        -> local_stem
        -> Mamba backbone
        -> Flow refine
        -> compact shortcut + delta head
        -> out = delta + shortcut_scale * base

    设计目标：
    1. 让 Mamba 成为主干，避免双分支平权导致表示被稀释
    2. FlowAttention 只做轻量 refine，而不是主路径并联
    3. shortcut 变弱，避免抢占主干学习空间
    4. 不使用 GRU
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_mamba_layers=2,
        pred_len=6,
        dropout=0.05,
        recent_window=12,
        local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,
        flow_expand=2,
        use_pos_enc=True,
        use_shortcut=True,
        shortcut_scale_init=0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.recent_window = recent_window
        self.use_pos_enc = use_pos_enc
        self.use_shortcut = use_shortcut

        # -----------------------------
        # shared input encoder
        # -----------------------------
        self.embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model)

        # -----------------------------
        # local stem
        # -----------------------------
        k1, k2 = local_kernel_sizes
        self.local_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k1, padding=k1 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=k2, padding=k2 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -----------------------------
        # Mamba backbone
        # -----------------------------
        self.mamba_blocks = nn.ModuleList([
            BetterMambaBlock(
                d_model=d_model,
                kernel_size=mamba_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_mamba_layers)
        ])
        self.mamba_norm = nn.LayerNorm(d_model)

        # -----------------------------
        # Flow refine branch
        # z = z_m + alpha * refine(z_m)
        # -----------------------------
        self.flow_in_norm = nn.LayerNorm(d_model)
        self.flow_attn = FlowAttention(d_model, n_heads=n_heads, dropout=dropout)

        self.flow_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * flow_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * flow_expand, d_model),
            nn.Dropout(dropout),
        )
        self.flow_out_norm = nn.LayerNorm(d_model)

        # 轻量门控：控制 refine 强度，而不是和主干做平权融合
        self.refine_gate = nn.Sequential(
            nn.Linear(d_model + input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # 可学习缩放，初始较小，让模型先信任 Mamba 主干
        self.refine_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        # -----------------------------
        # prediction head
        # -----------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2+ input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        # -----------------------------
        # weaker shortcut
        # 只从最后时刻原始输入学一个较弱基线
        # -----------------------------
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, pred_len),
            )
            self.shortcut_scale = nn.Parameter(
                torch.tensor(shortcut_scale_init, dtype=torch.float32)
            )

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, pred_len]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        x_raw = x

        # --------------------------------
        # shared encoder
        # --------------------------------
        z = self.embedding(x)
        z = self.input_norm(z)
        if self.use_pos_enc:
            z = self.pos_enc(z)
        z = self.input_dropout(z)

        # --------------------------------
        # local stem
        # --------------------------------
        z = z.transpose(1, 2)   # [B, D, T]
        z = self.local_stem(z)
        z = z.transpose(1, 2)   # [B, T, D]

        # --------------------------------
        # Mamba backbone
        # --------------------------------
        z_m = z
        for blk in self.mamba_blocks:
            z_m = z_m + blk(z_m)
        z_m = self.mamba_norm(z_m)

        # --------------------------------
        # Flow refine
        # refine(z_m) = Attn + FFN
        # z = z_m + alpha * gate * refine
        # --------------------------------
        refine = self.flow_attn(self.flow_in_norm(z_m))
        refine = refine + self.flow_ffn(refine)
        refine = self.flow_out_norm(refine)

        m_last = z_m[:, -1, :]
        last_raw = x_raw[:, -1, :]
        gate_in = torch.cat([m_last, last_raw], dim=-1)
        gate = self.refine_gate(gate_in).unsqueeze(1)   # [B,1,1]

        z = z_m + self.refine_scale * gate * refine

        # --------------------------------
        # readout
        # --------------------------------
        last_token = z[:, -1, :]
        rw = min(self.recent_window, z.size(1))
        recent_pool = z[:, -rw:, :].mean(dim=1)
        last_raw = x_raw[:, -1, :]

        # recent_max = z[:, -rw:, :].max(dim=1).values
        feat = torch.cat([last_token, recent_pool, last_raw], dim=-1)
        # print(feat.shape)
        # exit()
        delta = self.head(feat)

        if self.use_shortcut:
            base = self.shortcut(last_raw)
            out = delta + self.shortcut_scale * base
        else:
            out = delta

        return out
#################################################

 
########################################
class WindPowerPureMamba(nn.Module):
    """
    纯 Mamba 主干模型：
        x
        -> embedding + norm + pos_enc
        -> local_stem
        -> stacked BetterMambaBlock
        -> final_norm
        -> [last_token, recent_pool, last_raw]
        -> head
        -> optional shortcut
        -> out

    说明：
    1. 不使用 GRU
    2. 不使用 FlowAttention
    3. 保留 local_stem + Mamba 主干
    4. 默认不使用 persistence，避免 X / y 分别标准化导致的尺度错位
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        num_layers=2,
        pred_len=6,
        dropout=0.05,
        recent_window=12,
        use_pos_enc=True,
        use_shortcut=True,
        local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,
    ):
        super().__init__()

        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.recent_window = recent_window
        self.use_pos_enc = use_pos_enc
        self.use_shortcut = use_shortcut

        # 输入编码
        self.embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model)

        # 局部卷积 stem
        k1, k2 = local_kernel_sizes
        self.local_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k1, padding=k1 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=k2, padding=k2 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 纯 Mamba 主干
        self.mamba_blocks = nn.ModuleList([
            BetterMambaBlock(
                d_model=d_model,
                kernel_size=mamba_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 预测头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        if self.use_shortcut:
            self.shortcut = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, pred_len]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        x_raw = x

        # shared encoder
        z = self.embedding(x)
        z = self.input_norm(z)
        if self.use_pos_enc:
            z = self.pos_enc(z)
        z = self.input_dropout(z)

        # local stem
        z = z.transpose(1, 2)   # [B, D, T]
        z = self.local_stem(z)
        z = z.transpose(1, 2)   # [B, T, D]

        # stacked mamba blocks
        for blk in self.mamba_blocks:
            z = z + blk(z)
        z = self.final_norm(z)

        # readout
        last_token = z[:, -1, :]
        rw = min(self.recent_window, z.size(1))
        recent_pool = z[:, -rw:, :].mean(dim=1)
        last_raw = x_raw[:, -1, :]

        feat = torch.cat([last_token, recent_pool, last_raw], dim=-1)
        out = self.head(feat)

        if self.use_shortcut:
            out = out + self.shortcut(last_raw)

        return out


########################################
# 优化版模型：WindPowerMambaFlowLiteV2
# 核心改进：
# 1. 多尺度时序特征提取
# 2. LayerScale 稳定训练
# 3. SwiGLU 激活函数
# 4. 增强 FlowAttention 贡献
########################################

class LayerScale(nn.Module):
    """
    LayerScale: 可学习的缩放参数，用于稳定深层网络训练
    参考: "Going deeper with Image Transformers" (Touvron et al., 2021)
    """
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数: Swish + GLU 组合
    参考: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2
        
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU(x) = Swish(xW1) ⊗ (xW2)
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MultiScaleTemporalPool(nn.Module):
    """
    多尺度时序池化: 从不同时间窗口提取特征
    """
    def __init__(self, d_model, windows=[6, 12, 24], dropout=0.1):
        super().__init__()
        self.windows = windows
        self.num_windows = len(windows)
        self.d_model = d_model
        
        # 计算每个尺度的输出维度，确保总和等于 d_model
        # 使用 ceil 确保维度足够
        self.proj_dims = []
        remaining = d_model
        for i in range(self.num_windows - 1):
            dim = d_model // self.num_windows
            self.proj_dims.append(dim)
            remaining -= dim
        self.proj_dims.append(remaining)  # 最后一个取剩余维度
        
        # 每个尺度一个独立的投影层
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.proj_dims[i]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for i in range(self.num_windows)
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z):
        """
        z: [B, T, D]
        return: [B, D]
        """
        pooled_features = []
        
        for i, w in enumerate(self.windows):
            rw = min(w, z.size(1))
            pool = z[:, -rw:, :].mean(dim=1)  # [B, D]
            proj = self.projections[i](pool)   # [B, proj_dims[i]]
            pooled_features.append(proj)
        
        # 拼接所有尺度的特征
        multi_scale = torch.cat(pooled_features, dim=-1)  # [B, D]
        
        return self.fusion(multi_scale)


class WindPowerMambaFlowLiteV2(nn.Module):
    """
    优化版风电预测模型：
    
    核心改进：
    1. 多尺度时序特征提取 - 捕获不同时间粒度的模式
    2. LayerScale - 稳定深层网络训练
    3. SwiGLU 激活函数 - 更好的非线性表达能力
    4. 增强 FlowAttention 贡献 - 更大的初始 refine_scale
    5. 改进的预测头 - 残差连接 + 更深的网络
    
    架构流程：
        x
        -> embedding + norm + pos_enc
        -> local_stem (局部卷积)
        -> Mamba backbone (时序建模)
        -> Flow refine (全局注意力修正)
        -> Multi-scale temporal pooling
        -> Enhanced prediction head
        -> out
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_mamba_layers=2,
        pred_len=6,
        dropout=0.1,
        recent_windows=[6, 12, 24],  # 多尺度历史窗口
        local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,
        flow_expand=2,
        use_pos_enc=True,
        use_shortcut=True,
        shortcut_scale_init=0.1,
        refine_scale_init=0.2,  # 增大初始值
        layer_scale_init=1e-5,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.recent_windows = recent_windows
        self.use_pos_enc = use_pos_enc
        self.use_shortcut = use_shortcut

        # -----------------------------
        # shared input encoder
        # -----------------------------
        self.embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model)

        # -----------------------------
        # local stem (局部卷积)
        # -----------------------------
        k1, k2 = local_kernel_sizes
        self.local_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k1, padding=k1 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=k2, padding=k2 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -----------------------------
        # Mamba backbone (时序建模)
        # -----------------------------
        self.mamba_blocks = nn.ModuleList([
            BetterMambaBlock(
                d_model=d_model,
                kernel_size=mamba_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_mamba_layers)
        ])
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba_ls = LayerScale(d_model, init_value=layer_scale_init)

        # -----------------------------
        # Flow refine branch (全局注意力修正)
        # -----------------------------
        self.flow_in_norm = nn.LayerNorm(d_model)
        self.flow_attn = FlowAttention(d_model, n_heads=n_heads, dropout=dropout)
        
        # 使用 SwiGLU 替代普通 FFN
        self.flow_ffn = SwiGLU(d_model, hidden_dim=d_model * flow_expand, dropout=dropout)
        self.flow_out_norm = nn.LayerNorm(d_model)
        self.flow_ls = LayerScale(d_model, init_value=layer_scale_init)

        # 可学习的 refine_scale，初始值更大
        self.refine_scale = nn.Parameter(torch.tensor(refine_scale_init, dtype=torch.float32))

        # -----------------------------
        # Multi-scale temporal pooling
        # -----------------------------
        self.multi_scale_pool = MultiScaleTemporalPool(
            d_model=d_model,
            windows=recent_windows,
            dropout=dropout
        )

        # -----------------------------
        # Enhanced prediction head
        # -----------------------------
        # 输入维度: last_token (d_model) + multi_scale_pool (d_model) + last_raw (input_dim)
        head_input_dim = d_model * 2 + input_dim
        
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        # -----------------------------
        # Weaker shortcut (弱化基线)
        # -----------------------------
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, pred_len),
            )
            self.shortcut_scale = nn.Parameter(
                torch.tensor(shortcut_scale_init, dtype=torch.float32)
            )

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, pred_len]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        x_raw = x

        # --------------------------------
        # shared encoder
        # --------------------------------
        z = self.embedding(x)
        z = self.input_norm(z)
        if self.use_pos_enc:
            z = self.pos_enc(z)
        z = self.input_dropout(z)

        # --------------------------------
        # local stem
        # --------------------------------
        z = z.transpose(1, 2)   # [B, D, T]
        z = self.local_stem(z)
        z = z.transpose(1, 2)   # [B, T, D]

        # --------------------------------
        # Mamba backbone
        # --------------------------------
        z_m = z
        for blk in self.mamba_blocks:
            z_m = z_m + self.mamba_ls(blk(z_m))
        z_m = self.mamba_norm(z_m)

        # --------------------------------
        # Flow refine (增强版)
        # z = z_m + refine_scale * (Attn + FFN)
        # --------------------------------
        refine = self.flow_attn(self.flow_in_norm(z_m))
        refine = refine + self.flow_ffn(refine)
        refine = self.flow_out_norm(refine)
        refine = self.flow_ls(refine)

        # 使用可学习的 refine_scale
        z = z_m + self.refine_scale * refine

        # --------------------------------
        # Multi-scale temporal pooling
        # --------------------------------
        last_token = z[:, -1, :]
        multi_scale_feat = self.multi_scale_pool(z)
        last_raw = x_raw[:, -1, :]

        # --------------------------------
        # Enhanced prediction head
        # --------------------------------
        feat = torch.cat([last_token, multi_scale_feat, last_raw], dim=-1)
        delta = self.head(feat)

        if self.use_shortcut:
            base = self.shortcut(last_raw)
            out = delta + self.shortcut_scale * base
        else:
            out = delta

        return out


########################################
# V3 优化版模型：WindPowerMambaFlowLiteV3
# 核心改进：
# 1. 时序注意力增强模块
# 2. 双路特征融合（Mamba + GRU）
# 3. 改进的预测头（残差连接 + Skip Connection）
# 4. 更强的正则化
########################################

class TemporalAttentionEnhance(nn.Module):
    """
    时序注意力增强模块：捕获关键时刻的重要性
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        q = self.query(x)  # [B, T, D]
        k = self.key(x)
        v = self.value(x)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, T, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, T, D]
        out = self.norm(out + x)
        return out


class DualPathFusion(nn.Module):
    """
    双路特征融合：Mamba路径 + GRU路径
    """
    def __init__(self, d_model, gru_hidden=None, dropout=0.1):
        super().__init__()
        if gru_hidden is None:
            gru_hidden = d_model
            
        # GRU路径
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.gru_proj = nn.Linear(gru_hidden, d_model) if gru_hidden != d_model else nn.Identity()
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z_mamba, z_raw):
        """
        z_mamba: [B, T, D] - Mamba输出
        z_raw: [B, T, D] - 原始特征（embedding后）
        return: [B, T, D]
        """
        # GRU路径
        gru_out, _ = self.gru(z_raw)
        gru_out = self.gru_proj(gru_out)
        
        # 门控融合
        concat = torch.cat([z_mamba, gru_out], dim=-1)
        gate = self.gate(concat)
        
        out = gate * z_mamba + (1 - gate) * gru_out
        out = self.norm(out)
        return self.dropout(out)


class WindPowerMambaFlowLiteV3(nn.Module):
    """
    V3 优化版风电预测模型（增强版GRU）：
    
    核心改进：
    1. 基于GRU的稳定架构
    2. 添加位置编码和LayerNorm
    3. 多尺度特征融合
    4. 简化的预测头
    
    架构流程：
        x
        -> embedding + norm + pos_enc
        -> GRU backbone (时序建模)
        -> multi-scale pooling
        -> prediction head
        -> out
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_mamba_layers=2,
        pred_len=6,
        dropout=0.1,
        recent_windows=[6, 12, 24],
        local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,
        flow_expand=2,
        use_pos_enc=True,
        use_shortcut=True,
        shortcut_scale_init=0.1,
        refine_scale_init=0.2,
        layer_scale_init=1e-5,
        gru_hidden=None,
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.recent_windows = recent_windows
        self.use_shortcut = use_shortcut

        # -----------------------------
        # Input encoder
        # -----------------------------
        self.embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_enc = PositionalEncoding(d_model)

        # -----------------------------
        # GRU backbone (主时序建模)
        # -----------------------------
        gru_h = gru_hidden if gru_hidden else d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_h,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_mamba_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.gru_proj = nn.Linear(gru_h, d_model) if gru_h != d_model else nn.Identity()
        self.gru_norm = nn.LayerNorm(d_model)

        # -----------------------------
        # Multi-scale temporal pooling
        # -----------------------------
        self.multi_scale_pool = MultiScaleTemporalPool(
            d_model=d_model,
            windows=recent_windows,
            dropout=dropout
        )

        # -----------------------------
        # Simplified prediction head
        # -----------------------------
        head_input_dim = d_model * 2 + input_dim
        
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        # -----------------------------
        # Shortcut
        # -----------------------------
        if self.use_shortcut:
            self.shortcut = nn.Linear(input_dim, pred_len)
            self.shortcut_scale = nn.Parameter(
                torch.tensor(shortcut_scale_init, dtype=torch.float32)
            )
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, pred_len]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")

        x_raw = x

        # --------------------------------
        # Input encoder
        # --------------------------------
        z = self.embedding(x)
        z = self.input_norm(z)
        z = self.pos_enc(z)
        z = self.input_dropout(z)

        # --------------------------------
        # GRU backbone
        # --------------------------------
        gru_out, h_n = self.gru(z)
        z = self.gru_proj(gru_out)
        z = self.gru_norm(z)

        # --------------------------------
        # Multi-scale temporal pooling
        # --------------------------------
        last_token = z[:, -1, :]
        multi_scale_feat = self.multi_scale_pool(z)
        last_raw = x_raw[:, -1, :]

        # --------------------------------
        # prediction head
        # --------------------------------
        feat = torch.cat([last_token, multi_scale_feat, last_raw], dim=-1)
        delta = self.head(feat)

        if self.use_shortcut:
            base = self.shortcut(last_raw)
            out = delta + self.shortcut_scale * base
        else:
            out = delta

        return out


class WindPowerMambaFlowLiteV4(nn.Module):
    """
    V4 增强型GRU风电预测模型：
    
    设计理念：在保持GRU简洁性的基础上，添加轻量级增强
    核心改进：
    1. 输入层归一化 - 稳定训练
    2. 双向GRU - 捕获更丰富的时序特征
    3. 时序注意力 - 自适应加权不同时间步
    4. 残差预测头 - 利用最后时刻的原始特征
    
    架构流程：
        x -> LayerNorm -> BiGRU -> Attention -> concat(last_raw) -> Head -> out
    """
    
    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_mamba_layers=2,
        pred_len=6,
        dropout=0.1,
        recent_windows=[6, 12, 24],
        local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,
        flow_expand=2,
        use_pos_enc=True,
        use_shortcut=True,
        shortcut_scale_init=0.1,
        refine_scale_init=0.2,
        layer_scale_init=1e-5,
        gru_hidden=None,
        drop_path_rate=0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.use_shortcut = use_shortcut
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 双向GRU
        gru_h = gru_hidden if gru_hidden else d_model
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_h,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_mamba_layers > 1 else 0.0,
            bidirectional=True,  # 双向
        )
        self.gru_out_dim = gru_h * 2  # 双向输出维度
        
        # 时序注意力：自适应加权不同时间步
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.gru_out_dim, self.gru_out_dim // 2),
            nn.Tanh(),
            nn.Linear(self.gru_out_dim // 2, 1),
        )
        
        # 输出投影
        self.gru_proj = nn.Linear(self.gru_out_dim, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        
        # 预测头
        head_input_dim = d_model + input_dim  # GRU特征 + 最后时刻原始特征
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        
        # 残差连接
        if self.use_shortcut:
            self.shortcut = nn.Linear(input_dim, pred_len)
            self.shortcut_scale = nn.Parameter(
                torch.tensor(shortcut_scale_init, dtype=torch.float32)
            )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, pred_len]
        """
        B, T, D = x.shape
        
        # 保存原始输入的最后时刻
        last_raw = x[:, -1, :]
        
        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 双向GRU
        gru_out, h_n = self.gru(x_norm)  # gru_out: [B, T, gru_h*2]
        
        # 时序注意力
        attn_weights = self.temporal_attn(gru_out)  # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # 在时间维度上归一化
        context = (gru_out * attn_weights).sum(dim=1)  # [B, gru_h*2]
        
        # 投影和归一化
        feat = self.gru_proj(context)  # [B, d_model]
        feat = self.output_norm(feat)
        
        # 拼接原始特征
        combined = torch.cat([feat, last_raw], dim=-1)
        
        # 预测
        delta = self.head(combined)
        
        # 残差连接
        if self.use_shortcut:
            base = self.shortcut(last_raw)
            out = delta + self.shortcut_scale * base
        else:
            out = delta
        
        return out


class WindPowerMambaFlowLiteV5(nn.Module):
    """
    V5 简洁增强型GRU风电预测模型：
    
    设计理念：最小改动原则，在GRU基础上只添加必要的增强
    核心改进：
    1. 输入层归一化 - 稳定训练
    2. 单向GRU - 保持时序因果性
    3. 最后时刻特征 + 全局平均池化 - 双路特征
    4. 残差预测头 - 利用最后时刻的原始特征
    
    架构流程：
        x -> LayerNorm -> GRU -> [last_h, mean_pool] -> concat(last_raw) -> Head -> out
    """
    
    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_mamba_layers=2,
        pred_len=6,
        dropout=0.1,
        recent_windows=[6, 12, 24],
        local_kernel_sizes=(3, 5),
        mamba_kernel_size=5,
        flow_expand=2,
        use_pos_enc=True,
        use_shortcut=True,
        shortcut_scale_init=0.1,
        refine_scale_init=0.2,
        layer_scale_init=1e-5,
        gru_hidden=None,
        drop_path_rate=0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.use_shortcut = use_shortcut
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 单向GRU（保持因果性）
        gru_h = gru_hidden if gru_hidden else d_model
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_h,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_mamba_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.gru_h = gru_h
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(gru_h)
        
        # 预测头：last_h + mean_pool + last_raw
        head_input_dim = gru_h * 2 + input_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, gru_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_h, pred_len),
        )
        
        # 残差连接
        if self.use_shortcut:
            self.shortcut = nn.Linear(input_dim, pred_len)
            self.shortcut_scale = nn.Parameter(
                torch.tensor(shortcut_scale_init, dtype=torch.float32)
            )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, pred_len]
        """
        B, T, D = x.shape
        
        # 保存原始输入的最后时刻
        last_raw = x[:, -1, :]
        
        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 单向GRU
        gru_out, h_n = self.gru(x_norm)  # gru_out: [B, T, gru_h], h_n: [num_layers, B, gru_h]
        
        # 双路特征：最后隐藏状态 + 全局平均池化
        last_h = h_n[-1]  # [B, gru_h]
        mean_pool = gru_out.mean(dim=1)  # [B, gru_h]
        
        # 归一化
        last_h = self.output_norm(last_h)
        mean_pool = self.output_norm(mean_pool)
        
        # 拼接特征
        combined = torch.cat([last_h, mean_pool, last_raw], dim=-1)
        
        # 预测
        delta = self.head(combined)
        
        # 残差连接
        if self.use_shortcut:
            base = self.shortcut(last_raw)
            out = delta + self.shortcut_scale * base
        else:
            out = delta
        return out


class SimplifiedMamba(nn.Module):
    """
    纯 PyTorch 版 Mamba-like 模块。

    说明:
    - 结构尽量贴近 Mamba Block:
      in_proj -> split(x, z)
      x -> depthwise causal conv -> SiLU
      x -> 生成输入相关的 dt, B, C
      selective scan
      y -> gate(z) -> out_proj

    - 不依赖 mamba_ssm / Triton / CUDA selective_scan kernel。
    - 速度不如官方 fused selective_scan，但逻辑更接近 Mamba。
    - 输入输出形状均为 [B, L, D]，可直接替换原 SimplifiedMamba。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        dt_rank: int | str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = int(dt_rank)

        # 1. Mamba 输入投影：生成 x 分支和 z 门控分支
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 2. depthwise causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        self.act = nn.SiLU()

        # 3. 根据输入 x 生成 selective SSM 参数:
        #    dt: [B, L, dt_rank]
        #    B : [B, L, d_state]
        #    C : [B, L, d_state]
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )

        # dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 4. 连续状态矩阵 A，实际使用 A = -exp(A_log)，保证稳定负实部
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # 5. skip connection 参数 D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 6. 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._init_weights(dt_min, dt_max, dt_init_floor)

    def _init_weights(self, dt_min: float, dt_max: float, dt_init_floor: float):
        # 输入输出投影初始化
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        nn.init.xavier_uniform_(self.x_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Conv 初始化
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # dt_proj 初始化：让 softplus(dt_bias) 落在合理范围
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        # softplus 的近似反函数
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # 标记，避免外部统一初始化时把 dt_proj.bias 清零
        self.dt_proj.bias._no_reinit = True

    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        x : [B, L, d_inner]
        dt: [B, L, d_inner]
        B : [B, L, d_state]
        C : [B, L, d_state]

        return:
            y: [B, L, d_inner]
        """
        batch, seq_len, d_inner = x.shape

        # A: [d_inner, d_state]
        A = -torch.exp(self.A_log.float())

        # hidden state: [B, d_inner, d_state]
        state = x.new_zeros(batch, d_inner, self.d_state)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]          # [B, d_inner]
            dt_t = dt[:, t, :]        # [B, d_inner]
            B_t = B[:, t, :]          # [B, d_state]
            C_t = C[:, t, :]          # [B, d_state]

            # 离散化:
            # dA: [B, d_inner, d_state]
            # dB: [B, d_inner, d_state]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            # 状态更新
            state = state * dA + x_t.unsqueeze(-1) * dB

            # 输出 y_t: [B, d_inner]
            y_t = torch.sum(state * C_t.unsqueeze(1), dim=-1)

            # D skip
            y_t = y_t + self.D * x_t

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]

        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape

        # 1. 输入投影并切分成主分支 x 和门控分支 z
        xz = self.in_proj(x)  # [B, L, 2 * d_inner]
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # 2. depthwise causal conv
        x_branch = x_branch.transpose(1, 2)  # [B, d_inner, L]
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = x_branch.transpose(1, 2)  # [B, L, d_inner]
        x_branch = self.act(x_branch)

        # 3. 生成 selective SSM 参数
        x_dbl = self.x_proj(x_branch)

        dt, B_ssm, C_ssm = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # dt: [B, L, d_inner]，保持正值
        dt = F.softplus(self.dt_proj(dt))

        # 4. selective scan
        y = self.selective_scan(x_branch, dt, B_ssm, C_ssm)

        # 5. 门控
        y = y * self.act(z_branch)

        # 6. 输出投影
        out = self.out_proj(y)
        out = self.dropout(out)

        return out