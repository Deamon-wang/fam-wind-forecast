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
from src.VQC_cir_chip import Angle_VQC_chip
pyvqnet.backends.set_backend("torch")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale


class FlowAttention(nn.Module):
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
        bsz, t, _ = x.shape
        x = x.view(bsz, t, self.n_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x):
        bsz, heads, t, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, t, heads * head_dim)

    def forward(self, x):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = F.softplus(q * self.scale) + self.eps
        k = F.softplus(k) + self.eps
        k_sum = k.sum(dim=2)
        q_sum = q.sum(dim=2)
        row_norm = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k_sum) + self.eps)
        col_norm = 1.0 / (torch.einsum("bhtd,bhd->bht", k, q_sum) + self.eps)
        weighted_k = k * col_norm.unsqueeze(-1)
        kv = torch.einsum("bhtd,bhte->bhde", weighted_k, v)
        out = torch.einsum("bhtd,bhde->bhte", q, kv)
        out = out * row_norm.unsqueeze(-1)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class BetterMambaBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.1):
        super().__init__()
        # self.in_proj = nn.Linear(d_model, d_model * 2)
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

        n_bit, qlayer = 6,1
        self.W1 = tensor.randn([d_model, n_bit], requires_grad = True, device=pyvqnet.DEV_GPU) 
        
        self.Angle_VQC_cir1 = Angle_VQC_chip(n_bit, qlayer).toGPU() if torch.cuda.is_available() else Angle_VQC_chip(n_bit, qlayer)
        self.linear_post = nn.Linear(n_bit, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # u, v = self.in_proj(x).chunk(2, dim=-1)
        q_out=self.Angle_VQC_cir1(self.W1) 
        q_out = tensor.matmul(x,q_out)
        u = self.linear_post(q_out.data)
        v = self.v_proj(u)

        v = v.transpose(1, 2)
        v = self.dw_conv(v)
        v = v[:, :, :x.size(1)]
        v = self.pw_conv(v)
        v = v.transpose(1, 2)
        v = torch.sigmoid(self.norm(v))
        out = self.out_proj(u * v)
        return self.dropout(out)


class BetterTransformerMambaBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, mamba_kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FlowAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mamba = BetterMambaBlock(d_model, kernel_size=mamba_kernel_size, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.mamba(self.norm2(x)))
        x = x + self.ffn(self.norm3(x))
        return x



class WindPowerQTransformer(nn.Module):
    def __init__(self,input_dim,d_model=64, n_heads=4,num_layers=2,pred_len=1,dropout=0.1,
        power_idx=None,recent_window=6,use_persistence=True,local_kernel_sizes=(3, 5),mamba_kernel_size=3,
    qbits = 6,qlayers = 1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
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
        self.blocks = nn.ModuleList([
            BetterTransformerMambaBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                mamba_kernel_size=mamba_kernel_size
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )
        self.shortcut = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B, T, D), got {tuple(x.shape)}")
        _, t, d = x.shape
        if d != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {d}")
        x_raw = x
        x = self.embedding(x)
        x = self.input_dropout(x)
        x = self.pos_enc(x)
        x = x.transpose(1, 2)
        x = self.local_stem(x)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        last_token = x[:, -1, :]
        rw = min(self.recent_window, t)
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
        out = base + delta
        return out




class WindPowerQMambaFlowLite(nn.Module):
    """
   简洁增强型GRU风电预测模型：
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





class QWindPowerMambaFlowLiteV5(nn.Module):
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
        
        # 量子残差连接
        qbits = 6 
        qlayer =1

        if self.use_shortcut:
            self.linear_pre =  nn.Linear(input_dim, qbits)
            self.Angle_VQC = Angle_VQC_chip(qbits, qlayer).toGPU() if torch.cuda.is_available() else Angle_VQC_chip(qbits, qlayer)
            self.linear_post =  nn.Linear(qbits,pred_len)

            # self.shortcut = nn.Linear(input_dim, pred_len)
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
            # base = self.shortcut(last_raw)
            base1 = self.linear_pre(last_raw)
            base_vqc = self.Angle_VQC(base1)
            base = self.linear_post(base_vqc.data)
            out = delta + self.shortcut_scale * base
        else:
            out = delta
        
        return out