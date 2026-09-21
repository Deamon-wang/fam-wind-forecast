"""
优化版 Wind-Mambaformer 模型实现

包含:
- PositionalEncoding: 位置编码
- FlowAttention: 流注意力 (线性复杂度)
- SimplifiedMamba: 简化版Mamba实现
- WindMambaformer: 优化版Wind-Mambaformer主模型
- BaselineLSTM: LSTM基线模型
- BaselineTransformer: Transformer基线模型
- BaselineMLP: MLP基线模型

改进点:
1. 添加位置编码
2. 多层 Flow Attention + Mamba 堆叠
3. 改进的门控融合机制
4. 更深的预测头
5. 多尺度特征提取
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 mamba_ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using simplified Mamba implementation")


# =============================================================================
# 位置编码
# =============================================================================
class PositionalEncoding(nn.Module):
    """位置编码 - 为序列添加位置信息"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# Flow Attention（官方实现，线性复杂度）
# =============================================================================
class FlowAttention(nn.Module):
    """Flow Attention - 线性复杂度注意力机制
    
    参考: Wind-Mambaformer 论文
    """
    
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """核函数：使用 sigmoid"""
        return torch.sigmoid(x)

    def my_sum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """求和操作: nhld,nhd->nhl"""
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D_head]
        
        # kernel
        q = self.kernel(q)
        k = self.kernel(k)
        
        # normalizer
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        
        # allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        
        # competition
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])
        
        # multiply
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        
        x = x_update.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


# =============================================================================
# 简化版 Mamba（当 mamba_ssm 不可用时使用）
# =============================================================================
class SimplifiedMamba(nn.Module):
    """简化版 Mamba 实现，用于替代 mamba_ssm.Mamba
    
    使用更稳定的实现方式，避免 NaN 问题
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        # 输入投影 (使用更小的初始化)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        
        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # LayerNorm 用于稳定训练
        self.norm = nn.LayerNorm(self.d_inner)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化 (使用更小的值)
        self._init_weights()
    
    def _init_weights(self):
        # 使用较小的初始化值防止梯度爆炸
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        # 卷积层初始化
        nn.init.kaiming_uniform_(self.conv1d.weight, a=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape
        
        # 输入投影
        x_proj = self.in_proj(x)  # [B, L, d_inner]
        
        # 卷积
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)
        
        # LayerNorm 稳定训练
        x_norm = self.norm(x_conv)
        
        # 输出
        out = self.out_proj(x_norm)
        out = self.dropout(out)
        
        return out


# =============================================================================
# Wind-Mambaformer 优化版主干网络
# =============================================================================
class WindMambaformer(nn.Module):
    """
    优化版 Wind-Mambaformer
    
    结构:
    1. 输入投影层
    2. 位置编码
    3. 多层 Flow Attention + Mamba 双分支堆叠
    4. 门控融合机制
    5. 多尺度时间特征提取
    6. 全局特征聚合
    7. 预测头
    
    改进点:
    1. 添加位置编码
    2. 多层 Flow Attention + Mamba 堆叠
    3. 改进的门控融合机制
    4. 更深的预测头
    5. 多尺度特征提取
    """
    
    def __init__(
        self,
        input_dim: int = 10,          # 输入特征维度
        model_dim: int = 128,         # 模型隐藏维度
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        flow_attn_heads: int = 8,
        pred_horizon: int = 16,       # 预测未来多少步
        dropout: float = 0.1,
        use_mamba_ssm: bool = True,   # 是否使用 mamba_ssm 库
        num_layers: int = 2,          # 堆叠层数
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_layers = num_layers

        # 输入投影
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model=model_dim, dropout=dropout)

        # 多层 Flow Attention 分支
        self.flow_attn_layers = nn.ModuleList([
            FlowAttention(dim=model_dim, num_heads=flow_attn_heads, attn_drop=dropout)
            for _ in range(num_layers)
        ])
        self.flow_norms = nn.ModuleList([
            nn.LayerNorm(model_dim) for _ in range(num_layers)
        ])

        # 多层 Mamba 分支
        if use_mamba_ssm and MAMBA_AVAILABLE:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=model_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                for _ in range(num_layers)
            ])
        else:
            self.mamba_layers = nn.ModuleList([
                SimplifiedMamba(d_model=model_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, 
                               expand=mamba_expand, dropout=dropout)
                for _ in range(num_layers)
            ])
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(model_dim) for _ in range(num_layers)
        ])

        # 多层融合门控
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        self.fusion_norms = nn.ModuleList([
            nn.LayerNorm(model_dim) for _ in range(num_layers)
        ])

        # 多尺度时间特征提取
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(model_dim, model_dim, kernel_size=k, padding=k//2, groups=model_dim)
            for k in [1, 3, 5, 7]
        ])
        self.scale_weight = nn.Parameter(torch.ones(4) / 4)
        
        # 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 改进的预测头
        # self.pred_head = nn.Sequential(
        #     nn.Linear(model_dim * 2, model_dim * 2),  # *2 因为 avg + max pooling 拼接
        #     nn.LayerNorm(model_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(model_dim * 2, model_dim),
        #     nn.LayerNorm(model_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(model_dim, pred_horizon)
        # )
        self.pred_head = nn.Sequential(
            nn.Linear(model_dim * 2, int(model_dim)),  # *2 因为 avg + max pooling 拼接
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(model_dim), pred_horizon),
            # nn.LayerNorm(model_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(model_dim, pred_horizon)
        )

        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim] —— 历史多变量时间序列
        returns: [B, pred_horizon] —— 未来功率预测
        """
        B, T, _ = x.shape
        
        # 输入嵌入
        x = self.input_proj(x)  # [B, T, D]
        x = self.pos_encoding(x)  # 添加位置编码
        x = self.dropout(x)

        # 多层双分支处理
        for i in range(self.num_layers):
            # Flow Attention 分支
            f_out = self.flow_attn_layers[i](x)
            f_out = self.flow_norms[i](f_out + x)  # 残差连接
            
            # Mamba 分支
            m_out = self.mamba_layers[i](x)
            m_out = self.mamba_norms[i](m_out + x)  # 残差连接
            
            # 门控融合
            concat = torch.cat([f_out, m_out], dim=-1)
            gate = self.gate_layers[i](concat)
            fused = gate * f_out + (1 - gate) * m_out
            fused = self.fusion_norms[i](fused + x)  # 残差连接
            
            x = fused  # 更新 x 用于下一层

        # 多尺度时间特征提取
        x_transposed = x.transpose(1, 2)  # [B, D, T]
        scale_weights = F.softmax(self.scale_weight, dim=0)
        multi_scale_features = []
        for i, conv in enumerate(self.multi_scale_convs):
            scale_feat = conv(x_transposed)  # [B, D, T]
            multi_scale_features.append(scale_feat * scale_weights[i])
        x_multi_scale = sum(multi_scale_features)  # [B, D, T]
        
        # 全局特征聚合 (avg + max pooling)
        avg_pool = self.global_pool(x_multi_scale).squeeze(-1)  # [B, D]
        max_pool = self.global_max_pool(x_multi_scale).squeeze(-1)  # [B, D]
        global_feat = torch.cat([avg_pool, max_pool], dim=-1)  # [B, 2D]

        # 预测
        y_hat = self.pred_head(global_feat)  # [B, pred_horizon]

        return y_hat


# =============================================================================
# 基线模型
# =============================================================================

class BaselineLSTM(nn.Module):
    """LSTM 基准模型"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_horizon: int = 16,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_horizon = pred_horizon
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_horizon)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置 forget gate bias 为 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, input_dim]
        Returns:
            y: [B, pred_horizon]
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, L, hidden_dim]
        # 使用最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # [B, hidden_dim]
        # 预测
        y = self.fc(last_out)  # [B, pred_horizon]
        
        return y


class BaselineTransformer(nn.Module):
    """Transformer 基准模型"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 3,
        pred_horizon: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 输入嵌入
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_horizon)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, input_dim]
        Returns:
            y: [B, pred_horizon]
        """
        # 输入嵌入
        x = self.input_proj(x)  # [B, L, d_model]
        
        # 位置编码
        x = self.pos_encoding(x)  # [B, L, d_model]
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # [B, L, d_model]
        
        # 取最后一个时间步
        x = x[:, -1, :]  # [B, d_model]
        
        # 输出
        y = self.fc(x)  # [B, pred_horizon]
        
        return y


class BaselineMLP(nn.Module):
    """MLP 基准模型 - 将序列展平后使用全连接层"""
    
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 96,
        hidden_dim: int = 256,
        pred_horizon: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 输入展平后的维度
        flat_dim = input_dim * seq_len
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, pred_horizon)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, input_dim]
        Returns:
            y: [B, pred_horizon]
        """
        B, L, D = x.shape
        
        # 展平序列
        x = x.reshape(B, L * D)  # [B, L * D]
        
        # 全连接层
        y = self.fc(x)  # [B, pred_horizon]
        
        return y


# =============================================================================
# 工具函数
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name: str, input_dim: int, seq_len: int = 96, **kwargs) -> nn.Module:
    """
    根据名称获取模型
    
    Args:
        model_name: 模型名称 ('wind_mambaformer', 'lstm', 'transformer', 'mlp')
        input_dim: 输入特征维度
        seq_len: 序列长度 (MLP需要)
        **kwargs: 其他模型参数
    
    Returns:
        model: PyTorch模型
    """
    model_name = model_name.lower()
    
    if model_name == 'wind_mambaformer':
        return WindMambaformer(input_dim=input_dim, **kwargs)
    elif model_name == 'lstm':
        return BaselineLSTM(input_dim=input_dim, **kwargs)
    elif model_name == 'transformer':
        return BaselineTransformer(input_dim=input_dim, **kwargs)
    elif model_name == 'mlp':
        return BaselineMLP(input_dim=input_dim, seq_len=seq_len, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available: 'wind_mambaformer', 'lstm', 'transformer', 'mlp'")