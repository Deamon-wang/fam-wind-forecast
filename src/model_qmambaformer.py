"""
WindQMambaformer - 量子 Mambaformer 模型实现

包含:
- PositionalEncoding: 位置编码
- FlowQAttention: 量子 Flow 注意力 (线性复杂度)
- SimplifiedMamba: 简化版 Mamba 实现
- WindQMambaformer: 量子 Wind-Mambaformer 主模型
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using simplified Mamba implementation")

try:
    from src.VQC_cir_chip import Angle_VQC_chip
    QCHIP_AVAILABLE = True
except ImportError:
    Angle_VQC_chip = None
    QCHIP_AVAILABLE = False

import pyvqnet
from pyvqnet.qnn.vqc.torch import TorchQcloud3QuantumLayer
# from src.VQC_cir_chip import VQC_angle_cir

from src.mamba import SimplifiedMamba 

pyvqnet.backends.set_backend("torch-native")


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

import pyqpanda3.core as pq
 

def VQC_angle_cir(input, param):
    n_wires = len(input)
    print("n_wires:",n_wires)
    exit()
    n_qlayers = int(len(param)/(int(n_wires*3)))
    # print("param:",param)
    # print("n_qlayers:",n_qlayers)
    m_qlist = range(6)
    cbits = range(6)
    measure_qubits = range(6)

    m_prog = pq.QProg()
    circuit = pq.QCircuit()
   
    # 编码层
    for i in range(n_wires):
        circuit << pq.RY(i, input[i])
    
    # 变分层
    for k in range(n_qlayers):
        for i in range(n_wires):
            # circuit << pq.U3(param[int(i + k * n_wires + 0)], param[int(i + k * n_wires + 1)], param[int(i + k * n_wires + 2)], i)
            circuit << pq.U3(param[(k * 3 + 0) * n_wires + i],param[(k * 3 + 1) * n_wires + i],param[(k * 3 + 2) * n_wires + i], i)
        for i in range(n_wires - 1):
            circuit << pq.CNOT(i,i+1)
        circuit << pq.CNOT(n_wires-1,0)
 
    m_prog<<circuit
    for idx, ele in enumerate(measure_qubits):
        m_prog << pq.measure(m_qlist[idx], cbits[idx]) 
        
    return m_prog


def measure_z_expectations(probs, lsb=True):
    """Compute per-qubit <Z_i> from 2**N probability distributions.

    Args:
        probs: tensor of shape (..., 2**N). Last dim is the full
            computational-basis probability (index = basis state integer).
        lsb: bit-ordering convention.
            True  (default) - index bit 0 == qubit 0 (little-endian / LSB).
            False            - index bit 0 == qubit N-1 (big-endian / MSB).

    Returns:
        torch.Tensor of shape (..., N), element i is <Z_i>.
    """
    probs = torch.as_tensor(probs, dtype=torch.float)
    device = probs.device
    size = probs.shape[-1]
    n = size.bit_length() - 1
    if size != (1 << n):
        raise ValueError(f"last dim {size} is not a power of 2")

    states = torch.arange(size, device=device)
    bits = ((states[:, None] >> torch.arange(n, device=device)[None, :]) & 1).float()
    eigenvalues = 1.0 - 2.0 * bits  # (2^N, N): bit=0 -> +1, bit=1 -> -1
    z_exps = probs @ eigenvalues    # (..., N)

    if not lsb:
        z_exps = torch.flip(z_exps, dims=[-1])
    return z_exps



def VQC_angle_cir_mapping(input, param):

    n_wires = len(input)

    n_qlayers = int(len(param)/(int(n_wires*3)))
    m_prog = pq.QProg()
    circuit = pq.QCircuit()
    n_wires_list =  [23, 33, 42, 51, 41, 32]
    cbits =range(n_wires)
    # 编码层
    for i in range(n_wires):
        circuit << pq.RY(n_wires_list[i], input[i])
    
    # 变分层
    for k in range(n_qlayers):
        for i in range(n_wires):
            # circuit << pq.U3(param[int(i + k * n_wires + 0)], param[int(i + k * n_wires + 1)], param[int(i + k * n_wires + 2)], i)
            circuit << pq.U3(param[(k * 3 + 0) * n_wires + i],param[(k * 3 + 1) * n_wires + i],param[(k * 3 + 2) * n_wires + i], n_wires_list[i])
        for i in range(n_wires - 1):
            circuit << pq.CNOT(n_wires_list[i],n_wires_list[i+1])
        circuit << pq.CNOT(n_wires_list[n_wires-1],n_wires_list[0])
     
    
    m_prog<<circuit
    for idx in range(n_wires):
        m_prog << pq.measure(n_wires_list[idx], cbits[idx])  # pylint: disable=expression-not-assigned
    return m_prog


# =============================================================================
# HybridQuantumProjector - 混合量子投影器
# =============================================================================
class HybridQuantumProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bottleneck_dim: int, bias: bool = True, n_bit: int = 6, qlayer: int = 1,use_chip: bool =False):
        super().__init__()
        self.linear_pre = nn.Linear(in_dim, bottleneck_dim, bias=False)
        # self.Angle_VQC = Angle_VQC_chip(n_bit, qlayer)
        self.use_chip = use_chip
        if use_chip:
            print('使用悟空真机芯片...')
            # 236A5FB79168412FBE5498E419718315/80
            # 736aadab27035b2d76eb2384f350ddf0eec2f9d5317347e2fb7248c43d2ccd197554783175625a634345794532534a78
            # 302e020100301006072a8648ce3d020106052b8104001c041730150201010410aa80b675da18a33e79c146c5579cdff6/19653
            # WK_C180_2
            # VQC_angle_cir_mapping  VQC_angle_cir pauli_str_dict=[{'Z0':1},{'Z1':1},{'Z2':1},{'Z3':1},{'Z4':1},{'Z5':1}]
            self.Angle_VQC = TorchQcloud3QuantumLayer(VQC_angle_cir, "736aadab27035b2d76eb2384f350ddf0eec2f9d5317347e2fb7248c43d2ccd197554783175625a634345794532534a78",
        3*6, pauli_str_dict=None,
                shots = 1000,
                initializer=None,
                dtype=None,
                name="",
                diff_method="parameter_shift",
                submit_kwargs={"chip_id":"WK_C180_2","is_amend":False,"is_mapping":True,"is_optimization":True,
                                "compile_level":3,"default_task_group_size":200,
                                "test_qcloud_fake":False},query_kwargs={"total_timeout":36000,"print_query_info":False,
                                "if_print_qcloud_log":False} )   # False True
        else:
            self.Angle_VQC = Angle_VQC_chip(n_bit, qlayer) 
           
        self.linear_post = nn.Linear(bottleneck_dim, out_dim, bias=bias)

    def forward(self, last_raw: torch.Tensor) -> torch.Tensor:
        input_shape = last_raw.shape

        base1 = self.linear_pre(last_raw)
        # print('线性层输出:',base1.shape)
        
        if self.use_chip:
            base_vqc = self.Angle_VQC(last_raw)
            
            base_vqc = measure_z_expectations(base_vqc.data, lsb=True)
        else:
            base_vqc = self.Angle_VQC(last_raw).to(last_raw.device)
           
        base = self.linear_post(base_vqc)
        # print(base.shape)
        # exit()
        return base


# =============================================================================
# FlowQAttention - 量子 Flow 注意力（线性复杂度）
# =============================================================================
class FlowQAttention(nn.Module):
    """Flow Attention - 线性复杂度注意力机制"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        quantum_rank_ratio: float = 0.25,
        qbits: int = 6,
        qlayer: int = 1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        qkv_rank = max(num_heads, int(dim * quantum_rank_ratio))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = HybridQuantumProjector(dim, dim, bottleneck_dim=qbits, bias=True,
                                           n_bit=qbits, qlayer=qlayer)
        self.attn_drop = nn.Dropout(attn_drop)

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """核函数：使用 sigmoid"""
        return torch.sigmoid(x)

    def my_sum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """求和操作: nhld,nhd->nhl"""
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D_head]

        # kernel
        q = self.kernel(q)
        k = self.kernel(k)

        # normalizer
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)

        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)

        # allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        # competition
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])

        # multiply
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]

        x = x_update.transpose(1, 2).reshape(B, N, C)
        x = x.reshape(B*N, C)
        # print('x:',x.shape) # [12288, 128]
        # exit()
        x = self.proj(x) # 量子投影
        x = x.reshape(B, N, C)
        return x

 
# =============================================================================
# WindQMambaformer - 量子 Mambaformer 主模型
# =============================================================================
class WindQMambaformer(nn.Module):
    """
    优化版 Wind-QMambaformer

    结构:
    1. 输入投影层
    2. 位置编码
    3. 多层 Flow Attention + Mamba 双分支堆叠
    4. 门控融合机制
    5. 多尺度时间特征提取
    6. 全局特征聚合
    7. 预测头
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
        qbits: int = 6,
        qlayer: int = 1
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
            FlowQAttention(dim=model_dim, num_heads=flow_attn_heads, attn_drop=dropout,
                           qbits=qbits, qlayer=qlayer)
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

        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(model_dim * 2, int(model_dim/2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(model_dim/2), pred_horizon),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            # 不要重新初始化 Mamba 或 SimplifiedMamba 内部参数
            if name.startswith("mamba_layers"):
                continue

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    if getattr(module.bias, "_no_reinit", False):
                        continue
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
# 工具函数
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_name: str, input_dim: int, seq_len: int = 96, **kwargs) -> nn.Module:
    """
    根据名称获取模型

    Args:
        model_name: 模型名称 ('wind_qmambaformer')
        input_dim: 输入特征维度
        seq_len: 序列长度
        **kwargs: 其他模型参数

    Returns:
        model: PyTorch 模型
    """
    model_name = model_name.lower()

    if model_name == 'wind_qmambaformer':
        return WindQMambaformer(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
