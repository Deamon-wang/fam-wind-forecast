import math
import numpy as np
import torch
import torch.nn as nn
import pyvqnet
from src.VQC_cir import Angle_VQC_cir
import tednet.tnn.tensor_train.base as tt
from pyvqnet.nn.torch import TorchModule,Linear
from pyvqnet.tensor import tensor
pyvqnet.backends.set_backend("torch")


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class QMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), n_bit=6, qlayer=1, dropout=0.1,use_bn=False):
        super().__init__()
        layers = []
        hidden=(512, 128)
        prev = in_dim
        for h in hidden:
            if use_bn:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            else:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h

        self.net = nn.Sequential(*layers)
        
        self.linear_pre = nn.Linear(prev, n_bit)
        self.Angle_VQC_cir1 = Angle_VQC_cir(n_bit, qlayer) 
        self.linear_post = nn.Linear(n_bit, 1)
        
    def forward(self, x):
        layer1_out = self.net(x)
        classic_out = self.linear_pre(layer1_out)
        quantum_out=self.Angle_VQC_cir1(classic_out)
        post_out = self.linear_post(quantum_out.data +classic_out)
        
        return post_out


class QBridge(nn.Module):
    """
    量子桥接层：在传统MLP层之间插入量子变分层，实现维度压缩和特征提取
    优化版：增强残差连接和量子层间跳跃连接
    """
    def __init__(self, in_dim, out_dim, n_bit=6, qlayer=1, dropout=0.1, use_residual=False, use_skip_connection=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_bit = n_bit
        self.qlayer = qlayer
        self.use_residual = use_residual
        self.use_skip_connection = use_skip_connection
        
        # 线性映射到量子空间
        self.linear_pre = nn.Linear(in_dim, n_bit)
        
        # 量子变分层 - 添加多个量子层以增强表达能力
        self.Angle_VQC_cir1 = Angle_VQC_cir(n_bit, qlayer).toGPU()
        self.Angle_VQC_cir2 = Angle_VQC_cir(n_bit, qlayer).toGPU()
        self.Angle_VQC_cir3 = Angle_VQC_cir(n_bit, qlayer).toGPU()
        
        # 从量子空间映射回特征空间
        self.linear_post = nn.Linear(n_bit, out_dim)
        self.drop = nn.Dropout(dropout)
        
        # 残差连接（当输入输出维度相同时）
        if use_residual and in_dim == out_dim:
            self.residual = nn.Identity()
        else:
            # 维度不匹配时使用投影层进行残差连接
            if use_residual:
                self.residual = nn.Linear(in_dim, out_dim)
            else:
                self.residual = None
            
        # 批归一化
        self.bn = nn.BatchNorm1d(out_dim)
        
        # 量子层间的跳跃连接
        self.skip_quantum = use_skip_connection

    def forward(self, x, encoding="ry"):
        # 保存输入用于残差连接
        residual_x = x
        
        # 前向传播通过线性层映射到量子空间
        x_quantum = self.linear_pre(x)
        
        # 量子变分处理 - 带跳跃连接的量子层
        xq1 = self.Angle_VQC_cir1(x_quantum, encoding=encoding)
        if self.skip_quantum:
            # 跳跃连接：将量子层输出与输入量子特征相加
            xq1 = xq1 + x_quantum
        
        xq2 = self.Angle_VQC_cir2(xq1, encoding=encoding)
        if self.skip_quantum:
            xq2 = xq2 + xq1
            
        xq3 = self.Angle_VQC_cir3(xq2, encoding=encoding)
        
        # 从量子空间映射回特征空间
        out = self.linear_post(xq3.data)
        out = self.drop(out)
        out = self.bn(out)
        
        # 残差连接
        if self.residual is not None:
            if self.use_residual:
                residual_input = self.residual(residual_x)
                out = out + residual_input
            else:
                out = out + residual_x
        
        return out

    def get_parameter_count(self):
        """获取QBridge的参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

class QMLPRegressor(nn.Module):
    """
    改进的QMLPRegressor，在每两层隐藏层之间插入QBridge量子桥接层
    实现特征维度压缩和非线性变换，降低参数量同时提升表达能力
    """
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.1, n_bit=6, qlayer=1, insert_qbridge=True, use_residual=False):
        super().__init__()
        self.n_bit = n_bit
        self.qlayer = qlayer
        self.insert_qbridge = insert_qbridge
        self.use_residual = use_residual
        
        self.mlp_layers = nn.ModuleList()
        prev_dim = in_dim
      
        for i in range(len(hidden)):
            # 添加传统MLP层
            self.mlp_layers.append(nn.Linear(prev_dim, hidden[i]))
            self.mlp_layers.append(nn.ReLU(inplace=True))
            self.mlp_layers.append(nn.BatchNorm1d(hidden[i]))
            self.mlp_layers.append(nn.Dropout(dropout))
            
            if i == len(hidden) - 2:
                # 使用残差连接当维度相同时
                residual = use_residual and hidden[i] == hidden[i+1]
                qbridge = QBridge(
                    in_dim=hidden[i], 
                    out_dim=hidden[i+1], 
                    n_bit=n_bit, 
                    qlayer=qlayer, 
                    dropout=dropout,
                    use_residual=residual,
                    use_skip_connection=True)
                self.mlp_layers.append(qbridge)
                prev_dim = hidden[i+1]
            else:
                prev_dim = hidden[i]
    

        # 输出层
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 记录QBridge层数量
        self.qbridge_count = sum(1 for layer in self.mlp_layers if isinstance(layer, QBridge))

    def forward(self, x, encoding="ry"):
        for layer in self.mlp_layers:
            if isinstance(layer, QBridge):
                # QBridge支持encoding参数
                x = layer(x, encoding=encoding)
            else:
                # 其他层正常处理
                x = layer(x)
        
        # 输出层
        out = self.output_layer(x)
        return out

    def get_qbridge_parameter_count(self):
        """获取所有QBridge层的参数数量"""
        qbridge_params = 0
        qbridge_trainable = 0
        for layer in self.mlp_layers:
            if isinstance(layer, QBridge):
                total, trainable = layer.get_parameter_count()
                qbridge_params += total
                qbridge_trainable += trainable
        return qbridge_params, qbridge_trainable

    def get_layer_info(self):
        """获取模型层级信息"""
        layer_info = []
        for i, layer in enumerate(self.mlp_layers):
            if isinstance(layer, QBridge):
                layer_info.append(f"Layer {i}: QBridge({layer.in_dim}→{layer.out_dim}, n_bit={layer.n_bit})")
            else:
                layer_info.append(f"Layer {i}: {type(layer).__name__}")
        return layer_info


factors_dict = {"1320":[12,11,10], "1056":[12,11,8], "1024":[16,8,8], "792":[11,9,8], "768":[12,8,8], 
                "384":[8,8,6], "264":[11,6,4], "192":[8,6,4], "576":[12,8,6], "512":[8,8,8], "528":[11,8,6], "288":[8,6,6], "256":[8,8,4], "132":[11,6,2],
                "128":[8,8,2], "144":[6,6,4], "72":[6,4,3], "66":[11,3,2], "64":[4,4,4], "96":[6,4,4], "88":[11,4,2],  "32":[4,4,2], "48":[6,4,2],
                "24":[4,3,2], "16":[4,2,2],"12":[3,2,2], "8":[2,2,2]}

# 需要进行优化处理，此方案待定 2026 0224
from src.VQC_cir_chip import Angle_VQC_chip
class QMLPTT(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.1, n_bit=6, qlayer=1, rank=8,use_bn=False):
        super().__init__()
        layers = []
        hidden=(512, 128)
        prev = in_dim
        ttlinear = tt.TTLinear(in_shape=in_shape, out_shape=out_shape, ranks=[rank, rank])
        for h in hidden:
            if use_bn:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            else:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h

        self.net = nn.Sequential(*layers) 
        
        self.linear_pre = nn.Linear(prev, n_bit)
        self.Angle_VQC_cir1 = Angle_VQC_chip(n_bit, qlayer) 
        self.linear_post = nn.Linear(n_bit, 1)
        
    def forward(self, x):
        layer1_out = self.net(x)
        classic_out = self.linear_pre(layer1_out)
        quantum_out=self.Angle_VQC_cir1(classic_out)
        post_out = self.linear_post(quantum_out.data +classic_out)
        
        return post_out

# class QMLPTT(nn.Module):
#     def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.1, n_bit=6, qlayer=1, rank=8):
#         super().__init__()
#         self.mlp_layers = nn.ModuleList()
#         prev_dim = in_dim

#         for curr_dim in hidden:
#             in_shape = factors_dict.get(str(prev_dim))
#             out_shape = factors_dict.get(str(curr_dim))
            
#             if in_shape is not None and out_shape is not None:
#                 self.mlp_layers.append(tt.TTLinear(in_shape=in_shape, out_shape=out_shape, ranks=[rank, rank]))
#             else:
#                 self.mlp_layers.append(nn.Linear(prev_dim, curr_dim))

#             self.mlp_layers.append(nn.ReLU(inplace=True))
#             self.mlp_layers.append(nn.BatchNorm1d(curr_dim))
#             self.mlp_layers.append(nn.Dropout(dropout))
#             prev_dim = curr_dim

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.n_bit, self.qlayer = n_bit, qlayer
#         self.linear_pre = nn.Linear(prev_dim, self.n_bit)
#         self.Angle_VQC_cir1 = Angle_VQC_chip(self.n_bit, self.qlayer).toGPU()
#         # self.Angle_VQC_cir2 = Angle_VQC_cir(self.n_bit, self.qlayer).toGPU()
#         self.linear = nn.Linear(self.n_bit, 1)

#     def forward(self, x, encoding="ry"):
#         for layer in self.mlp_layers:
#             x = layer(x)
#         x_residual = self.linear_pre(x)
#         x_q = self.Angle_VQC_cir1(x_residual, encoding=encoding)
#         # x_q = self.Angle_VQC_cir2(x_q, encoding=encoding)
#         out = self.linear(x_q.data)
#         return out


class MLPRegressorMulti(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QMLPRegressorMulti(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 128, 64), dropout=0.1, n_bit=6, qlayer=1):
        super().__init__()
        self.mlp_layers = nn.ModuleList()
        prev_dim = in_dim
        for curr_dim in hidden:
            self.mlp_layers.append(nn.Linear(prev_dim, curr_dim))
            self.mlp_layers.append(nn.ReLU(inplace=True))
            self.mlp_layers.append(nn.BatchNorm1d(curr_dim))
            self.mlp_layers.append(nn.Dropout(dropout))
            prev_dim = curr_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.n_bit, self.qlayer = n_bit, qlayer
        self.linear_pre = nn.Linear(prev_dim, self.n_bit)
        self.Angle_VQC_cir1 = Angle_VQC_cir(self.n_bit, self.qlayer).toGPU()
        self.Angle_VQC_cir2 = Angle_VQC_cir(self.n_bit, self.qlayer).toGPU()
        self.linear = nn.Linear(self.n_bit, out_dim)

    def forward(self, x, encoding="ry"):
        for layer in self.mlp_layers:
            x = layer(x)
        x_residual = self.linear_pre(x)
        x_q = self.Angle_VQC_cir1(x_residual, encoding=encoding)
        x_q = self.Angle_VQC_cir2(x_q, encoding=encoding)
        out = self.linear(x_q.data)
        return out


class QMLPTTMulti(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 128, 64), dropout=0.1, n_bit=6, qlayer=1, rank=8):
        super().__init__()
        self.mlp_layers = nn.ModuleList()
        prev_dim = in_dim
        for curr_dim in hidden:
            in_shape = factors_dict.get(str(prev_dim))
            out_shape = factors_dict.get(str(curr_dim))
            if in_shape is not None and out_shape is not None:
                self.mlp_layers.append(tt.TTLinear(in_shape=in_shape, out_shape=out_shape, ranks=[rank, rank]))
            else:
                self.mlp_layers.append(nn.Linear(prev_dim, curr_dim))

            self.mlp_layers.append(nn.ReLU(inplace=True))
            self.mlp_layers.append(nn.BatchNorm1d(curr_dim))
            self.mlp_layers.append(nn.Dropout(dropout))
            prev_dim = curr_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.n_bit, self.qlayer = n_bit, qlayer
        self.linear_pre = nn.Linear(prev_dim, self.n_bit)
        self.Angle_VQC_cir1 = Angle_VQC_cir(self.n_bit, self.qlayer).toGPU()
        self.Angle_VQC_cir2 = Angle_VQC_cir(self.n_bit, self.qlayer).toGPU()
        self.linear = nn.Linear(self.n_bit, out_dim)
        
        # Shortcut: Linear projection from input to output to learn simple identity/trend
        self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x, encoding="ry"):
        identity = self.shortcut(x)
        
        out_mlp = x
        for layer in self.mlp_layers:
            out_mlp = layer(out_mlp)
        x_residual = self.linear_pre(out_mlp)
        x_q = self.Angle_VQC_cir1(x_residual, encoding=encoding)
        x_q = self.Angle_VQC_cir2(x_q, encoding=encoding)
        out = self.linear(x_q.data)
        
        return out + identity

def make_mlp(in_dim, hidden=(128, 64), dropout=0.1, use_bn=True):
    layers = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers), prev

class MLPMulti(nn.Module):
    def __init__(self,in_dim,out_dim, scale_indices,
        branch_hidden=(128, 64),
        fusion_hidden=(128, 64),
        dropout=0.1,
        fusion="gating",              # "concat" or "gating"
        use_bn=True,
        shortcut=True
    ):
        super().__init__()
        self.scale_names = ["short", "mid", "long", "other"]
        self.scale_indices = {}
        for k in self.scale_names:
            ids = scale_indices.get(k, [])
            self.scale_indices[k] = torch.tensor(ids, dtype=torch.long)  # 注册为tensor，forward时搬到device

        # 各尺度分支
        self.branches = nn.ModuleDict()
        self.branch_out_dim = {}
        for k in self.scale_names:
            dim_k = len(scale_indices.get(k, []))
            if dim_k == 0:
                # 没有该尺度特征时，给一个空分支占位
                self.branches[k] = nn.Identity()
                self.branch_out_dim[k] = 0
            else:
                mlp, outd = make_mlp(dim_k, hidden=branch_hidden, dropout=dropout, use_bn=use_bn)
                self.branches[k] = mlp
                self.branch_out_dim[k] = outd

        # 融合
        self.fusion = fusion
        if fusion == "concat":
            fused_in = sum(self.branch_out_dim.values())
            self.fusion_mlp, fused_dim = make_mlp(fused_in, hidden=fusion_hidden, dropout=dropout, use_bn=use_bn)
        elif fusion == "gating":
            # # 每个尺度分支都输出同一个维度，便于加权求和
            # common_dim = fusion_hidden[0]  # 也可以单独设定
            # self.proj = nn.ModuleDict()
            # for k in self.scale_names:
            #     outd = self.branch_out_dim[k]
            #     if outd == 0:
            #         self.proj[k] = nn.Identity()
            #     else:
            #         self.proj[k] = nn.Linear(outd, common_dim)
            # # gate 根据拼接后的分支表征生成权重
            # gate_in = sum(self.branch_out_dim.values())
            # self.gate = nn.Sequential(
            #     nn.Linear(gate_in, len(self.scale_names)),
            #     nn.Softmax(dim=-1)
            # )
            # # 融合后再过一个 MLP
            # self.post_fusion, fused_dim = make_mlp(common_dim, hidden=fusion_hidden, dropout=dropout, use_bn=use_bn)
            
            
            common_dim = fusion_hidden[0]
            self.proj = nn.ModuleDict()
            self.non_empty_scales = [k for k in self.scale_names if self.branch_out_dim[k] > 0]

            for k in self.non_empty_scales:
                outd = self.branch_out_dim[k]
                self.proj[k] = nn.Linear(outd, common_dim)

            # gate 根据非空分支表征生成权重
            gate_in = sum([self.branch_out_dim[k] for k in self.non_empty_scales])
            self.gate = nn.Sequential(
                nn.Linear(gate_in, len(self.non_empty_scales)),
                nn.Softmax(dim=-1)
            )
            self.post_fusion, fused_dim = make_mlp(common_dim, hidden=fusion_hidden, dropout=dropout, use_bn=use_bn)
        
        else:
            raise ValueError("fusion must be 'concat' or 'gating'")

        # 多 horizon 输出头（最简单：一个线性层输出 out_dim）
        self.head = nn.Linear(fused_dim, out_dim)

        # 可选：捷径（线性残差），帮助学习“简单趋势/线性项”
        self.use_shortcut = shortcut
        if shortcut:
            self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
 
        # x: (B, in_dim)
        reps = {}
        rep_list_for_gate = []

        for k in self.scale_names:
            ids = self.scale_indices[k].to(x.device)
            if ids.numel() == 0:
                reps[k] = None
                continue
            xk = x.index_select(dim=1, index=ids)
            hk = self.branches[k](xk)
            reps[k] = hk
            rep_list_for_gate.append(hk)

        if self.fusion == "concat":
            z = torch.cat([reps[k] for k in self.scale_names if reps[k] is not None], dim=-1)
            z = self.fusion_mlp(z)

        else:  # gating
            # # gate权重来自所有分支拼接
            # z_cat = torch.cat([reps[k] for k in self.scale_names if reps[k] is not None], dim=-1)
            # w = self.gate(z_cat)  # (B, 4)

            # # 各分支投影到同维度并加权求和
            # z_sum = 0.0
            # for i, k in enumerate(self.scale_names):
            #     if reps[k] is None:
            #         continue
            #     zk = self.proj[k](reps[k])
            #     z_sum = z_sum + w[:, i:i+1] * zk

            # z = self.post_fusion(z_sum)

            z_cat = torch.cat(rep_list_for_gate, dim=-1)
            w = self.gate(z_cat)  # (B, num_non_empty_scales)
            z_sum = torch.zeros((x.size(0), self.post_fusion[0].in_features), device=x.device)
            for i, k in enumerate(self.non_empty_scales):
                zk = self.proj[k](reps[k])
                z_sum = z_sum + w[:, i:i+1] * zk

            z = self.post_fusion(z_sum)

        out = self.head(z)

        if self.use_shortcut:
            out = out + self.shortcut(x)
        return out

class QuantumBridge(nn.Module):
    def __init__(self, in_dim, out_dim, n_bit=6, qlayer=1, dropout=0.1):
        super().__init__()
        self.linear_pre = nn.Linear(in_dim, n_bit)
      
        self.Angle_VQC_cir1 = Angle_VQC_chip(n_bit, qlayer).toGPU()
      
        self.linear_post = nn.Linear(n_bit, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, encoding="ry"):
        x = self.linear_pre(x)
        xq = self.Angle_VQC_cir1(x, encoding=encoding)
        # xq = self.Angle_VQC_cir2(xq, encoding=encoding)
        y = self.linear_post(xq.data)
        # y = self.drop(y)
        return y

class QuantumBridge_phn(nn.Module):
    def __init__(self, in_dim, out_dim, n_bit=6, qlayer=1, dropout=0.1):
        super().__init__()
        self.linear_pre = nn.Linear(in_dim, n_bit)
      
        self.Angle_VQC_cir1 = Angle_VQC_chip(n_bit, qlayer).toGPU()
      
        self.linear_post = nn.Linear(n_bit, out_dim)

        self.weight_classic = nn.Linear(n_bit, n_bit)
        self.weight_quantum = nn.Linear(n_bit, n_bit)

    
        self.drop = nn.Dropout(dropout)

    def forward(self, x, encoding="ry"):
        x = self.linear_pre(x)
        # print('x:',x.shape)
        # exit()
        xq = self.Angle_VQC_cir1(x, encoding=encoding)
        # xq = self.Angle_VQC_cir2(xq, encoding=encoding)
        y = self.linear_post(self.weight_classic(x) + self.weight_quantum(xq.data))
     
        return y

def make_qmlp(in_dim, hidden=(128, 64), dropout=0.1, use_bn=True, n_bit=6, qlayer=1, insert_quantum=True):
    layers = []
    prev = in_dim
    i = 0
    while i < len(hidden):
        h = hidden[i]
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.Dropout(dropout))
        prev = h
        if insert_quantum and i == 0 and (i + 1) < len(hidden):
            next_h = hidden[i + 1]
            # layers.append(QuantumBridge(in_dim=h, out_dim=next_h, n_bit=n_bit, qlayer=qlayer, dropout=dropout))

            layers.append(QuantumBridge_phn(in_dim=h, out_dim=next_h, n_bit=n_bit, qlayer=qlayer, dropout=dropout))
            
            # layers.append(nn.ReLU(inplace=True))
            # if use_bn:
            #     layers.append(nn.BatchNorm1d(next_h))
            # layers.append(nn.Dropout(dropout))
            prev = next_h
            i += 2
        else:
            i += 1
    return nn.Sequential(*layers), prev
   
class QMLPMulti(nn.Module):
    def __init__(self,in_dim,out_dim, scale_indices,
        branch_hidden=(128, 64),
        fusion_hidden=(128, 64),
        fusion="gating",              # "concat" or "gating"
        dropout=0.1,
        shortcut=True,
        qlayer=1,
        qbit=6,
        use_bn=True,):
    
        super().__init__()
        self.scale_names = ["short", "mid", "long", "other"]
        self.scale_indices = {}
        for k in self.scale_names:
            ids = scale_indices.get(k, [])
            self.scale_indices[k] = torch.tensor(ids, dtype=torch.long)  # 注册为tensor，forward时搬到device

        # 各尺度分支
        self.branches = nn.ModuleDict()
        self.branch_out_dim = {}
        for k in self.scale_names:
            dim_k = len(scale_indices.get(k, []))
            if dim_k == 0:
                # 没有该尺度特征时，给一个空分支占位
                self.branches[k] = nn.Identity()
                self.branch_out_dim[k] = 0
            else:
                mlp, outd = make_mlp(dim_k, hidden=branch_hidden, dropout=dropout, use_bn=use_bn)
                self.branches[k] = mlp
                self.branch_out_dim[k] = outd

        # 融合
        self.fusion = fusion
        if fusion == "concat":
            fused_in = sum(self.branch_out_dim.values())
            # self.fusion_mlp, fused_dim = make_mlp(fused_in, hidden=fusion_hidden, dropout=dropout, use_bn=use_bn)
            self.fusion_mlp, fused_dim =  make_qmlp(fused_in, hidden=fusion_hidden, dropout=dropout, use_bn=use_bn,qlayer=qlayer,n_bit=qbit)
        elif fusion == "gating":
            # # 每个尺度分支都输出同一个维度，便于加权求和
            # common_dim = fusion_hidden[0]  # 也可以单独设定
            # self.proj = nn.ModuleDict()
            # for k in self.scale_names:
            #     outd = self.branch_out_dim[k]
            #     if outd == 0:
            #         self.proj[k] = nn.Identity()
            #     else:
            #         self.proj[k] = nn.Linear(outd, common_dim)
            # # gate 根据拼接后的分支表征生成权重
            # gate_in = sum(self.branch_out_dim.values())
            # self.gate = nn.Sequential(
            #     nn.Linear(gate_in, len(self.scale_names)),
            #     nn.Softmax(dim=-1)
            # )

            common_dim = fusion_hidden[0]
            self.proj = nn.ModuleDict()
            self.non_empty_scales = [k for k in self.scale_names if self.branch_out_dim[k] > 0]

            for k in self.non_empty_scales:
                outd = self.branch_out_dim[k]
                self.proj[k] = nn.Linear(outd, common_dim)

            # gate 根据非空分支表征生成权重
            gate_in = sum([self.branch_out_dim[k] for k in self.non_empty_scales])
            self.gate = nn.Sequential(
                nn.Linear(gate_in, len(self.non_empty_scales)),
                nn.Softmax(dim=-1)
            )
           
            # 融合后再过一个 QMLP
            self.post_fusion, fused_dim = make_qmlp(common_dim, hidden=fusion_hidden, dropout=dropout, use_bn=use_bn,qlayer=qlayer,n_bit=qbit)
        else:
            raise ValueError("fusion must be 'concat' or 'gating'")

        # 多 horizon 输出头（最简单：一个线性层输出 out_dim）
        self.head = nn.Linear(fused_dim, out_dim)

        # 可选：捷径（线性残差），帮助学习“简单趋势/线性项”
        self.use_shortcut = shortcut
        if shortcut:
            self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, in_dim)
        reps = {}
        rep_list_for_gate = []

        for k in self.scale_names:
            ids = self.scale_indices[k].to(x.device)
            if ids.numel() == 0:
                reps[k] = None
                continue
            xk = x.index_select(dim=1, index=ids)
            hk = self.branches[k](xk)
            reps[k] = hk
            rep_list_for_gate.append(hk)

        if self.fusion == "concat":
            z = torch.cat([reps[k] for k in self.scale_names if reps[k] is not None], dim=-1)
            z = self.fusion_mlp(z)

        else:  # gating
            # gate权重来自所有分支拼接
            # z_cat = torch.cat([reps[k] for k in self.scale_names if reps[k] is not None], dim=-1)
            # w = self.gate(z_cat)  # (B, 4)
            # # 各分支投影到同维度并加权求和
            # z_sum = 0.0
            # for i, k in enumerate(self.scale_names):
            #     if reps[k] is None:
            #         continue
            #     zk = self.proj[k](reps[k])
            #     z_sum = z_sum + w[:, i:i+1] * zk
            # z = self.post_fusion(z_sum)

            z_cat = torch.cat(rep_list_for_gate, dim=-1)
            w = self.gate(z_cat)  # (B, num_non_empty_scales)

            z_sum = torch.zeros((x.size(0), self.post_fusion[0].in_features), device=x.device)
            for i, k in enumerate(self.non_empty_scales):
                zk = self.proj[k](reps[k])
                z_sum = z_sum + w[:, i:i+1] * zk
            z = self.post_fusion(z_sum)

        out = self.head(z)

        if self.use_shortcut:
            out = out + self.shortcut(x)
        return out


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        # print(x.shape)
        # exit()
        return x + self.pe[:, :x.shape[1]]

# Flow Attention
class FlowAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5  # Standard attention scaling
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
        
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, d_head)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)

# Mamba Block
 
# class MambaBlock(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.conv = nn.Conv1d(d_model,d_model,kernel_size=3,padding=1)
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.act = nn.SiLU()

#         n_bit, qlayer = 6,1
#         # 量子设置
#         self.W1 = tensor.randn([d_model, n_bit], requires_grad = True, device=pyvqnet.DEV_GPU) 
#         self.Angle_VQC_cir1 = Angle_VQC_chip(n_bit, qlayer).toGPU()
#         self.linear_post = Linear(n_bit, d_model)
#         self.in_proj2 = nn.Linear(d_model, d_model)

#     def forward(self, x):
#         # print(x.shape)  # [128, 72, 128]
#         u =  self.in_proj2(x) 
#         q_out=self.Angle_VQC_cir1(self.W1) # 
     
#         q_out = tensor.matmul(x, self.W1)
#         v = self.linear_post(q_out)
#         v = self.conv(v.transpose(1, 2)).transpose(1, 2)
#         # print(v.shape)  # [128,72,128]
#         # exit()
#         return self.out_proj(self.act(u * v))

# class MambaBlock(nn.Module):
#     def __init__(self, d_model, dropout=0.1):
#         super().__init__()
#         self.in_proj = nn.Linear(d_model, d_model * 2)
#         # self.conv = nn.Conv1d(d_model,d_model,kernel_size=3,padding=1)

#         # v 的处理：线性变换 + GELU + dropout
#         self.v_proj = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
       
#         self.out_proj = nn.Linear(d_model, d_model)
       

#     def forward(self, x):
#         u, v = self.in_proj(x).chunk(2, dim=-1)  # (B, T, D) each
#         v = self.v_proj(v)
#         res = self.out_proj(u * v)
#         return res
class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1, kernel_size=3):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)

        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model)

        self.v_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout) )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        u, v = self.in_proj(x).chunk(2, dim=-1)   # [B, T, D], [B, T, D]
        # causal depthwise conv on time dimension
        v_conv = self.dw_conv(v.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        v = self.v_proj(v_conv)

        out = self.out_proj(u * torch.sigmoid(v))
        out = self.dropout(out)
        return out


# Transformer + Mamba Block
class TransformerMambaBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = FlowAttention(d_model, n_heads, dropout)
        self.mamba = MambaBlock(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network after mamba
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + residual
        
        # Mamba block with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward network with residual connection
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + residual
        
        return x


################# 3-19 ######################################
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
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

import torch.nn.functional as F



class GRUBlock(nn.Module):
    def __init__(self, d_model, hidden_size=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_size = hidden_size or d_model

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.out_proj = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.gru(x)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out

class BetterTransformerMambaGRUBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, mamba_kernel_size=3, gru_hidden=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FlowAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mamba = BetterMambaBlock(d_model, dropout=dropout, kernel_size=mamba_kernel_size)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.gru = GRUBlock(d_model, hidden_size=gru_hidden, num_layers=1, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

        self.norm4 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.mamba(self.norm2(x)))
        x = x + self.drop3(self.gru(self.norm3(x)))
        x = x + self.ffn(self.norm4(x))
        return x

class FlowAttention(nn.Module):
    """
    更稳一点的 Flow-like attention：
    - head 内做 RMSNorm 风格约束
    - 正域映射
    - 双边归一
    - 适合 encoder 历史窗口建模
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
        k_sum = k.sum(dim=2)   # [B,H,Dh]
        q_sum = q.sum(dim=2)   # [B,H,Dh]

        row_norm = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k_sum) + self.eps)
        col_norm = 1.0 / (torch.einsum("bhtd,bhd->bht", k, q_sum) + self.eps)

        weighted_k = k * col_norm.unsqueeze(-1)
        kv = torch.einsum("bhtd,bhte->bhde", weighted_k, v)   # [B,H,Dh,Dh]
        out = torch.einsum("bhtd,bhde->bhte", q, kv)          # [B,H,T,Dh]
        out = out * row_norm.unsqueeze(-1)

        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


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
        return out # self.dropout(out)


class BetterTransformerMambaBlock(nn.Module):
    """
    结构：
    Pre-LN Attention -> Residual
    Pre-LN Mamba     -> Residual
    Pre-LN FFN       -> Residual
    """
    def __init__(self, d_model, n_heads, dropout=0.1, mamba_kernel_size=3):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FlowAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mamba = BetterMambaBlock(d_model, kernel_size=mamba_kernel_size,dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.mamba(self.norm2(x)))
        x = x + self.ffn(self.norm3(x))
        return x


class WindPowerTransformer(nn.Module):
    """
    改进版：
    - 更强的局部时序建模
    - 更适合回归预测的 summary head
    - residual forecasting: base + delta
    输入:  [B, T, input_dim]
    输出:  [B, pred_len]
    """
    def __init__(self,input_dim,d_model=64,n_heads=4,num_layers=2,pred_len=1,dropout=0.1,power_idx=None,
        recent_window=6,use_persistence=True,local_kernel_sizes=(3, 5),mamba_kernel_size=3,):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.power_idx = power_idx
        self.recent_window = recent_window
        self.use_persistence = use_persistence

        # 输入嵌入
        self.embedding = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # 局部时序卷积 stem
        k1, k2 = local_kernel_sizes
        self.local_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k1, padding=k1 // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=k2, padding=k2 // 2),
            nn.GELU(),
            # nn.Dropout(dropout),
            )

        # 主干块
        self.blocks = nn.ModuleList([
            BetterTransformerMambaBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                mamba_kernel_size=mamba_kernel_size
            )
            for _ in range(num_layers)])

        self.norm = nn.LayerNorm(d_model)

        # 更聚焦预测任务的 head：last_token + recent_pool + last_raw
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

        # shortcut 负责原始输入的直接线性映射
        self.shortcut = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        # x: [B, T, input_dim]
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B, T, D), got {tuple(x.shape)}")

        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        x_raw = x

        # embedding
        x = self.embedding(x)          # [B, T, d_model]
        x = self.input_dropout(x)
        x = self.pos_enc(x)

        # local temporal stem
        x = x.transpose(1, 2)          # [B, d_model, T]
        x = self.local_stem(x)
        x = x.transpose(1, 2)          # [B, T, d_model]

        # backbone
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # temporal summary
        last_token = x[:, -1, :]                           # [B, d_model]
        rw = min(self.recent_window, T)
        recent_pool = x[:, -rw:, :].mean(dim=1)           # [B, d_model]
        last_raw = x_raw[:, -1, :]                        # [B, input_dim]

        feat = torch.cat([last_token, recent_pool, last_raw], dim=-1)
        delta = self.head(feat)                           # [B, pred_len]

        # 强基线：最后一个时刻原始输入线性映射
        base = self.shortcut(last_raw)                    # [B, pred_len]

        # persistence residual
        if self.use_persistence and (self.power_idx is not None):
            last_power = x_raw[:, -1, self.power_idx].unsqueeze(-1)  # [B, 1]
            if self.pred_len > 1:
                last_power = last_power.expand(-1, self.pred_len)
            base = base + last_power

        out = base + delta
        return out

#########################################################################
# 3.20

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
            # nn.GELU(),
            # nn.Dropout(dropout),
        )

        self.mamba_blocks = nn.ModuleList([
            BetterMambaBlock(d_model, dropout=dropout, kernel_size=mamba_kernel_size)
            for _ in range(num_layers)
        ])
        self.mamba_norm = nn.LayerNorm(d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.flow_norm = nn.LayerNorm(d_model)
        self.flow_attn = FlowAttention(d_model, n_heads, dropout)

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
        x = gru_out

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
# 3.20

class GRUExpert(nn.Module):
    def __init__(self, d_model, hidden_size=None, num_layers=1, dropout=0.1):
        super().__init__()
        hidden_size = hidden_size or d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.out_proj(out)
        return self.norm(out)


class MambaExpert(nn.Module):
    def __init__(self, d_model, block_cls, num_blocks=1, dropout=0.1, kernel_size=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            block_cls(d_model=d_model, dropout=dropout, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)
        return self.norm(x)


class FlowExpert(nn.Module):
    def __init__(self, d_model, n_heads, flow_attn_cls, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = flow_attn_cls(d_model, n_heads, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.drop(self.attn(x)))


class FusionRouter(nn.Module):
    """
    每个未来步单独输出三个分支权重
    """
    def __init__(self, d_model, cond_dim, pred_len, num_experts=3):
        super().__init__()
        self.pred_len = pred_len
        self.horizon_emb = nn.Embedding(pred_len, d_model)
        self.router = nn.Sequential(
            nn.Linear(d_model + cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_experts)
        )

    def forward(self, cond_feat):
        # cond_feat: [B, cond_dim]
        B = cond_feat.size(0)
        steps = torch.arange(self.pred_len, device=cond_feat.device)
        h_emb = self.horizon_emb(steps).unsqueeze(0).expand(B, -1, -1)   # [B,P,D]
        cond = cond_feat.unsqueeze(1).expand(-1, self.pred_len, -1)      # [B,P,C]
        logits = self.router(torch.cat([h_emb, cond], dim=-1))           # [B,P,3]
        return torch.softmax(logits, dim=-1)


class HorizonDecoder(nn.Module):
    def __init__(self, d_model, input_dim, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.horizon_emb = nn.Embedding(pred_len, d_model)

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * 2 + input_dim),
            nn.Linear(d_model * 2 + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, fused_step_feat, recent_pool, last_raw):
        # fused_step_feat: [B,P,D]
        B, P, D = fused_step_feat.shape
        steps = torch.arange(P, device=fused_step_feat.device)
        h_emb = self.horizon_emb(steps).unsqueeze(0).expand(B, -1, -1)
        last_raw = last_raw.unsqueeze(1).expand(-1, P, -1)
        feat = torch.cat([fused_step_feat, h_emb, last_raw], dim=-1)
        return self.proj(feat).squeeze(-1)


class WindPowerRoleFusionNet(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        pred_len=6,
        n_heads=4,
        power_idx=None,
        local_window=48,
        recent_window=12,
        use_persistence=True,
        dropout=0.1,
        mamba_block_cls=None,
        flow_attn_cls=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len
        self.power_idx = power_idx
        self.local_window = local_window
        self.recent_window = recent_window
        self.use_persistence = use_persistence

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.mamba_expert = MambaExpert(
            d_model=d_model,
            block_cls=mamba_block_cls,
            num_blocks=1,
            dropout=dropout,
            kernel_size=5
        )

        self.gru_expert = GRUExpert(
            d_model=d_model,
            hidden_size=256,
            num_layers=2,
            dropout=dropout
        )

        self.flow_expert = FlowExpert(
            d_model=d_model,
            n_heads=n_heads,
            flow_attn_cls=flow_attn_cls,
            dropout=dropout
        )

        # 根据工况统计信息 + horizon 决定专家权重
        cond_dim = 6
        self.router = FusionRouter(
            d_model=d_model,
            cond_dim=cond_dim,
            pred_len=pred_len,
            num_experts=3
        )

        self.decoder = HorizonDecoder(
            d_model=d_model,
            input_dim=input_dim,
            pred_len=pred_len,
            dropout=dropout
        )

        self.base_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

        self.horizon_scale = nn.Parameter(torch.ones(pred_len))
        self.horizon_bias = nn.Parameter(torch.zeros(pred_len))

    def build_condition_feat(self, x_raw):
        # x_raw: [B,T,F]
        power = x_raw[..., self.power_idx] if self.power_idx is not None else x_raw[..., 0]
        recent = power[:, -self.recent_window:]

        mean_ = recent.mean(dim=1, keepdim=True)
        std_ = recent.std(dim=1, keepdim=True, unbiased=False)
        max_ = recent.max(dim=1, keepdim=True).values
        min_ = recent.min(dim=1, keepdim=True).values
        last_ = recent[:, -1:].contiguous()
        ramp_ = (recent[:, 1:] - recent[:, :-1]).abs().mean(dim=1, keepdim=True)

        return torch.cat([mean_, std_, max_, min_, last_, ramp_], dim=-1)   # [B,6]

    def forward(self, x):
        # x: [B,T,F]
        x_raw = x
        z = self.embed(x)   # [B,T,D]

        # 1) Mamba 分支：只看最近局部窗口
        x_local = z[:, -min(self.local_window, z.size(1)):, :]
        mamba_feat = self.mamba_expert(x_local)
        mamba_summary = mamba_feat[:, -1:, :].expand(-1, self.pred_len, -1)  # [B,P,D]

        # 2) GRU 分支：看完整历史
        gru_feat = self.gru_expert(z)
        gru_summary = gru_feat[:, -1:, :].expand(-1, self.pred_len, -1)

        # 3) Flow 分支：建议实际实现时喂“锚点 token”，这里先简化成全序列
        flow_feat = self.flow_expert(z)
        flow_summary = flow_feat[:, -1:, :].expand(-1, self.pred_len, -1)

        # 4) 路由融合
        cond_feat = self.build_condition_feat(x_raw)          # [B,6]
        weights = self.router(cond_feat)                      # [B,P,3]

        fused = (
            weights[..., 0:1] * mamba_summary +
            weights[..., 1:2] * gru_summary +
            weights[..., 2:3] * flow_summary
        )                                                     # [B,P,D]

        rw = min(self.recent_window, z.size(1))
        recent_pool = z[:, -rw:, :].mean(dim=1)               # [B,D]
        last_raw = x_raw[:, -1, :]                            # [B,F]

        delta = self.decoder(fused, recent_pool, last_raw)    # [B,P]
        base = self.base_head(last_raw)

        if self.use_persistence and (self.power_idx is not None):
            last_power = x_raw[:, -1, self.power_idx].unsqueeze(-1)
            persistence = last_power * self.horizon_scale.unsqueeze(0) + self.horizon_bias.unsqueeze(0)
            base = base + persistence

        return base + delta





##################################################################
class QMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # self.in_proj = nn.Linear(d_model, d_model * 2)

        # self.in_proj2 = nn.Linear(d_model, d_model)
        n_bit, qlayer = 6,2
        self.W1 = tensor.randn([d_model, n_bit], requires_grad = True, device=pyvqnet.DEV_GPU) 
        self.Angle_VQC_cir1 = Angle_VQC_chip(n_bit, qlayer).toGPU()
        # self.Angle_VQC_cir2 = Angle_VQC_chip(n_bit, qlayer).toGPU()
        self.linear_post =  nn.Linear(n_bit, d_model)

        # v 的处理：线性变换 + GELU + dropout
        self.v_proj = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(d_model, d_model)
       

    def forward(self, x):
        # u =  self.in_proj2(x)
        q_out=self.Angle_VQC_cir1(self.W1) 
        
        q_out = tensor.matmul(x,q_out)
        u = self.linear_post(q_out.data)
        v = self.v_proj(u)

        # # u, v = self.in_proj(x).chunk(2, dim=-1)  # (B, T, D) each
        # v = self.v_proj(v)
        # res = self.out_proj(u * v)
        res = self.out_proj(u *torch.sigmoid(v))
        return res



# Transformer +    QMamba Block
class QTransformerMambaBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = FlowAttention(d_model, n_heads, dropout)
        self.quan_mamba = QMambaBlock(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network after mamba
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + residual
        
        # Mamba block with residual connection
        residual = x
        x = self.norm2(x)
        x = self.quan_mamba(x)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward network with residual connection
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + residual
        
        return x


class QWindPowerTransformer(nn.Module):
    def __init__(self,input_dim,d_model=128,n_heads=4,num_layers=3,pred_len=1,dropout=0.1,
        power_idx=None,          # 输入特征中 power 所在列索引，例如 0 / 5 / -1
        recent_window=6,use_persistence=True     # 是否使用 last_power + delta
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.power_idx = power_idx
        self.recent_window = recent_window
        self.use_persistence = use_persistence

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # qTransformer blocks
        self.blocks = nn.ModuleList([
            QTransformerMambaBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(d_model)
        # Attention pooling for sequence aggregation
        self.attention_pool = nn.Linear(d_model, 1)

        # Head: [last_token, recent_pool, global_pool] -> delta / prediction
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

        # 线性捷径：从最后一个原始时刻直接映射到输出
        self.shortcut = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape (B, T, D), got {tuple(x.shape)}")

        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        x_raw = x  # 保留原始输入，供 persistence residual / shortcut 使用

        # Embedding + positional encoding
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.pos_enc(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        # Final norm
        x = self.norm(x)
        # Multi-view temporal summary
        last_token = x[:, -1, :]  # (B, d_model)
        rw = min(self.recent_window, T)
        recent_pool = x[:, -rw:, :].mean(dim=1)  # (B, d_model)

        weights = torch.softmax(self.attention_pool(x), dim=1)  # (B, T, 1)
        global_pool = torch.sum(x * weights, dim=1)             # (B, d_model)

        feat = torch.cat([last_token, recent_pool, global_pool], dim=-1)  # (B, 3*d_model)
        delta = self.head(feat)  # (B, pred_len)
        # 线性shortcut
        out = delta + self.shortcut(x_raw[:, -1, :])
        # persistence residual
        if self.use_persistence and (self.power_idx is not None):
            last_power = x_raw[:, -1, self.power_idx].unsqueeze(-1)  # (B, 1)
            if self.pred_len == 1:
                base = last_power
            else:
                base = last_power.expand(-1, self.pred_len)
            out = out + base
        return out

# python wind_compare_models.py --pred_len 3
# python wind_compare_models.py --pred_len 6
# python wind_compare_models.py --pred_len 9
# python wind_compare_models.py --pred_len 12
# python wind_compare_models.py 