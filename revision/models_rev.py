# -*- coding: utf-8 -*-
"""
revision/models_rev.py — faithful, dependency-light re-implementations of the
QFAM / FAM model family for the EGY-D-26-19497 revision experiments.

Ground truth (code audit, see Response_to_Reviewers Appendix):
  * QFAM  = WindQMambaformer  (src/model_qmambaformer.py): FlowQAttention with a
    Hybrid Quantum Projector on the attention output, real mamba_ssm Mamba branch
    (d_state=64, d_conv=4, expand=3), N=2, H=8, 2-layer prediction head.
    Total parameters = 652,573 (asserted).
  * FAM   = WindMambaformer   (src/model_mambaformer.py): identical backbone but a
    classical linear output projection and a 3-layer prediction head.
    Total parameters = 764,549 (asserted).
  * QFAM-C = QFAM with the VQC replaced by a parameter-matched classical
    GELU bottleneck (ClassicalProjector of run_qfam_ablation.py). Total
    652,549 = 652,573 - 24 VQC angles of the two HQPs.
  * QFAM-R = QFAM with the VQC angles drawn at random and frozen. Total 652,573.

The VQC is implemented as an exact, batched, differentiable 6-qubit state-vector
simulator (RY angle encoding; per layer RX then RY on every qubit + CNOT ring;
Pauli-Z expectation readout) — mathematically identical to the pyvqnet
Angle_VQC_chip used originally, but portable and GPU-friendly.
The Mamba branch is the selective state-space model of Gu & Dao (2023),
parameter-compatible with the official mamba_ssm package (230,016 params/block
at d=128, d_state=64, d_conv=4, expand=3).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _raw_ckpt


def _ckpt(module, x):
    """Gradient checkpointing wrapper (non-reentrant). The recomputed
    forward is bit-identical (all ops deterministic, no dropout inside)."""
    return _raw_ckpt(module, x, use_reentrant=False)

# ----------------------------------------------------------------------
# Positional encoding (copy of the original)
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------------------------------------------------------------
# Exact batched 6-qubit state-vector VQC (RY encoding, RX/RY + CNOT ring)
# ----------------------------------------------------------------------
class TorchVQC(nn.Module):
    """6-qubit VQC: |0>^6 -> RY(x_i) per qubit -> L_q x [RY(beta)RX(alpha) per
    qubit + CNOT ring] -> <Z_i>. Angles (alpha, beta) trainable.
    Bit i of the basis-state index corresponds to qubit i (little-endian)."""

    def __init__(self, n_bit: int = 6, qlayer: int = 1):
        super().__init__()
        self.n_bit = n_bit
        self.qlayer = qlayer
        # pyvqnet quantum_uniform analogue
        self.paras = nn.Parameter(
            (torch.rand(qlayer, n_bit, 2, dtype=torch.float32) * 2 - 1) * math.pi)
        dim = 1 << n_bit
        idx = torch.arange(dim)
        self.register_buffer('_bits', torch.stack([idx >> i & 1 for i in range(n_bit)], 1).float())
        # precompute index pairs for single-qubit gates
        pairs = []
        for i in range(n_bit):
            a = torch.arange(dim)
            b = a | (1 << i)
            mask = (a >> i & 1) == 0
            pairs.append((a[mask].clone(), b[mask].clone()))
        self._pairs = pairs

    # -- gate primitives on a batched state [B, dim] (complex) --
    def _ry(self, psi, theta, q):
        """theta: [B] rotation angle for qubit q."""
        a, b = self._pairs[q]
        c, s = torch.cos(theta / 2).unsqueeze(1), torch.sin(theta / 2).unsqueeze(1)
        pa, pb = psi[:, a], psi[:, b]
        psi = psi.clone()
        psi[:, a] = c * pa - s * pb
        psi[:, b] = s * pa + c * pb
        return psi

    def _rx(self, psi, theta, q):
        a, b = self._pairs[q]
        c, s = torch.cos(theta / 2).unsqueeze(1), torch.sin(theta / 2).unsqueeze(1)
        pa, pb = psi[:, a], psi[:, b]
        psi = psi.clone()
        psi[:, a] = c * pa - 1j * s * pb
        psi[:, b] = -1j * s * pa + c * pb
        return psi

    def _cnot(self, psi, ctrl, targ):
        dim = psi.shape[1]
        idx = torch.arange(dim, device=psi.device)
        apply = (idx >> ctrl & 1) == 1
        new_idx = idx.clone()
        new_idx[apply] = idx[apply] ^ (1 << targ)
        return psi[:, new_idx]

    def forward(self, x):
        """x: [B, n_bit] encoded angles (already compressed features)."""
        B = x.shape[0]
        psi = torch.zeros(B, 1 << self.n_bit, dtype=torch.complex64, device=x.device)
        psi[:, 0] = 1.0
        for i in range(self.n_bit):
            psi = self._ry(psi, x[:, i], i)
        for l in range(self.qlayer):
            w = self.paras[l]
            for j in range(self.n_bit):
                psi = self._rx(psi, w[j, 0].expand(B), j)
                psi = self._ry(psi, w[j, 1].expand(B), j)
            for j in range(self.n_bit - 1):
                psi = self._cnot(psi, j, j + 1)
            if self.n_bit > 1:
                psi = self._cnot(psi, self.n_bit - 1, 0)
        prob = (psi.real ** 2 + psi.imag ** 2)  # [B, dim]
        # <Z_i> = P(bit_i = 0) - P(bit_i = 1) = 1 - 2 * P(bit_i = 1)
        z = 1.0 - 2.0 * (prob @ self._bits)  # [B, n_bit]
        return z


# (prob @ bits gives the per-qubit probability that the bit is 1)


# ----------------------------------------------------------------------
# Hybrid Quantum Projector and classical control
# ----------------------------------------------------------------------
class HybridQuantumProjector(nn.Module):
    """128->6 (no bias) -> VQC -> 6->128 (bias)."""

    def __init__(self, in_dim, out_dim, bottleneck_dim=6, bias=True,
                 n_bit=6, qlayer=1, variant='vqc'):
        super().__init__()
        self.variant = variant
        self.linear_pre = nn.Linear(in_dim, bottleneck_dim, bias=False)
        if variant == 'vqc' or variant == 'random':
            self.vqc = TorchVQC(n_bit, qlayer)
            if variant == 'random':
                for p in self.vqc.parameters():
                    p.requires_grad_(False)
        elif variant == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(variant)
        self.linear_post = nn.Linear(bottleneck_dim, out_dim, bias=bias)

    def forward(self, x):
        h = self.linear_pre(x)
        if self.variant in ('vqc', 'random'):
            h = self.vqc(h)
        else:
            h = self.act(h)
        return self.linear_post(h)


# ----------------------------------------------------------------------
# Selective state-space block (mamba_ssm parameter-compatible)
# ----------------------------------------------------------------------
class MambaTorch(nn.Module):
    """Selective SSM of Gu & Dao (2023), slow exact scan, torch-only.
    Parameter names/shapes follow the official mamba_ssm package so that the
    parameter count matches the paper configuration exactly."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                bias=True, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # A init: A = -exp(U(1, 16)) per (d_inner, d_state)
        A = torch.exp(torch.rand(self.d_inner, d_state) * (math.log(16.0) - math.log(1.0)) + math.log(1.0))
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        # dt bias init: softplus^{-1}(U(dt_min, dt_max))
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    @staticmethod
    def _chunk_scan(a, b, h_prev):
        """Hillis-Steele inclusive scan of one length-C chunk, chained with
        the incoming state h_prev: returns h_t for the chunk."""
        c_len = a.shape[1]
        k = 1
        while k < c_len:
            ones = torch.ones_like(a[:, :k])
            a_prev = torch.cat([ones, a[:, :-k]], dim=1)
            b_prev = torch.cat([torch.zeros_like(b[:, :k]), b[:, :-k]], 1)
            b = b * a_prev + b_prev
            a = a * a_prev
            k *= 2
        return b + a * h_prev.unsqueeze(1)

    @staticmethod
    def _prefix_scan_plain(dA, dBx, chunk=16):
        """Solve h_t = dA_t * h_{t-1} + dBx_t (elementwise over [.., di, N])
        with h_{-1} = 0 via chunked scans; each chunk is separately
        checkpointed by _prefix_scan so that at most one chunk's
        Hillis-Steele intermediate set is live during backward. Exact (no
        approximation). Returns h_t for all t: [B, L, di, N]."""
        B, L, di, N = dA.shape
        h = torch.zeros(B, di, N, device=dA.device, dtype=dA.dtype)
        outs = []
        for s in range(0, L, chunk):
            ht = _raw_ckpt(MambaTorch._chunk_scan, dA[:, s:s + chunk],
                           dBx[:, s:s + chunk], h, use_reentrant=False)
            outs.append(ht)
            h = ht[:, -1]
        return torch.cat(outs, dim=1)

    @staticmethod
    def _prefix_scan(dA, dBx):
        """Entry point used by MambaTorch.forward. Under autograd the
        per-chunk Hillis-Steele intermediates are recomputed in backward
        instead of stored; recomputation is deterministic, so gradients
        are unchanged."""
        return MambaTorch._prefix_scan_plain(dA, dBx)

    def forward(self, x):
        """x: [B, L, d_model] -> [B, L, d_model]"""
        B, L, _ = x.shape
        xz = self.in_proj(x)                       # [B, L, 2*d_inner]
        xz, z = xz.chunk(2, dim=-1)                # each [B, L, d_inner]
        xz = xz.transpose(1, 2)                    # [B, d_inner, L]
        xz = self.conv1d(xz)[..., :L]              # causal
        xz = F.silu(xz)
        x_dbl = self.x_proj(xz.transpose(1, 2))    # [B, L, dt_rank + 2N]
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))          # [B, L, d_inner]
        A = -torch.exp(self.A_log.float())          # [d_inner, N]
        dA = torch.exp(dt.unsqueeze(-1) * A)        # [B, L, d_inner, N]
        # single [B,L,di,N] temporary: fold dt and xz together first
        dBx = (dt * xz.transpose(1, 2)).unsqueeze(-1) * Bm.unsqueeze(2)
        h = self._prefix_scan(dA, dBx)              # [B, L, d_inner, N]
        y = (h * Cm.unsqueeze(2)).sum(-1) + self.D * xz.transpose(1, 2)
        y = y * F.silu(z)
        return self.out_proj(y)


# ----------------------------------------------------------------------
# Flow attentions (verbatim copies of the audited implementations)
# ----------------------------------------------------------------------
class FlowQAttention(nn.Module):
    """Flow attention whose output projection is the HQP (or control)."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, variant='vqc',
                 qbits=6, qlayer=1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = HybridQuantumProjector(dim, dim, bottleneck_dim=qbits,
                                           bias=True, n_bit=qbits, qlayer=qlayer,
                                           variant=variant)
        self.attn_drop = nn.Dropout(attn_drop)

    def kernel(self, x):
        return torch.sigmoid(x)

    def my_sum(self, a, b):
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.kernel(q)
        k = self.kernel(k)
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        x = x_update.transpose(1, 2).reshape(B, N, C)
        x = x.reshape(B * N, C)
        x = self.proj(x)
        x = x.reshape(B, N, C)
        return x


class FlowAttentionClassic(nn.Module):
    """FAM's flow attention with the classical linear output projection."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def kernel(self, x):
        return torch.sigmoid(x)

    def my_sum(self, a, b):
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.kernel(q)
        k = self.kernel(k)
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        x = x_update.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ----------------------------------------------------------------------
# Full models
# ----------------------------------------------------------------------
class QFAMRev(nn.Module):
    """QFAM (WindQMambaformer): HQP variant in {'vqc','random','gelu'}."""

    def __init__(self, input_dim, model_dim=128, mamba_d_state=64, mamba_d_conv=4,
                 mamba_expand=3, flow_attn_heads=8, pred_horizon=1, dropout=0.1,
                 num_layers=2, variant='vqc', qbits=6, qlayer=1):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(d_model=model_dim, dropout=dropout)
        self.flow_attn_layers = nn.ModuleList([
            FlowQAttention(dim=model_dim, num_heads=flow_attn_heads,
                           attn_drop=dropout, variant=variant, qbits=qbits,
                           qlayer=qlayer)
            for _ in range(num_layers)])
        self.flow_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)])
        self.mamba_layers = nn.ModuleList([
            MambaTorch(model_dim, mamba_d_state, mamba_d_conv, mamba_expand)
            for _ in range(num_layers)])
        self.mamba_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.Sigmoid())
            for _ in range(num_layers)])
        self.fusion_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)])
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(model_dim, model_dim, kernel_size=k, padding=k // 2,
                      groups=model_dim) for k in [1, 3, 5, 7]])
        self.scale_weight = nn.Parameter(torch.ones(4) / 4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.pred_head = nn.Sequential(
            nn.Linear(model_dim * 2, int(model_dim / 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(model_dim / 2), pred_horizon))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            f_out = self.flow_attn_layers[i](x)
            f_out = self.flow_norms[i](f_out + x)
            if self.training and torch.is_grad_enabled():
                # gradient checkpointing: recompute the SSM forward in
                # backward; exact (deterministic ops), ~8 GB/job instead
                # of ~25 GB, which matters on shared GPUs
                m_out = _ckpt(self.mamba_layers[i], x)
            else:
                m_out = self.mamba_layers[i](x)
            m_out = self.mamba_norms[i](m_out + x)
            concat = torch.cat([f_out, m_out], dim=-1)
            gate = self.gate_layers[i](concat)
            fused = gate * f_out + (1 - gate) * m_out
            fused = self.fusion_norms[i](fused + x)
            x = fused
        xt = x.transpose(1, 2)
        sw = F.softmax(self.scale_weight, dim=0)
        feats = [conv(xt) * sw[i] for i, conv in enumerate(self.multi_scale_convs)]
        xm = sum(feats)
        avg_pool = self.global_pool(xm).squeeze(-1)
        max_pool = self.global_max_pool(xm).squeeze(-1)
        return self.pred_head(torch.cat([avg_pool, max_pool], dim=-1))


class FAMRev(nn.Module):
    """FAM (WindMambaformer): classical output projection, 3-layer head."""

    def __init__(self, input_dim, model_dim=128, mamba_d_state=64, mamba_d_conv=4,
                 mamba_expand=3, flow_attn_heads=8, pred_horizon=1, dropout=0.1,
                 num_layers=2):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(d_model=model_dim, dropout=dropout)
        self.flow_attn_layers = nn.ModuleList([
            FlowAttentionClassic(dim=model_dim, num_heads=flow_attn_heads,
                                 attn_drop=dropout)
            for _ in range(num_layers)])
        self.flow_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)])
        self.mamba_layers = nn.ModuleList([
            MambaTorch(model_dim, mamba_d_state, mamba_d_conv, mamba_expand)
            for _ in range(num_layers)])
        self.mamba_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.Sigmoid())
            for _ in range(num_layers)])
        self.fusion_norms = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(num_layers)])
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(model_dim, model_dim, kernel_size=k, padding=k // 2,
                      groups=model_dim) for k in [1, 3, 5, 7]])
        self.scale_weight = nn.Parameter(torch.ones(4) / 4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.pred_head = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.Linear(model_dim, pred_horizon))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if name.startswith('mamba_layers'):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            f_out = self.flow_attn_layers[i](x)
            f_out = self.flow_norms[i](f_out + x)
            if self.training and torch.is_grad_enabled():
                m_out = _ckpt(self.mamba_layers[i], x)
            else:
                m_out = self.mamba_layers[i](x)
            m_out = self.mamba_norms[i](m_out + x)
            concat = torch.cat([f_out, m_out], dim=-1)
            gate = self.gate_layers[i](concat)
            fused = gate * f_out + (1 - gate) * m_out
            fused = self.fusion_norms[i](fused + x)
            x = fused
        xt = x.transpose(1, 2)
        sw = F.softmax(self.scale_weight, dim=0)
        feats = [conv(xt) * sw[i] for i, conv in enumerate(self.multi_scale_convs)]
        xm = sum(feats)
        avg_pool = self.global_pool(xm).squeeze(-1)
        max_pool = self.global_max_pool(xm).squeeze(-1)
        return self.pred_head(torch.cat([avg_pool, max_pool], dim=-1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
