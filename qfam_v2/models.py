"""QFAM: FAM h60 plus one six-qubit correction per forecast.

The only quantum model kept in this release is ``fam_qh60lat``.  The classical
reference is ``fam`` (h64); ``fam_h60`` is the no-circuit head control.  LSTM,
CNN, Transformer and DLinear are accuracy baselines.
"""
from __future__ import annotations

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyvqnet
    pyvqnet.backends.set_backend("torch")
    from pyvqnet.qnn.vqc.sv.torch import (
        QMachine, QModule, RY, RX, RZ, CNOT, MeasureAll,
    )
    HAS_PYVQNET = True
except Exception as _e:  # pragma: no cover - local CPU fallback
    HAS_PYVQNET = False
    _PYVQNET_ERR = _e

# The Mamba/flow-attention reference implementation lives in revision/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from revision.models_rev import MambaTorch, PositionalEncoding  # noqa: E402


# ----------------------------------------------------------------------
# Quantum circuit
# ----------------------------------------------------------------------
if HAS_PYVQNET:
    class PyVQNetCircuit(QModule):
        """6-qubit, L-layer hardware-efficient circuit with Z and X readout."""

        def __init__(self, n_qubits=6, layers=2, measure_x=True):
            super().__init__()
            self.n = n_qubits
            self.layers = layers
            self.qm = QMachine(n_qubits, dtype=pyvqnet.kcomplex64, save_ir=True)
            for i in range(n_qubits):
                setattr(self, f"enc_{i}", RY(has_params=True, trainable=False, wires=i))
            for l in range(layers):
                for i in range(n_qubits):
                    setattr(self, f"rx_{l}_{i}", RX(has_params=True, trainable=True, wires=i))
                    setattr(self, f"ry_{l}_{i}", RY(has_params=True, trainable=True, wires=i))
                    setattr(self, f"rz_{l}_{i}", RZ(has_params=True, trainable=True, wires=i))
                for i in range(n_qubits):
                    setattr(self, f"cx_{l}_{i}", CNOT(wires=[i, (i + 1) % n_qubits]))
            obs = [{f"Z{i}": 1.0} for i in range(n_qubits)]
            if measure_x:
                obs += [{f"X{i}": 1.0} for i in range(n_qubits)]
            self.meas = MeasureAll(obs=obs)

        def forward(self, x):
            self.qm.reset_states(x.shape[0])
            for i in range(self.n):
                getattr(self, f"enc_{i}")(params=x[:, i:i + 1], q_machine=self.qm)
            for l in range(self.layers):
                for i in range(self.n):
                    getattr(self, f"rx_{l}_{i}")(q_machine=self.qm)
                    getattr(self, f"ry_{l}_{i}")(q_machine=self.qm)
                    getattr(self, f"rz_{l}_{i}")(q_machine=self.qm)
                for i in range(self.n):
                    getattr(self, f"cx_{l}_{i}")(q_machine=self.qm)
            out = self.meas(q_machine=self.qm)
            if isinstance(out, (list, tuple)):
                out = torch.stack([torch.as_tensor(o) for o in out], dim=-1)
            return out
else:
    class PyVQNetCircuit(nn.Module):
        def __init__(self, *a, **k):
            raise ImportError(f"pyvqnet is not available: {_PYVQNET_ERR}")




# ----------------------------------------------------------------------
# Exact torch state-vector simulator for the same circuit
# ----------------------------------------------------------------------
class TorchVQCv2(nn.Module):
    """6-qubit RY-encode -> L x (RX, RY, RZ; ring CNOT) -> <Z>,<X>.

    This is the training backend.  It implements exactly the same unitary as
    PyVQNetCircuit, but runs as a batched state vector on the GPU.  The
    PyVQNet circuit remains the hardware/export definition.
    """

    def __init__(self, n_qubits=6, layers=2, measure_x=True):
        super().__init__()
        self.n = n_qubits
        self.layers = layers
        self.measure_x = measure_x
        self.paras = nn.Parameter((torch.rand(layers, n_qubits, 3) * 2 - 1) * math.pi / 2)
        dim = 1 << n_qubits
        idx = torch.arange(dim)
        self.register_buffer("_bits", torch.stack([idx >> i & 1 for i in range(n_qubits)], 1).float())
        pairs = []
        for i in range(n_qubits):
            a = torch.arange(dim)
            b = a | (1 << i)
            m = (a >> i & 1) == 0
            pairs.append((a[m].clone(), b[m].clone()))
        for i, (a, b) in enumerate(pairs):
            self.register_buffer(f"_a{i}", a)
            self.register_buffer(f"_b{i}", b)
        cnot = []
        for c in range(n_qubits):
            t = (c + 1) % n_qubits
            idx2 = torch.arange(dim)
            m = (idx2 >> c & 1) == 1
            idx2[m] = idx2[m] ^ (1 << t)
            cnot.append(idx2)
        self.register_buffer("_cnot_idx", torch.stack(cnot))

    def _ry(self, psi, theta, q):
        a, b = getattr(self, f"_a{q}"), getattr(self, f"_b{q}")
        c, s = torch.cos(theta / 2).unsqueeze(1), torch.sin(theta / 2).unsqueeze(1)
        pa, pb = psi[:, a], psi[:, b]
        psi = psi.clone()
        psi[:, a] = c * pa - s * pb
        psi[:, b] = s * pa + c * pb
        return psi

    def _rx(self, psi, theta, q):
        a, b = getattr(self, f"_a{q}"), getattr(self, f"_b{q}")
        c, s = torch.cos(theta / 2).unsqueeze(1), torch.sin(theta / 2).unsqueeze(1)
        pa, pb = psi[:, a], psi[:, b]
        psi = psi.clone()
        psi[:, a] = c * pa - 1j * s * pb
        psi[:, b] = -1j * s * pa + c * pb
        return psi

    def _rz(self, psi, theta, q):
        a, b = getattr(self, f"_a{q}"), getattr(self, f"_b{q}")
        ph = theta / 2
        psi = psi.clone()
        psi[:, a] = psi[:, a] * torch.exp(-1j * ph).unsqueeze(1)
        psi[:, b] = psi[:, b] * torch.exp(1j * ph).unsqueeze(1)
        return psi

    def _h(self, psi, q):
        a, b = getattr(self, f"_a{q}"), getattr(self, f"_b{q}")
        pa, pb = psi[:, a], psi[:, b]
        psi = psi.clone()
        psi[:, a] = (pa + pb) / math.sqrt(2)
        psi[:, b] = (pa - pb) / math.sqrt(2)
        return psi

    def _apply_cnot(self, psi, q):
        return psi[:, self._cnot_idx[q]]

    def forward(self, x):
        B = x.shape[0]
        psi = torch.zeros(B, 1 << self.n, dtype=torch.complex64, device=x.device)
        psi[:, 0] = 1.0
        for i in range(self.n):
            psi = self._ry(psi, x[:, i], i)
        for l in range(self.layers):
            for i in range(self.n):
                psi = self._rx(psi, self.paras[l, i, 0].expand(B), i)
                psi = self._ry(psi, self.paras[l, i, 1].expand(B), i)
                psi = self._rz(psi, self.paras[l, i, 2].expand(B), i)
            for i in range(self.n):
                psi = self._apply_cnot(psi, i)
        prob = psi.real ** 2 + psi.imag ** 2
        z = 1.0 - 2.0 * (prob @ self._bits)
        if not self.measure_x:
            return z
        psix = psi
        for i in range(self.n):
            psix = self._h(psix, i)
        probx = psix.real ** 2 + psix.imag ** 2
        xx = 1.0 - 2.0 * (probx @ self._bits)
        return torch.cat([z, xx], dim=-1)


# ----------------------------------------------------------------------
# Projectors
# ----------------------------------------------------------------------
class ClassicalProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)


def make_projector(variant, dim, n_qubits=6, layers=2, measure_x=True, backend="torch", low_rank=0, use_gate=False):
    if variant != "classical":
        raise ValueError("Only the classical projector is kept in the QFAM release")
    return ClassicalProjector(dim)




class FlowBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, projector="classical",
                 n_qubits=6, q_layers=2, measure_x=True, qbackend="torch",
                 q_low_rank=0, q_gate=False, trim=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=not trim)
        self.proj = make_projector(projector, dim, n_qubits, q_layers, measure_x,
                                   qbackend, q_low_rank, q_gate)
        self.attn_drop = nn.Dropout(dropout)
        self.mamba = MambaTorch(dim, d_state=16, d_conv=4, expand=2)
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=not trim)
        self.norm_m = nn.LayerNorm(dim, elementwise_affine=not trim)
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim, bias=not trim), nn.Sigmoid())
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=not trim)

    def _kernel(self, x):
        return torch.sigmoid(x)

    def _my_sum(self, a, b):
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self._kernel(q), self._kernel(k)
        sink_in = 1.0 / (self._my_sum(q + 1e-6, k.sum(2) + 1e-6) + 1e-6)
        source_out = 1.0 / (self._my_sum(k + 1e-6, q.sum(2) + 1e-6) + 1e-6)
        cons_sink = self._my_sum(
            q + 1e-6, (k * source_out[:, :, :, None]).sum(2) + 1e-6) + 1e-6
        cons_source = self._my_sum(
            k + 1e-6, (q * sink_in[:, :, :, None]).sum(2) + 1e-6) + 1e-6
        cons_source = torch.clamp(cons_source, -1.0, 1.0)
        sink_alloc = torch.sigmoid(cons_sink * (float(q.shape[2]) / float(k.shape[2])))
        source_comp = torch.softmax(cons_source, dim=-1) * float(k.shape[2])
        kv = k.transpose(-2, -1) @ (v * source_comp[:, :, :, None])
        x_att = ((q @ kv) * sink_in[:, :, :, None]) * sink_alloc[:, :, :, None]
        x_att = x_att.transpose(1, 2).reshape(B, N, C)
        x_att = self.proj(x_att.reshape(B * N, C)).reshape(B, N, C)
        x_att = self.norm_f(x_att + x)
        x_m = self.mamba(x)
        x_m = self.norm_m(x_m + x)
        g = self.gate(torch.cat([x_att, x_m], dim=-1))
        return self.norm_out(g * x_att + (1 - g) * x_m + x)


class FAMv2(nn.Module):
    def __init__(self, input_dim, dim=64, horizon=1, layers=1, heads=8,
                 dropout=0.1, projector="classical", n_qubits=6, q_layers=2,
                 measure_x=True, qbackend="torch", q_low_rank=0, q_gate=False,
                 head_hidden=None, trim=False):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Linear(input_dim, dim, bias=not trim)
        self.pos = PositionalEncoding(dim, dropout=dropout)
        self.blocks = nn.ModuleList([
            FlowBlock(dim, heads, dropout, projector, n_qubits, q_layers, measure_x, qbackend,
                      q_low_rank, q_gate, trim)
            for _ in range(layers)])
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=not trim)
            for k in [1, 3, 5, 7]])
        self.scale_w = nn.Parameter(torch.ones(4) / 4)
        hh = int(head_hidden) if head_hidden else dim
        self.head = nn.Sequential(
            nn.Linear(2 * dim, hh, bias=not trim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hh, horizon, bias=not trim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, '_skip_init', False):
                    continue
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x0 = x
        x = self.pos(self.inp(x))
        for b in self.blocks:
            x = b(x)
        xt = x.transpose(1, 2)
        sw = torch.softmax(self.scale_w, dim=0)
        xm = sum(c(xt) * sw[i] for i, c in enumerate(self.convs))
        avg = xm.mean(-1)
        mx = xm.max(-1).values
        return self.head(torch.cat([avg, mx], -1))




class FAMQuantumCorrection(nn.Module):
    """FAM trunk plus a tiny quantum correction on pooled power/wind features."""

    def __init__(self, input_dim, dim=64, horizon=1, qbackend="torch", fam_hidden=64, latent=False, q_layers=2, q_qubits=6):
        super().__init__()
        self.fam = FAMv2(input_dim, dim, horizon, layers=1,
                         projector="classical", qbackend=qbackend, head_hidden=fam_hidden)
        self.idx = (23, 59, 95, 4, 40, 76)  # pooled mean/std/last of Power, ws100
        self.skip = nn.Linear(6, horizon)
        self.qhead = QuantumHeadLatent(6, q_qubits, q_layers, qbackend)

    def forward(self, x):
        pooled = torch.cat([x.mean(1), x.std(1), x[:, -1]], dim=-1)
        feat = pooled[:, list(self.idx)]
        return self.fam(x) + self.skip(feat) + self.qhead(feat)


class QuantumHeadLatent(nn.Module):
    """VQC readout that also keeps the encoded latent (q plus z)."""

    def __init__(self, n_features=6, n_qubits=6, layers=2, backend="torch"):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.q = TorchVQCv2(n_qubits, layers, True)
        self.post = nn.Linear(2 * n_qubits + n_qubits, 1)
        nn.init.zeros_(self.post.weight); nn.init.zeros_(self.post.bias)

    def forward(self, x):
        z = torch.tanh(x)
        if self.n_qubits > self.n_features:
            z = F.pad(z, (0, self.n_qubits - self.n_features))
        q = self.q(z * math.pi).to(device=x.device, dtype=x.dtype)
        return self.post(torch.cat([q, z], dim=-1))


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
class CNN(nn.Module):
    """Compact 1D CNN baseline over the 96x36 window."""

    def __init__(self, input_dim, dim=32, horizon=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(dim, horizon))

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class LSTM(nn.Module):
    def __init__(self, input_dim, dim=64, horizon=1, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, dim, bias=True)
        self.rnn = nn.LSTM(dim, dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.rnn(self.drop(self.inp(x)))
        return self.head(h[:, -1])


class Transformer(nn.Module):
    def __init__(self, input_dim, dim=64, horizon=1, heads=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, dim, bias=True)
        self.pos = PositionalEncoding(dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon))

    def forward(self, x):
        h = self.enc(self.pos(self.inp(x)))
        return self.head(h.mean(1))


class DLinear(nn.Module):
    def __init__(self, input_dim, seq_len=96, horizon=1, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.trend = nn.Linear(seq_len, horizon)
        self.season = nn.Linear(seq_len, horizon)
        self.out = nn.Linear(input_dim * horizon, horizon)

    def forward(self, x):
        # x: [B,L,F]
        xt = x.transpose(1, 2)  # [B,F,L]
        pad = self.kernel // 2
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"),
                             self.kernel, stride=1)
        season = xt - trend
        y = self.trend(trend) + self.season(season)  # [B,F,H]
        return self.out(y.reshape(y.shape[0], -1))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name, input_dim, seq_len=96, horizon=1, dim=64, qbackend="torch"):
    name = name.lower()
    if name == "persistence":
        return None
    if name == "lstm":
        return LSTM(input_dim, dim, horizon)
    if name == "cnn":
        return CNN(input_dim, seq_len, horizon)
    if name == "transformer":
        return Transformer(input_dim, dim, horizon)
    if name == "dlinear":
        return DLinear(input_dim, seq_len, horizon)
    if name == "fam":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical", qbackend=qbackend)
    if name == "fam_h60":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, head_hidden=60)
    if name == "fam_qh60lat":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend,
                                    fam_hidden=60, latent=True)
    raise ValueError(f"Unknown model: {name}")
