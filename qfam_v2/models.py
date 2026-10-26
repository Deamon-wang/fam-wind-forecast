"""Classical and quantum Flow-Attention Mamba models for the aligned data.

The quantum projector is a residual adapter: a classical linear projection plus
a zero-initialised quantum correction.  The circuit is built with the PyVQNet
torch VQC API (QMachine / RY / RX / RZ / CNOT / MeasureAll), which is the same
circuit family that can be exported to pyqpanda3 for hardware runs.
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


class ClassicalResidualProjector(nn.Module):
    """Parameter-matched classical control for the quantum residual."""

    def __init__(self, dim, hidden=12):
        super().__init__()
        self.base = nn.Linear(dim, dim)
        self.pre = nn.Linear(dim, hidden, bias=False)
        self.ln = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                 nn.Linear(2 * hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1]._skip_init = True

    def forward(self, x):
        z = torch.tanh(self.ln(self.pre(x)))
        return self.base(x) + self.mlp(z)


class QuantumResidualProjector(nn.Module):
    """classical linear projection + zero-init quantum correction."""

    def __init__(self, dim, n_qubits=6, layers=2, measure_x=True,
                 random_angles=False, backend="torch", low_rank=0, use_gate=False,
                 init_scale=0.0):
        super().__init__()
        self.base = nn.Linear(dim, dim)
        self.pre = nn.Linear(dim, n_qubits, bias=False)
        self.ln = nn.LayerNorm(n_qubits)
        self.scale = nn.Parameter(torch.ones(n_qubits))
        self.q = (TorchVQCv2(n_qubits, layers, measure_x) if backend == "torch"
                  else PyVQNetCircuit(n_qubits, layers, measure_x))
        n_out = n_qubits * (2 if measure_x else 1)
        self.use_gate = use_gate
        self.gate_param = nn.Parameter(torch.tensor(-4.0)) if use_gate else None
        if low_rank > 0:
            self.post_down = nn.Linear(n_out + n_qubits, low_rank, bias=False)
            self.post_up = nn.Linear(low_rank, dim, bias=False)
            nn.init.zeros_(self.post_up.weight)
            self.post_up._skip_init = True
            self.post = None
        else:
            self.post = nn.Linear(n_out + n_qubits, dim)
            if init_scale > 0:
                nn.init.normal_(self.post.weight, 0.0, init_scale)
            else:
                nn.init.zeros_(self.post.weight)
            nn.init.zeros_(self.post.bias)
            self.post._skip_init = True
            self.post_down = None; self.post_up = None
        if random_angles:
            for p in self.q.parameters():
                p.requires_grad_(False)
        self.last_z_std = None

    def forward(self, x):
        z = self.ln(self.pre(x))
        self.last_z_std = z.std(dim=0).mean()
        z = torch.tanh(z) * self.scale
        q = self.q(z * math.pi)
        q = q.to(device=z.device, dtype=z.dtype)
        feat = torch.cat([q, z], dim=-1)
        if self.post is not None:
            corr = self.post(feat)
        else:
            corr = self.post_up(self.post_down(feat))
        if self.use_gate:
            corr = torch.sigmoid(self.gate_param) * corr
        return self.base(x) + corr


class QuantumBottleneckProjector(nn.Module):
    """Original HQP-style replacement: no classical skip."""

    def __init__(self, dim, n_qubits=6, layers=2, measure_x=True, backend="torch"):
        super().__init__()
        self.pre = nn.Linear(dim, n_qubits, bias=False)
        self.ln = nn.LayerNorm(n_qubits)
        self.scale = nn.Parameter(torch.ones(n_qubits))
        self.q = (TorchVQCv2(n_qubits, layers, measure_x) if backend == "torch"
                  else PyVQNetCircuit(n_qubits, layers, measure_x))
        n_out = n_qubits * (2 if measure_x else 1)
        self.post = nn.Linear(n_out + n_qubits, dim)
        self.last_z_std = None

    def forward(self, x):
        z = self.ln(self.pre(x))
        self.last_z_std = z.std(dim=0).mean()
        z = torch.tanh(z) * self.scale
        q = self.q(z * math.pi)
        q = q.to(device=z.device, dtype=z.dtype)
        return self.post(torch.cat([q, z], dim=-1))


def make_projector(variant, dim, n_qubits=6, layers=2, measure_x=True, backend="torch", low_rank=0, use_gate=False):
    if variant == "classical":
        return ClassicalProjector(dim)
    if variant == "quantum_res":
        return QuantumResidualProjector(dim, n_qubits, layers, measure_x, backend=backend,
                                        low_rank=low_rank, use_gate=use_gate)
    if variant == "quantum_rand":
        return QuantumResidualProjector(dim, n_qubits, layers, measure_x,
                                        random_angles=True, backend=backend,
                                        low_rank=low_rank, use_gate=use_gate)
    if variant == "quantum_bot":
        return QuantumBottleneckProjector(dim, n_qubits, layers, measure_x, backend=backend)
    if variant == "classical_res":
        return ClassicalResidualProjector(dim)
    raise ValueError(variant)




class QuantumHead(nn.Module):
    """Quantum residual correction on the current-observation vector.

    A 6-angle encoding of the last observed feature vector is passed through
    the same VQC family and mapped to a scalar correction of the forecast.
    The correction is zero-initialised, so the model starts as the classical
    FAM head.
    """

    def __init__(self, input_dim, n_qubits=6, layers=2, backend="torch",
                 random_angles=False, init_scale=0.0):
        super().__init__()
        self.pre = nn.Linear(input_dim, n_qubits, bias=False)
        self.ln = nn.LayerNorm(n_qubits)
        self.scale = nn.Parameter(torch.ones(n_qubits))
        self.q = (TorchVQCv2(n_qubits, layers, True) if backend == "torch"
                  else PyVQNetCircuit(n_qubits, layers, True))
        self.post = nn.Linear(3 * n_qubits, 1)
        if init_scale > 0:
            nn.init.normal_(self.post.weight, 0.0, init_scale)
        else:
            nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)
        self.post._skip_init = True
        if random_angles:
            for p in self.q.parameters():
                p.requires_grad_(False)

    def forward(self, last):
        z = torch.tanh(self.ln(self.pre(last))) * self.scale
        q = self.q(z * math.pi).to(device=z.device, dtype=z.dtype)
        return self.post(torch.cat([q, z], dim=-1))


class ClassicalHeadResidual(nn.Module):
    def __init__(self, input_dim, hidden=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * hidden), nn.GELU(),
            nn.Linear(2 * hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.net[-1]._skip_init = True

    def forward(self, last):
        return self.net(last)


# ----------------------------------------------------------------------
# Flow-Attention + Mamba block
# ----------------------------------------------------------------------
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


class QuantumPredHead(nn.Module):
    """Replace FAM's 128->64->1 head with a quantum temporal readout."""

    def __init__(self, in_dim, horizon=1, qbackend="torch", n_qubits=6, layers=2):
        super().__init__()
        self.pre = nn.Linear(in_dim, n_qubits, bias=False)
        self.lin = nn.Linear(in_dim, horizon)
        self.qhead = QuantumHeadLatent(n_qubits, n_qubits, layers, qbackend)

    def forward(self, h):
        return self.lin(h) + self.qhead(self.pre(h))


class FAMv2(nn.Module):
    def __init__(self, input_dim, dim=64, horizon=1, layers=1, heads=8,
                 dropout=0.1, projector="classical", n_qubits=6, q_layers=2,
                 measure_x=True, qbackend="torch", q_low_rank=0, q_gate=False,
                 qhead="none", phys_idx=(3, 4, 7, 8, 0, 23), head_hidden=None,
                 trim=False, head_type="mlp"):
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
        if head_type == "quantum":
            self.head = QuantumPredHead(2 * dim, horizon, qbackend)
        elif head_type == "quantum12":
            self.head = QuantumPredHead(2 * dim, horizon, qbackend, n_qubits=12)
        elif head_type == "quantum16":
            self.head = QuantumPredHead(2 * dim, horizon, qbackend, n_qubits=16)
        else:
            self.head = nn.Sequential(
                nn.Linear(2 * dim, hh, bias=not trim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hh, horizon, bias=not trim))
        self.phys_idx = tuple(phys_idx)
        if qhead == "quantum":
            self.qhead = QuantumHead(input_dim, 6, 2, qbackend)
        elif qhead == "phys":
            self.qhead = QuantumHead(len(self.phys_idx), 6, 2, qbackend)
        elif qhead == "random":
            self.qhead = QuantumHead(input_dim, 6, 2, qbackend, random_angles=True)
        elif qhead == "classical":
            self.qhead = ClassicalHeadResidual(input_dim)
        else:
            self.qhead = None
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
        y = self.head(torch.cat([avg, mx], -1))
        if self.qhead is not None:
            last = x0[:, -1, :]
            if (hasattr(self.qhead, "pre")
                    and self.qhead.pre.in_features != last.shape[-1]):
                last = last[:, list(self.phys_idx)]
            y = y + self.qhead(last)
        return y




class QDLinear(nn.Module):
    """DLinear plus a zero-initialised quantum residual head.

    The linear branch is the strongest overall model in the aligned benchmark;
    the quantum head learns a small nonlinear correction from the current
    physical state.  With the residual initialised at zero the model starts as
    DLinear and can only improve (or be ignored) during training.
    """

    def __init__(self, input_dim, seq_len=96, horizon=1, dim=64, qbackend="torch",
                 qhead="quantum", phys_idx=(3, 4, 7, 8, 0, 23)):
        super().__init__()
        self.base = DLinear(input_dim, seq_len, horizon)
        self.phys_idx = tuple(phys_idx)
        if qhead == "quantum":
            self.qhead = QuantumHead(len(self.phys_idx), 6, 2, qbackend)
        elif qhead == "random":
            self.qhead = QuantumHead(len(self.phys_idx), 6, 2, qbackend,
                                     random_angles=True)
        elif qhead == "classical":
            self.qhead = ClassicalHeadResidual(len(self.phys_idx))
        else:
            self.qhead = None

    def forward(self, x):
        y = self.base(x)
        if self.qhead is not None:
            y = y + self.qhead(x[:, -1, list(self.phys_idx)])
        return y




class DLinearMLP(nn.Module):
    """DLinear plus a classical nonlinear residual on window summaries."""

    def __init__(self, input_dim, seq_len=96, horizon=1, hidden=32):
        super().__init__()
        self.base = DLinear(input_dim, seq_len, horizon)
        self.net = nn.Sequential(
            nn.Linear(3 * input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, horizon))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1]._skip_init = True

    def forward(self, x):
        s = torch.cat([x.mean(1), x.std(1), x[:, -1]], dim=-1)
        return self.base(x) + self.net(s)


class DLinearQSummary(nn.Module):
    """Quantum counterpart of DLinearMLP: VQC on the same summary vector."""

    def __init__(self, input_dim, seq_len=96, horizon=1, qbackend="torch",
                 n_qubits=6, layers=2):
        super().__init__()
        self.base = DLinear(input_dim, seq_len, horizon)
        self.qhead = QuantumHead(3 * input_dim, n_qubits, layers, qbackend)

    def forward(self, x):
        s = torch.cat([x.mean(1), x.std(1), x[:, -1]], dim=-1)
        return self.base(x) + self.qhead(s)




class QuantumHeadNoPre(nn.Module):
    """Direct angle encoding of a small feature vector into the VQC."""

    def __init__(self, n_features=6, n_qubits=6, layers=1, backend="torch"):
        super().__init__()
        self.q = (TorchVQCv2(n_qubits, layers, True) if backend == "torch"
                  else PyVQNetCircuit(n_qubits, layers, True))
        self.post = nn.Linear(2 * n_qubits, 1)
        nn.init.zeros_(self.post.weight); nn.init.zeros_(self.post.bias)
        self.post._skip_init = True

    def forward(self, x):
        q = self.q(torch.tanh(x) * math.pi).to(device=x.device, dtype=x.dtype)
        return self.post(q)


class QDLinearLite(nn.Module):
    """Quantumised DLinear with fewer parameters than the full DLinear.

    The per-channel trend/season projections are kept (192 parameters).  Their
    six selected outputs are mixed by a linear skip (6) plus a 6-qubit,
    one-layer VQC readout (18 angles + 13 readout), total 229 parameters versus
    231 for DLinear.
    """

    def __init__(self, input_dim, seq_len=96, horizon=1, qbackend="torch",
                 idx=(3, 4, 7, 8, 0, 23), qlayer=1, bias=False):
        super().__init__()
        self.idx = tuple(idx)
        self.trend = nn.Linear(seq_len, horizon, bias=False)
        self.season = nn.Linear(seq_len, horizon, bias=False)
        self.skip = nn.Linear(len(self.idx), horizon, bias=bias)
        self.qhead = QuantumHeadNoPre(len(self.idx), len(self.idx), qlayer, qbackend)

    def forward(self, x):
        xt = x.transpose(1, 2)
        pad = 12
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"), 25, stride=1)
        season = xt - trend
        f = (self.trend(trend) + self.season(season))[:, list(self.idx), :]  # [B,k,H]
        f = f.squeeze(-1)
        return self.skip(f) + self.qhead(f)


class QDLinearLiteProj(nn.Module):
    """Quantumised DLinear with a fixed 36->6 projection.

    The projection has no trainable parameters: ``group`` averages six
    contiguous channels, ``rand`` uses a fixed orthogonal random projection.
    The parameter count is identical to QDLinearLite (229).
    """

    def __init__(self, input_dim, seq_len=96, horizon=1, qbackend="torch",
                 mode="group", seed=0, qlayer=1):
        super().__init__()
        self.trend = nn.Linear(seq_len, horizon, bias=False)
        self.season = nn.Linear(seq_len, horizon, bias=False)
        if mode == "group":
            p = torch.zeros(input_dim, 6)
            g = max(1, input_dim // 6)
            for j in range(6):
                lo, hi = j * g, min((j + 1) * g, input_dim)
                p[lo:hi, j] = 1.0 / max(1, hi - lo)
        elif mode == "rand":
            gen = torch.Generator().manual_seed(seed)
            q, _ = torch.linalg.qr(torch.randn(input_dim, 6, generator=gen))
            p = q[:, :6]
        else:
            raise ValueError(mode)
        self.register_buffer("proj", p)
        self.skip = nn.Linear(6, horizon, bias=False)
        self.qhead = QuantumHeadNoPre(6, 6, qlayer, qbackend)

    def forward(self, x):
        xt = x.transpose(1, 2)
        pad = 12
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"), 25, stride=1)
        season = xt - trend
        f = (self.trend(trend) + self.season(season))[:, :, 0]  # [B,F]
        f = f @ self.proj
        return self.skip(f) + self.qhead(f)


class QDLinearClassical(nn.Module):
    """Parameter-matched classical control for QDLinearLite.

    Same trend/season trunk and six selected channels; the quantum head is
    replaced by a quadratic classical head with 31 parameters, so the total is
    229 (same as QDLinearLite).
    """

    def __init__(self, input_dim, seq_len=96, horizon=1, idx=(23, 4, 7, 3, 16, 17)):
        super().__init__()
        self.idx = tuple(idx)
        self.trend = nn.Linear(seq_len, horizon, bias=False)
        self.season = nn.Linear(seq_len, horizon, bias=False)
        self.skip = nn.Linear(len(self.idx), horizon, bias=False)
        self.lin = nn.Linear(len(self.idx), horizon)
        self.sq = nn.Linear(len(self.idx), horizon, bias=False)
        k = len(self.idx)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        self.pairs = pairs
        self.inter = nn.Linear(len(pairs), horizon, bias=False)
        self.extra = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        xt = x.transpose(1, 2)
        pad = 12
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"), 25, stride=1)
        season = xt - trend
        f = (self.trend(trend) + self.season(season))[:, list(self.idx), :].squeeze(-1)
        inter = torch.stack([f[:, i] * f[:, j] for i, j in self.pairs], dim=-1)
        out = self.skip(f) + self.lin(f) + self.sq(f * f) + self.inter(inter)
        return out + self.extra[0] + self.extra[1]


class QDLinear4Q(nn.Module):
    """DLinear trunk with a 4-qubit, 2-layer VQC output (229 parameters)."""

    def __init__(self, input_dim, seq_len=96, horizon=1, qbackend="torch"):
        super().__init__()
        self.idx = (23, 4, 7, 16)
        self.trend = nn.Linear(seq_len, horizon, bias=False)
        self.season = nn.Linear(seq_len, horizon, bias=False)
        self.skip = nn.Linear(4, horizon, bias=False)
        self.qhead = QuantumHeadNoPre(4, 4, 2, qbackend)

    def forward(self, x):
        xt = x.transpose(1, 2)
        pad = 12
        trend = F.avg_pool1d(F.pad(xt, (pad, pad), mode="replicate"), 25, stride=1)
        season = xt - trend
        f = (self.trend(trend) + self.season(season))[:, list(self.idx), :].squeeze(-1)
        return self.skip(f) + self.qhead(f)


class QLSTMLite(nn.Module):
    """Compact LSTM with a 6-qubit VQC readout (about 9.8k parameters)."""

    def __init__(self, input_dim, dim=32, horizon=1, qbackend="torch"):
        super().__init__()
        self.inp = nn.Linear(input_dim, dim)
        self.rnn = nn.LSTM(dim, dim, num_layers=1, batch_first=True)
        self.pre = nn.Linear(dim, 6)
        self.skip = nn.Linear(dim, horizon)
        self.qhead = QuantumHeadNoPre(6, 6, 1, qbackend)

    def forward(self, x):
        h, _ = self.rnn(self.inp(x))
        z = h[:, -1]
        return self.skip(z) + self.qhead(self.pre(z))


class LSTMLite(nn.Module):
    """Parameter-matched classical control for QLSTMLite (linear head)."""

    def __init__(self, input_dim, dim=32, horizon=1):
        super().__init__()
        self.inp = nn.Linear(input_dim, dim)
        self.rnn = nn.LSTM(dim, dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(dim, horizon)

    def forward(self, x):
        h, _ = self.rnn(self.inp(x))
        return self.head(h[:, -1])


class FAMQuantumCorrection(nn.Module):
    """FAM trunk plus a tiny quantum correction on pooled power/wind features."""

    def __init__(self, input_dim, dim=64, horizon=1, qbackend="torch", fam_hidden=64, latent=False, q_layers=2, q_qubits=6):
        super().__init__()
        self.fam = FAMv2(input_dim, dim, horizon, layers=1,
                         projector="classical", qbackend=qbackend, head_hidden=fam_hidden)
        self.idx = (23, 59, 95, 4, 40, 76)  # pooled mean/std/last of Power, ws100
        self.skip = nn.Linear(6, horizon)
        self.qhead = (QuantumHeadLatent(6, q_qubits, q_layers, qbackend) if latent
                       else QuantumHeadNoPre(6, 6, q_layers, qbackend))

    def forward(self, x):
        pooled = torch.cat([x.mean(1), x.std(1), x[:, -1]], dim=-1)
        feat = pooled[:, list(self.idx)]
        return self.fam(x) + self.skip(feat) + self.qhead(feat)


class FAMClassicalCorrection(nn.Module):
    """Classical control for FAMQuantumCorrection: same trunk, MLP on 6 pooled features."""

    def __init__(self, input_dim, dim=64, horizon=1, fam_hidden=60):
        super().__init__()
        self.fam = FAMv2(input_dim, dim, horizon, layers=1,
                         projector="classical", head_hidden=fam_hidden)
        self.idx = (23, 59, 95, 4, 40, 76)
        self.skip = nn.Linear(6, horizon)
        self.head = nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, horizon))

    def forward(self, x):
        pooled = torch.cat([x.mean(1), x.std(1), x[:, -1]], dim=-1)
        feat = pooled[:, list(self.idx)]
        return self.fam(x) + self.skip(feat) + self.head(feat)


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
class FAMTemporalQuantum(nn.Module):
    """FAM trunk with a 6-qubit readout on low-frequency power features."""

    def __init__(self, input_dim, dim=64, horizon=1, qbackend="torch",
                 fam_hidden=63, quantum=True):
        super().__init__()
        self.fam = FAMv2(input_dim, dim, horizon, layers=1,
                         projector="classical", head_hidden=fam_hidden)
        self.skip = nn.Linear(6, horizon)
        if quantum:
            self.head = QuantumHeadLatent(6, 6, 2, qbackend)
        else:
            self.head = nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, horizon))

    def forward(self, x):
        p = x[:, :, 23]
        spec = torch.fft.rfft(p, dim=1)
        low = spec[:, 1:4]
        feat = torch.cat([low.real, low.imag], dim=-1)
        feat = torch.tanh(feat * 0.1)
        return self.fam(x) + self.skip(feat) + self.head(feat)


class QuantumHeadZ(nn.Module):
    """Z-only readout control for QuantumHeadLatent."""

    def __init__(self, n_features=6, n_qubits=6, layers=2, backend="torch"):
        super().__init__()
        self.q = TorchVQCv2(n_qubits, layers, False)
        self.post = nn.Linear(n_qubits + n_qubits, 1)
        nn.init.zeros_(self.post.weight); nn.init.zeros_(self.post.bias)

    def forward(self, x):
        z = torch.tanh(x)
        q = self.q(z * math.pi).to(device=x.device, dtype=x.dtype)
        return self.post(torch.cat([q, z], dim=-1))


class FAMQuantumCorrectionRand(FAMQuantumCorrection):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for p in self.qhead.q.parameters():
            p.requires_grad_(False)


class FAMQuantumCorrectionZ(FAMQuantumCorrection):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.qhead = QuantumHeadZ(6, 6, 2, "torch")


class FAMHybridCorrection(nn.Module):
    """FAM trunk with a classical pooled MLP plus a small quantum residual."""

    def __init__(self, input_dim, dim=64, horizon=1, qbackend="torch", fam_hidden=60):
        super().__init__()
        self.fam = FAMv2(input_dim, dim, horizon, layers=1,
                         projector="classical", head_hidden=fam_hidden)
        self.idx = (23, 59, 95, 4, 40, 76)
        self.skip = nn.Linear(6, horizon)
        self.cls = nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, horizon))
        self.qhead = QuantumHeadLatent(6, 6, 2, qbackend)

    def forward(self, x):
        pooled = torch.cat([x.mean(1), x.std(1), x[:, -1]], dim=-1)
        feat = pooled[:, list(self.idx)]
        return self.fam(x) + self.skip(feat) + self.cls(feat) + self.qhead(feat)


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


class MLP(nn.Module):
    """Pooled MLP baseline: mean, std and last observation."""
    def __init__(self, input_dim, seq_len=96, horizon=1, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, horizon))

    def forward(self, x):
        return self.net(torch.cat([x.mean(1), x.std(1), x[:, -1]], dim=-1))


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


class PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len=96, horizon=1, patch=16, stride=8,
                 dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        n = (seq_len - patch) // stride + 1
        self.proj = nn.Linear(patch * input_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, horizon))

    def forward(self, x):
        B, L, F = x.shape
        patches = x.unfold(1, self.patch, self.stride)  # [B,n,F,patch]
        patches = patches.permute(0, 1, 3, 2).reshape(B, -1, self.patch * F)
        h = self.enc(self.proj(patches) + self.pos)
        return self.head(torch.cat([h.mean(1), h[:, -1]], dim=-1))


class ITransformer(nn.Module):
    def __init__(self, input_dim, seq_len=96, horizon=1, dim=64, heads=4,
                 dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(seq_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, horizon))

    def forward(self, x):
        # [B,L,F] -> [B,F,L] -> variate tokens
        h = self.enc(self.inp(x.transpose(1, 2)))
        return self.head(torch.cat([h.mean(1), h[:, -1]], dim=-1))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name, input_dim, seq_len=96, horizon=1, dim=64, qbackend="torch"):
    name = name.lower()
    if name == "persistence":
        return None
    if name == "mlp":
        return MLP(input_dim, seq_len, horizon)
    if name == "lstm":
        return LSTM(input_dim, dim, horizon)
    if name == "transformer":
        return Transformer(input_dim, dim, horizon)
    if name == "dlinear":
        return DLinear(input_dim, seq_len, horizon)
    if name == "patchtst":
        return PatchTST(input_dim, seq_len, horizon, dim=dim)
    if name == "itransformer":
        return ITransformer(input_dim, seq_len, horizon, dim=dim)
    if name == "fam":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical", qbackend=qbackend)
    if name == "fam_qh":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend)
    if name == "fam_cls60":
        return FAMClassicalCorrection(input_dim, dim, horizon, fam_hidden=60)
    if name == "fam_cls64":
        return FAMClassicalCorrection(input_dim, dim, horizon, fam_hidden=64)
    if name == "fam_qhp":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, head_type="quantum")
    if name == "fam_qhp12":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, head_type="quantum12")
    if name == "fam_qhp16":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, head_type="quantum16")
    if name == "fam_tq":
        return FAMTemporalQuantum(input_dim, dim, horizon, qbackend)
    if name == "fam_tc":
        return FAMTemporalQuantum(input_dim, dim, horizon, qbackend, quantum=False)
    if name == "fam_hyb":
        return FAMHybridCorrection(input_dim, dim, horizon, qbackend)
    if name == "fam_qh_lat1":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True, q_layers=1)
    if name == "fam_qh_1":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=False, q_layers=1)
    if name == "fam_qh_2":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=False, q_layers=2)
    if name == "fam_qh_lat":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True)
    if name == "fam_qh_lat3":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True, q_layers=3)
    if name == "fam_qh_lat4":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True, q_layers=4)
    if name == "fam_qh8":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True, q_qubits=8)
    if name == "fam_qh8_63":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True,
                                    fam_hidden=63, q_qubits=8)
    if name == "fam_qh_lat6_63":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True,
                                    fam_hidden=63, q_layers=6)
    if name == "fam_qh8_60l4":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, latent=True,
                                    fam_hidden=60, q_qubits=8, q_layers=4)
    if name == "fam_qh60lat_rand":
        return FAMQuantumCorrectionRand(input_dim, dim, horizon, qbackend, fam_hidden=60, latent=True)
    if name == "fam_qh60lat_z":
        return FAMQuantumCorrectionZ(input_dim, dim, horizon, qbackend, fam_hidden=60, latent=True)
    if name == "fam_qh60lat":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, fam_hidden=60, latent=True)
    if name == "fam_qh60lat3":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, fam_hidden=60, latent=True, q_layers=3)
    if name == "fam_qh58lat3":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, fam_hidden=58, latent=True, q_layers=3)
    if name == "fam_qh56lat4":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, fam_hidden=56, latent=True, q_layers=4)
    if name == "fam_qh60":
        return FAMQuantumCorrection(input_dim, dim, horizon, qbackend, fam_hidden=60)
    if name == "fam_h60":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, head_hidden=60)
    if name == "qfam_res":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res", qbackend=qbackend)
    if name == "qfam_rand":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_rand", qbackend=qbackend)
    if name == "qfam_bot":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_bot", qbackend=qbackend)
    if name == "qfam_cls":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical_res", qbackend=qbackend)
    if name == "qfam_z":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, measure_x=False)
    if name == "qfam_low":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=4)
    if name == "qfam_gate":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_gate=True)
    if name == "qfam_4q":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, n_qubits=4, q_layers=1)
    if name == "qfam_8q":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, n_qubits=8, q_layers=1)
    if name == "qfam_head":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, qhead="quantum")
    if name == "qfam_head_rand":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, qhead="random")
    if name == "qfam_head_cls":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, qhead="classical")
    if name == "qfam_full":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, qhead="quantum")
    if name == "qfam_phys":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="classical",
                     qbackend=qbackend, qhead="phys")
    if name == "q_dlinear":
        return QDLinear(input_dim, seq_len, horizon, dim, qbackend, "quantum")
    if name == "q_dlinear_rand":
        return QDLinear(input_dim, seq_len, horizon, dim, qbackend, "random")
    if name == "q_dlinear_cls":
        return QDLinear(input_dim, seq_len, horizon, dim, qbackend, "classical")
    if name == "dlinear_mlp":
        return DLinearMLP(input_dim, seq_len, horizon, hidden=32)
    if name == "dlinear_qsum":
        return DLinearQSummary(input_dim, seq_len, horizon, qbackend)
    if name == "q_dlinear_lite":
        return QDLinearLite(input_dim, seq_len, horizon, qbackend)
    if name == "qfam_low2":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=2)
    if name == "qfam_low8":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=8)
    if name == "qfam_low_h32":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=4, head_hidden=32)
    if name == "qfam_low_h48":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=4, head_hidden=48)
    if name == "qfam_low_h60r1":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=1, head_hidden=60)
    if name == "qfam_low_h56":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=4, head_hidden=56)
    if name == "qfam_low_trim":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=4, trim=True)
    if name == "cnn":
        return CNN(input_dim, dim=32, horizon=horizon)
    if name == "lstm_lite":
        return LSTMLite(input_dim, dim=32, horizon=horizon)
    if name == "qlstm":
        return QLSTMLite(input_dim, dim=32, horizon=horizon, qbackend=qbackend)
    if name == "qdlinear_4q":
        return QDLinear4Q(input_dim, seq_len, horizon, qbackend=qbackend)
    if name == "qdlinear_cls":
        return QDLinearClassical(input_dim, seq_len, horizon)
    if name == "qdlinear_top":
        return QDLinearLite(input_dim, seq_len, horizon, qbackend=qbackend,
                            idx=(23, 14, 30, 4, 32, 25))
    if name == "qdlinear_power":
        return QDLinearLite(input_dim, seq_len, horizon, qbackend=qbackend,
                            idx=(23, 4, 7, 3, 16, 17))
    if name == "qdlinear_power_b":
        return QDLinearLite(input_dim, seq_len, horizon, qbackend=qbackend,
                            idx=(23, 4, 7, 3, 16, 17), bias=True)
    if name == "qdlinear_met":
        return QDLinearLite(input_dim, seq_len, horizon, qbackend=qbackend,
                            idx=(23, 4, 3, 7, 12, 18))
    if name == "qdlinear_mix":
        return QDLinearLite(input_dim, seq_len, horizon, qbackend=qbackend,
                            idx=(23, 4, 7, 16, 17, 18))
    if name == "qdlinear_group":
        return QDLinearLiteProj(input_dim, seq_len, horizon, qbackend=qbackend, mode="group")
    if name == "qdlinear_rand":
        return QDLinearLiteProj(input_dim, seq_len, horizon, qbackend=qbackend, mode="rand")
    if name == "qfam_low_z":
        return FAMv2(input_dim, dim, horizon, layers=1, projector="quantum_res",
                     qbackend=qbackend, q_low_rank=4, measure_x=False)
    raise ValueError(name)
