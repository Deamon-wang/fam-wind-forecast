"""FAM / QFAM wind-power forecasting models (multi-station generation).

Quick start::

    from qfam import WindQMambaformer

    fam  = WindQMambaformer(proj_mode="linear")   # FAM    (classical linear projection)
    qfam = WindQMambaformer(proj_mode="quantum")  # QFAM   (6-qubit VQC projector)
    ctrl = WindQMambaformer(proj_mode="gelu")     # QFAM_C (parameter-matched GELU control)

Quantum circuits:
    vqc_circuit.vqc_circuit       -- pyvqnet VQC used by QFAM training/inference
    vqc_circuit_chip.Angle_VQC_chip / Angle_VQC_chip_multi -- pyqpanda3 on-chip circuits
"""

from .model_qmambaformer import WindQMambaformer, HybridQuantumProjector, FlowQAttention
from .baselines import (
    WindPowerLSTM,
    WindPowerSimpleTransformer,
    WindPowerInformer,
    WindPowerTimesNet,
)

__all__ = [
    "WindQMambaformer",
    "HybridQuantumProjector",
    "FlowQAttention",
    "WindPowerLSTM",
    "WindPowerSimpleTransformer",
    "WindPowerInformer",
    "WindPowerTimesNet",
]
