"""Replay the QFAM correction with PyVQNet and pyqpanda3.

The script requires a trained QFAM checkpoint and a copy of the external
Huaneng file (not distributed). It compares the torch state vector, PyVQNet
and pyqpanda3 readouts and writes the forecast metrics to JSON.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from qfam_v2.data import DataConfig, make_datasets_external
from qfam_v2.models import PyVQNetCircuit, build_model
from qfam_v2.qpanda3_circuit import expvals
from qfam_v2.train import metrics


def load_model(checkpoint, device):
    model = build_model("fam_qh60lat", 36, 96, 1, 64, qbackend="torch").to(device)
    state = torch.load(checkpoint, map_location=device)
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if key in state:
                value.copy_(state[key].to(value.device))
    model.eval()
    return model


def pyvqnet_q(model, z):
    circuit = PyVQNetCircuit(6, 2)
    params = model.qhead.q.paras.detach().cpu()
    for layer in range(2):
        for qubit in range(6):
            getattr(circuit, f"rx_{layer}_{qubit}").params.data.copy_(
                params[layer, qubit, 0].reshape(1, 1))
            getattr(circuit, f"ry_{layer}_{qubit}").params.data.copy_(
                params[layer, qubit, 1].reshape(1, 1))
            getattr(circuit, f"rz_{layer}_{qubit}").params.data.copy_(
                params[layer, qubit, 2].reshape(1, 1))
    out = []
    with torch.no_grad():
        for start in range(0, len(z), 64):
            out.append(circuit(z[start:start + 64] * np.pi).detach().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="dataset/Location1.csv")
    parser.add_argument("--external_csv", default="dataset/winf_fam1.csv")
    parser.add_argument("--checkpoint", required=True,
                        help="best_weights.pt from a QFAM run")
    parser.add_argument("--out", default="results/sim_backends.json")
    parser.add_argument("--max_qpanda3", type=int, default=0,
                        help="0 means all test windows")
    args = parser.parse_args()
    if not os.path.exists(args.external_csv):
        raise SystemExit(
            f"External file not found: {args.external_csv}. It is proprietary and "
            "must be obtained from the data provider; it is not distributed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test, data = make_datasets_external(
        [args.train_csv], args.external_csv,
        DataConfig("", seq_len=96, horizon=1, stride=4))
    model = load_model(args.checkpoint, device)

    loader = DataLoader(test, batch_size=64, shuffle=False)
    targets, fam_out, features, q_torch = [], [], [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            targets.append(yb.numpy())
            fam_out.append(model.fam(xb).detach().cpu().numpy())
            pooled = torch.cat([xb.mean(1), xb.std(1), xb[:, -1]], dim=-1)
            feat = pooled[:, list(model.idx)]
            features.append(feat.detach().cpu().numpy())
            q_torch.append(model.qhead.q(torch.tanh(feat) * np.pi)
                           .detach().cpu().numpy())
    y = np.concatenate(targets).reshape(-1)
    fam = np.concatenate(fam_out).reshape(-1)
    feat = np.concatenate(features, 0)
    q_torch = np.concatenate(q_torch, 0)
    z = np.tanh(feat)
    q_pv = pyvqnet_q(model, z)

    n_qp = len(z) if args.max_qpanda3 <= 0 else min(args.max_qpanda3, len(z))
    q_qp = np.asarray([
        expvals(z[i] * np.pi, model.qhead.q.paras.detach().cpu().numpy())
        for i in range(n_qp)
    ])

    post = model.qhead.post.cpu()
    skip_layer = model.skip.cpu()
    with torch.no_grad():
        skip = skip_layer(torch.tensor(feat, dtype=torch.float32)).numpy().reshape(-1)

        def correction(q):
            qq = torch.tensor(q, dtype=torch.float32)
            zz = torch.tensor(z[:len(q)], dtype=torch.float32)
            return post(torch.cat([qq, zz], -1)).numpy().reshape(-1)

        y_torch = fam + skip + correction(q_torch)
        y_pv = fam + skip + correction(q_pv)
        y_qp = fam[:n_qp] + skip[:n_qp] + correction(q_qp)

    ymu, ysd = data["y_mu"], data["y_sd"]
    y = y * ysd + ymu
    y_torch = y_torch * ysd + ymu
    y_pv = y_pv * ysd + ymu
    y_qp = y_qp * ysd + ymu
    last = data["external_meta"]["last_power"].to_numpy(float)
    cap = 50.0

    result = {
        "n_test": int(len(y)),
        "parity_torch_pyvqnet": float(np.abs(q_torch - q_pv).max()),
        "parity_torch_qpanda3": float(np.abs(q_torch[:n_qp] - q_qp).max()),
        "n_qpanda3": int(n_qp),
        "metrics_torch": metrics(y, y_torch, cap, last),
        "metrics_pyvqnet": metrics(y, y_pv, cap, last),
        "metrics_qpanda3": metrics(y[:n_qp], y_qp, cap, last[:n_qp]),
        "metrics_torch_subset": metrics(y[:n_qp], y_torch[:n_qp], cap, last[:n_qp]),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
