"""Inference-time readout controls for a trained QFAM checkpoint.

Replaces the 12 quantum expectations by zero, their test-set mean or a shuffled
vector, without retraining, and reports RMSE. Requires the proprietary external
file; it is not distributed.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from qfam_v2.data import DataConfig, make_datasets_external
from qfam_v2.models import build_model
from qfam_v2.train import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="dataset/Location1.csv")
    parser.add_argument("--external_csv", default="dataset/winf_fam1.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/readout_controls.json")
    args = parser.parse_args()
    if not os.path.exists(args.external_csv):
        raise SystemExit(
            f"External file not found: {args.external_csv}. It is proprietary and "
            "is not distributed with this repository.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test, data = make_datasets_external(
        [args.train_csv], args.external_csv,
        DataConfig("", seq_len=96, horizon=1, stride=4))
    model = build_model("fam_qh60lat", 36, 96, 1, 64, qbackend="torch").to(device)
    state = torch.load(args.checkpoint, map_location=device)
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if key in state:
                value.copy_(state[key].to(value.device))
    model.eval()

    targets, fam_out, features, q_out = [], [], [], []
    with torch.no_grad():
        for xb, yb, _ in DataLoader(test, batch_size=64, shuffle=False):
            xb = xb.to(device)
            targets.append(yb.numpy())
            fam_out.append(model.fam(xb).detach().cpu().numpy())
            pooled = torch.cat([xb.mean(1), xb.std(1), xb[:, -1]], dim=-1)
            feat = pooled[:, list(model.idx)]
            features.append(feat.detach().cpu().numpy())
            q_out.append(model.qhead.q(torch.tanh(feat) * np.pi)
                         .detach().cpu().numpy())
    y = np.concatenate(targets).reshape(-1)
    fam = np.concatenate(fam_out).reshape(-1)
    feat = np.concatenate(features, 0)
    q = np.concatenate(q_out, 0)
    z = np.tanh(feat)
    post = model.qhead.post.cpu()
    skip = model.skip.cpu()
    rng = np.random.default_rng(42)

    with torch.no_grad():
        skip_v = skip(torch.tensor(feat, dtype=torch.float32)).numpy().reshape(-1)

        def predict(qq):
            qq = torch.tensor(qq, dtype=torch.float32)
            zz = torch.tensor(z, dtype=torch.float32)
            return (fam + skip_v
                    + post(torch.cat([qq, zz], -1)).numpy().reshape(-1))

        variants = {
            "trained": predict(q),
            "zero": predict(np.zeros_like(q)),
            "mean": predict(np.tile(q.mean(0), (len(q), 1))),
            "shuffled": predict(q[rng.permutation(len(q))]),
        }

    y_mw = y * data["y_sd"] + data["y_mu"]
    last = data["external_meta"]["last_power"].to_numpy(float)
    result = {name: metrics(y_mw, pred * data["y_sd"] + data["y_mu"], 50.0, last)
              for name, pred in variants.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v["RMSE_MW"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
