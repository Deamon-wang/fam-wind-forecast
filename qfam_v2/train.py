"""Single-run trainer/evaluator for the aligned wind-power benchmarks."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DataConfig, make_datasets, make_datasets_external
from .models import build_model, count_params


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y, p, cap, last=None):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    err = p - y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss if ss > 0 else float("nan")
    out = {
        "RMSE_pu": rmse, "MAE_pu": mae, "R2": r2,
        "RMSE_MW": rmse * cap, "MAE_MW": mae * cap,
        "nRMSE": rmse / max(float(np.mean(y)), 1e-6),
        "n": int(len(y)),
    }
    if last is not None:
        ramp = np.abs(y - np.asarray(last, float))
        thr = np.quantile(ramp, 0.80)
        m = ramp >= thr
        out["ramp_n"] = int(m.sum())
        out["ramp_RMSE_MW"] = float(np.sqrt(np.mean((p[m] - y[m]) ** 2)) * cap)
        out["ramp_MAE_MW"] = float(np.mean(np.abs(p[m] - y[m])) * cap)
        peak = y >= np.quantile(y, 0.80)
        out["peak_RMSE_MW"] = float(np.sqrt(np.mean((p[peak] - y[peak]) ** 2)) * cap)
        low = y <= np.quantile(y, 0.20)
        out["low_RMSE_MW"] = float(np.sqrt(np.mean((p[low] - y[low]) ** 2)) * cap)
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys, idx = [], [], []
    for xb, yb, ib in loader:
        xb = xb.to(device)
        out = model(xb).detach().cpu().numpy()
        preds.append(out)
        ys.append(yb.numpy())
        idx.append(ib.numpy())
    return np.concatenate(preds, 0).reshape(-1), np.concatenate(ys, 0).reshape(-1), np.concatenate(idx, 0)


def run(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drops = tuple(x.strip() for x in args.drop_features.split(",") if x.strip())
    cfg = DataConfig(args.csv_path, seq_len=args.seq_len, horizon=args.horizon,
                     stride=args.stride,
                     add_power_lags=not args.no_power_lags, drop_features=drops)
    if args.external_path:
        paths = [x.strip() for x in args.csv_path.split(",") if x.strip()]
        tr, va, te, data = make_datasets_external(paths, args.external_path, cfg)
    else:
        tr, va, te, data = make_datasets(cfg)

    if args.train_frac < 1.0:
        import random as _random
        _random.seed(args.seed)
        _idx = sorted(_random.sample(range(len(tr)), max(32, int(len(tr) * args.train_frac))))
        from torch.utils.data import Subset
        tr = Subset(tr, _idx)
        print(f"train subset: {len(tr)} windows ({args.train_frac:.2f})", flush=True)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False)
    y_mu, y_sd = data["y_mu"], data["y_sd"]
    cap = 50.0 if ("fam1" in os.path.basename(args.csv_path).lower()
                    or "winf" in os.path.basename(args.external_path).lower()) else 150.0

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    log = open(os.path.join(out_dir, "train.log"), "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    say(f"data train/val/test = {len(tr)}/{len(va)}/{len(te)}  features={len(data['feature_cols'])}")
    say(f"target {data['meta']['target'].min()} .. {data['meta']['target'].max()}")

    if args.model == "persistence":
        p = data["meta"].loc[data["test"], "last_power"].to_numpy(float)
        y = data["y"][data["test"]]
        last = data["meta"].loc[data["test"], "last_power"].to_numpy(float)
        res = {"model": args.model, "seed": args.seed, "horizon": args.horizon,
               "params": 0, "metrics": metrics(y, p, cap, last), "time_s": 0.0}
        json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
        say(json.dumps(res["metrics"], indent=2))
        return res

    model = build_model(args.model, len(data["feature_cols"]), args.seq_len,
                        1, args.dim, qbackend=args.qbackend).to(device)
    if args.init_dlinear:
        sd = torch.load(args.init_dlinear, map_location="cpu")
        with torch.no_grad():
            for name in ("trend", "season"):
                if hasattr(model, name) and f"{name}.weight" in sd:
                    getattr(model, name).weight.copy_(sd[f"{name}.weight"])
        say(f"initialised {args.model} trend/season from {args.init_dlinear}")
    nparam = count_params(model)
    say(f"model={args.model} params={nparam} device={device}")

    q_params = [p for n, p in model.named_parameters() if ".q." in n and p.requires_grad]
    other = [p for n, p in model.named_parameters() if ".q." not in n and p.requires_grad]
    groups = [{"params": other, "lr": args.lr}]
    if q_params:
        groups.append({"params": q_params, "lr": args.lr * args.q_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    if q_params:
        say(f"quantum params={sum(p.numel() for p in q_params)} lr_mult={args.q_lr_mult}")
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=3, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    ramp_w = None
    if args.ramp_weight > 0:
        if hasattr(tr, "meta"):
            last_all = torch.tensor(tr.meta["last_power"].to_numpy(float), device=device)
            y_raw_tr = tr.y.numpy() * y_sd + y_mu
            delta_scale = float(np.median(np.abs(y_raw_tr - tr.meta["last_power"].to_numpy(float)))) + 1e-6
        else:
            raise ValueError("ramp weighting currently requires the full training set")
        say(f"ramp_weight={args.ramp_weight} delta_scale={delta_scale:.4f}")
    best = float("inf"); best_epoch = -1; bad = 0; best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0; nb = 0
        for xb, yb, ib in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            if args.ramp_weight > 0:
                last_b = last_all[ib.to(device)]
                yraw = yb * y_sd + y_mu
                w = 1.0 + args.ramp_weight * (yraw - last_b).abs() / delta_scale
                loss = (w.unsqueeze(-1) * (out - yb.unsqueeze(-1)) ** 2).mean()
            else:
                loss = loss_fn(out, yb.unsqueeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        model.eval(); vtot = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                vtot += float(loss_fn(model(xb), yb.unsqueeze(-1)).item()); vn += 1
        vloss = vtot / max(vn, 1)
        sched.step(vloss)
        say(f"epoch {epoch:02d} train={tot/max(nb,1):.5f} val={vloss:.5f} lr={opt.param_groups[0]['lr']:.2e}")
        if vloss < best - 1e-5:
            best, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if torch.is_tensor(v)}
            torch.save(best_state, os.path.join(out_dir, "best_weights.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                say(f"early stop at {epoch}")
                break

    if best_state is not None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in best_state:
                    v.copy_(best_state[k].to(v.device))
    pvz, yvz, _ = predict(model, val_loader, device)
    pv = pvz * y_sd + y_mu
    yv = yvz * y_sd + y_mu
    meta_val = data["meta"].loc[data["val"]].reset_index(drop=True)
    pd.DataFrame({"target": meta_val["target"], "y": yv, "pred": pv,
                  "last_power": meta_val["last_power"]}).to_csv(
        os.path.join(out_dir, "val_predictions.csv"), index=False)

    pz, yz, idx = predict(model, test_loader, device)
    p = pz * y_sd + y_mu
    y = yz * y_sd + y_mu
    if data.get("external_meta") is not None:
        meta_test = data["external_meta"].reset_index(drop=True)
    else:
        meta_test = data["meta"].loc[data["test"]].reset_index(drop=True)
    last = meta_test["last_power"].to_numpy(float)

    res = {
        "model": args.model, "seed": args.seed, "horizon": args.horizon,
        "params": nparam, "best_epoch": best_epoch, "best_val": best,
        "train_frac": args.train_frac, "qbackend": args.qbackend,
        "time_s": time.time() - t0, "metrics": metrics(y, p, cap, last),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "feature_cols": data["feature_cols"],
    }
    # save raw predictions for paired tests / analysis
    pd.DataFrame({
        "target": meta_test["target"], "origin": meta_test["origin"],
        "y": y, "pred": p, "last_power": last,
    }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    say(json.dumps(res["metrics"], indent=2))
    say(f"done in {res['time_s']:.1f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--qbackend", default="torch", choices=["torch", "pyvqnet"])
    ap.add_argument("--q_lr_mult", type=float, default=1.0)
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--no_power_lags", action="store_true")
    ap.add_argument("--ramp_weight", type=float, default=0.0)
    ap.add_argument("--drop_features", default="")
    ap.add_argument("--init_dlinear", default="")
    ap.add_argument("--external_path", default="")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
