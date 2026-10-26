"""Data pipeline for the Location1-aligned wind-power experiments.

The aligned files have one row per available hour and may contain large gaps.
Windows are therefore built inside contiguous hourly segments; a window never
crosses a gap.  Chronological train/val/test splits are assigned by the target
timestamp, so no future information enters training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

BASE_COLS = [
    "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
    "windspeed_10m", "windspeed_100m",
    "winddirection_10m", "winddirection_100m",
]


@dataclass
class DataConfig:
    csv_path: str
    seq_len: int = 96
    horizon: int = 1
    train_frac: float = 0.70
    val_frac: float = 0.15
    stride: int = 1
    add_power_lags: bool = True
    drop_features: tuple = ()


def _segment_id(t: pd.Series) -> pd.Series:
    gaps = t.diff().dt.total_seconds().fillna(3600) > 3600.0 + 1e-6
    return gaps.cumsum()


def load_frame(cfg: DataConfig):
    df = pd.read_csv(cfg.csv_path)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    df = df.drop_duplicates("Time", keep="last").reset_index(drop=True)

    # time features
    dt = df["Time"]
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)

    # wind direction encoding
    for c in ["winddirection_10m", "winddirection_100m"]:
        rad = np.deg2rad(df[c].astype(float) % 360.0)
        df[c + "_sin"] = np.sin(rad)
        df[c + "_cos"] = np.cos(rad)
    df["wd_diff"] = np.minimum(
        (df["winddirection_100m"] - df["winddirection_10m"]).abs(),
        360.0 - (df["winddirection_100m"] - df["winddirection_10m"]).abs(),
    )

    # physical transforms
    df["ws10_sq"] = df["windspeed_10m"] ** 2
    df["ws10_cb"] = df["windspeed_10m"] ** 3
    df["ws100_sq"] = df["windspeed_100m"] ** 2
    df["ws100_cb"] = df["windspeed_100m"] ** 3
    df["shear_100_10"] = df["windspeed_100m"] / (df["windspeed_10m"] + 1e-3)

    # contiguous segment id (computed before lags so lags stay inside a segment)
    df["segment"] = _segment_id(df["Time"])

    if cfg.add_power_lags:
        g = df.groupby("segment", sort=False)
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f"power_lag{lag}"] = g["Power"].shift(lag)
        for win in [6, 24]:
            df[f"power_roll{win}_mean"] = g["Power"].transform(
                lambda s: s.shift(1).rolling(win, min_periods=max(2, win // 2)).mean())
            df[f"power_roll{win}_std"] = g["Power"].transform(
                lambda s: s.shift(1).rolling(win, min_periods=max(2, win // 2)).std())
        for win in [6, 24]:
            df[f"ws100_roll{win}_mean"] = g["windspeed_100m"].transform(
                lambda s: s.shift(1).rolling(win, min_periods=max(2, win // 2)).mean())

    feature_cols = [
        *BASE_COLS,
        "winddirection_10m_sin", "winddirection_10m_cos",
        "winddirection_100m_sin", "winddirection_100m_cos",
        "wd_diff", "ws10_sq", "ws10_cb", "ws100_sq", "ws100_cb",
        "shear_100_10",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "Power",
    ]
    if cfg.add_power_lags:
        feature_cols += [c for c in df.columns if c.startswith("power_lag")
                         or c.startswith("power_roll") or c.startswith("ws100_roll")]
    if cfg.drop_features:
        drop = set(cfg.drop_features)
        feature_cols = [c for c in feature_cols if c not in drop]
        if "Power" not in feature_cols:
            raise ValueError("cannot drop the target Power feature")
    return df, feature_cols


class WindowDataset(Dataset):
    def __init__(self, X, y, meta):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.meta = meta

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], i


def build_windows(df: pd.DataFrame, feature_cols, cfg: DataConfig):
    """Return scaled arrays and metadata; split by target time."""
    L, H = cfg.seq_len, cfg.horizon
    need = list(dict.fromkeys(["Time", "Power", *feature_cols]))
    d = df.dropna(subset=need).sort_values("Time").reset_index(drop=True)

    # split boundaries by target timestamp (chronological, global)
    q_train = d["Time"].quantile(cfg.train_frac)
    q_val = d["Time"].quantile(cfg.train_frac + cfg.val_frac)

    Xs, ys, meta = [], [], []
    for seg, g in d.groupby("segment", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < L + H:
            continue
        for s in range(0, n - L - H + 1, cfg.stride):
            e = s + L
            tgt = e + H - 1
            origin_t = g.loc[e - 1, "Time"]
            target_t = g.loc[tgt, "Time"]
            if target_t <= q_train:
                split = "train"
            elif target_t <= q_val:
                split = "val"
            else:
                split = "test"
            Xs.append(g.loc[s:e - 1, feature_cols].to_numpy(np.float32))
            ys.append(float(g.loc[tgt, "Power"]))
            meta.append((split, str(origin_t), str(target_t), seg,
                         float(g.loc[e - 1, "Power"])))

    X = np.stack(Xs)
    y = np.asarray(ys, np.float32)
    meta = pd.DataFrame(meta, columns=["split", "origin", "target", "segment", "last_power"])

    tr = meta["split"].values == "train"
    va = meta["split"].values == "val"
    te = meta["split"].values == "test"

    # scale features on training rows only
    flat = X[tr].reshape(-1, X.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd
    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    if ysd < 1e-8:
        ysd = 1.0
    yz = (y - ymu) / ysd

    return {
        "X": X.astype(np.float32), "y": y.astype(np.float32), "yz": yz.astype(np.float32),
        "meta": meta, "train": tr, "val": va, "test": te,
        "x_mu": mu, "x_sd": sd, "y_mu": ymu, "y_sd": ysd,
        "feature_cols": feature_cols,
    }



def load_combined(paths, cfg: DataConfig):
    """Load several aligned files as sites; windows never cross a site."""
    frames = []
    cols = None
    for p in paths:
        c = DataConfig(p, seq_len=cfg.seq_len, horizon=cfg.horizon,
                       add_power_lags=cfg.add_power_lags,
                       drop_features=cfg.drop_features)
        df, fc = load_frame(c)
        site = os.path.splitext(os.path.basename(p))[0]
        df["site"] = site
        df["segment"] = site + ":" + df["segment"].astype(str)
        frames.append(df)
        cols = fc
    return pd.concat(frames, ignore_index=True), cols


def external_windows(df, feature_cols, cfg, mu, sd, ymu, ysd):
    """Build all windows of an external site and apply training scalers."""
    L, H = cfg.seq_len, cfg.horizon
    need = list(dict.fromkeys(["Time", "Power", *feature_cols]))
    d = df.dropna(subset=need).sort_values("Time").reset_index(drop=True)
    Xs, ys, meta = [], [], []
    for seg, g in d.groupby("segment", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < L + H:
            continue
        for s in range(0, len(g) - L - H + 1, cfg.stride):
            e = s + L; tgt = e + H - 1
            Xs.append(g.loc[s:e - 1, feature_cols].to_numpy(np.float32))
            ys.append(float(g.loc[tgt, "Power"]))
            meta.append(("test", str(g.loc[e - 1, "Time"]), str(g.loc[tgt, "Time"]),
                         int(seg) if str(seg).isdigit() else seg,
                         float(g.loc[e - 1, "Power"])))
    X = (np.stack(Xs) - mu) / sd
    y = np.asarray(ys, np.float32)
    meta = pd.DataFrame(meta, columns=["split", "origin", "target", "segment", "last_power"])
    return X.astype(np.float32), ((y - ymu) / ysd).astype(np.float32), y, meta


def make_datasets_external(train_paths, external_path, cfg: DataConfig,
                           train_frac=0.85, val_frac=0.15):
    """Train/validate on Location1-4; test on an external file."""
    df, feature_cols = load_combined(train_paths, cfg)
    base = build_windows(df, feature_cols,
                         DataConfig("", seq_len=cfg.seq_len, horizon=cfg.horizon,
                                    train_frac=train_frac, val_frac=val_frac,
                                    stride=cfg.stride, add_power_lags=cfg.add_power_lags,
                                    drop_features=cfg.drop_features))
    mu, sd = base["x_mu"], base["x_sd"]
    ymu, ysd = base["y_mu"], base["y_sd"]
    edf, _ = load_frame(DataConfig(external_path, seq_len=cfg.seq_len,
                                   horizon=cfg.horizon, add_power_lags=cfg.add_power_lags,
                                   drop_features=cfg.drop_features))
    edf["segment"] = "ext:" + edf["segment"].astype(str)
    eX, eyz, ey, emeta = external_windows(edf, feature_cols, cfg, mu, sd, ymu, ysd)
    # append external to the same data dict; scalers stay from training
    base["external"] = {"X": eX, "yz": eyz, "y": ey, "meta": emeta}
    tr = WindowDataset(base["X"][base["train"]], base["yz"][base["train"]],
                       base["meta"][base["train"]].reset_index(drop=True))
    va = WindowDataset(base["X"][base["val"]], base["yz"][base["val"]],
                       base["meta"][base["val"]].reset_index(drop=True))
    te = WindowDataset(eX, eyz, emeta.reset_index(drop=True))
    data = dict(base)
    data["external_meta"] = emeta
    data["external_y"] = ey
    return tr, va, te, data


def make_datasets(cfg: DataConfig):
    df, feature_cols = load_frame(cfg)
    data = build_windows(df, feature_cols, cfg)
    tr = WindowDataset(data["X"][data["train"]], data["yz"][data["train"]],
                       data["meta"][data["train"]].reset_index(drop=True))
    va = WindowDataset(data["X"][data["val"]], data["yz"][data["val"]],
                       data["meta"][data["val"]].reset_index(drop=True))
    te = WindowDataset(data["X"][data["test"]], data["yz"][data["test"]],
                       data["meta"][data["test"]].reset_index(drop=True))
    return tr, va, te, data
