import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset


def load_and_rename(csv_path: str):
    df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "date": "date",
        "Time": "date",
        "temperature_2m": "temp_2m",
        "relativehumidity_2m": "rh_2m",
        "dewpoint_2m": "dew_2m",
        "windspeed_10m": "ws10",
        "windspeed_100m": "ws100",
        "winddirection_10m": "wd10",
        "winddirection_100m": "wd100",
        "windgusts_10m": "gust10",
        "Power": "power",
    }
    for col in list(df.columns):
        if col in rename_map:
            continue
        c = col
        if c.startswith("max. wv"):
            rename_map[c] = "wv_max"
        elif c.startswith("wv"):
            rename_map[c] = "wv"
        elif c.startswith("wd"):
            rename_map[c] = "wd"
        elif "p (mbar)" in c:
            rename_map[c] = "p"
        elif c.startswith("T ("):
            rename_map[c] = "T"

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def normalize_freq(freq: str):
    f = str(freq).strip()
    f = f.replace("hour", "h").replace("Hour", "h").replace("HOUR", "h")
    f = f.replace("min", "min").replace("MIN", "min")
    f = f.replace(" ", "")
    return f


def align_and_clean(df: pd.DataFrame, freq="10min"):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in df.columns:
        if c == "date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dup_cnt = df["date"].duplicated().sum()
    if dup_cnt > 0:
        logging.warning(f"found duplicate timestamps: {dup_cnt} rows duplicated by 'date'. Aggregating...")
        df = df.groupby("date", as_index=False).mean(numeric_only=True)

    df = df.set_index("date").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=normalize_freq(freq))
    df = df.reindex(full_index)
    df.index.name = "date"

    def clip_to_nan(s, low=None, high=None):
        if low is not None:
            s = s.where(s >= low, np.nan)
        if high is not None:
            s = s.where(s <= high, np.nan)
        return s

    
    if "wv_max" in df.columns:
        df["wv_max"] = clip_to_nan(df["wv_max"], low=0.0, high=80.0)
    if "wd" in df.columns:
        df["wd"] = clip_to_nan(df["wd"], low=0.0, high=360.0)
    if "wd10" in df.columns:
        df["wd10"] = clip_to_nan(df["wd10"], low=0.0, high=360.0)
    if "wd100" in df.columns:
        df["wd100"] = clip_to_nan(df["wd100"], low=0.0, high=360.0)
    if "rh" in df.columns:
        df["rh"] = clip_to_nan(df["rh"], low=0.0, high=100.0)
    if "rh_2m" in df.columns:
        df["rh_2m"] = clip_to_nan(df["rh_2m"], low=0.0, high=100.0)
    if "p" in df.columns:
        df["p"] = clip_to_nan(df["p"], low=850.0, high=1100.0)
    if "T" in df.columns:
        df["T"] = clip_to_nan(df["T"], low=-60.0, high=60.0)
    if "temp_2m" in df.columns:
        df["temp_2m"] = clip_to_nan(df["temp_2m"], low=-60.0, high=60.0)
    if "dew_2m" in df.columns:
        df["dew_2m"] = clip_to_nan(df["dew_2m"], low=-60.0, high=60.0)
    if "ws10" in df.columns:
        df["ws10"] = clip_to_nan(df["ws10"], low=0.0, high=60.0)
    if "ws100" in df.columns:
        df["ws100"] = clip_to_nan(df["ws100"], low=0.0, high=70.0)
    if "gust10" in df.columns:
        df["gust10"] = clip_to_nan(df["gust10"], low=0.0, high=80.0)
    if "power" in df.columns:
        df["power"] = clip_to_nan(df["power"], low=0.0, high=20.0)
     

    # 只对外生连续特征前向填充，不对 power 前填
    exog_ffill_cols = [c for c in df.columns if c not in ["power", "rain", "raining"]]
    for c in exog_ffill_cols:
        df[c] = df[c].ffill(limit=6)

    # 降水类缺失补 0
    for c in ["rain", "raining"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    df = df.reset_index()
    return df


def add_time_features(df: pd.DataFrame):
    dt = pd.to_datetime(df["date"])
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["dow"] = dt.dt.dayofweek
    df["doy"] = dt.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.0)
    return df


def encode_wd(df: pd.DataFrame):
    if "wd" in df.columns:
        wd = df["wd"].astype(float) % 360.0
        wd_rad = np.deg2rad(wd)
        df["wd_sin"] = np.sin(wd_rad)
        df["wd_cos"] = np.cos(wd_rad)
    if "wd10" in df.columns:
        wd10 = df["wd10"].astype(float) % 360.0
        wd10_rad = np.deg2rad(wd10)
        df["wd10_sin"] = np.sin(wd10_rad)
        df["wd10_cos"] = np.cos(wd10_rad)
    if "wd100" in df.columns:
        wd100 = df["wd100"].astype(float) % 360.0
        wd100_rad = np.deg2rad(wd100)
        df["wd100_sin"] = np.sin(wd100_rad)
        df["wd100_cos"] = np.cos(wd100_rad)
    return df

def add_power_features(df: pd.DataFrame):
    df = df.copy()
    eps = 1e-3
    if "ws10" in df.columns:
        df["ws10_sq"] = df["ws10"] ** 2
        df["ws10_cb"] = df["ws10"] ** 3
    if "ws100" in df.columns:
        df["ws100_sq"] = df["ws100"] ** 2
        df["ws100_cb"] = df["ws100"] ** 3
    if "ws10" in df.columns and "ws100" in df.columns:
        df["shear_100_10"] = df["ws100"] / (df["ws10"] + eps)
    if "gust10" in df.columns and "ws10" in df.columns:
        df["gust_factor"] = df["gust10"] / (df["ws10"] + eps)
    if "wd10" in df.columns and "wd100" in df.columns:
        diff = (df["wd100"] - df["wd10"]).abs()
        df["wd_diff"] = np.minimum(diff, 360.0 - diff)
    return df

def add_lag_rolling_features(
    df: pd.DataFrame,
    target="power",
    lags=(1, 2, 3, 6, 12, 24, 48, 72, 168),
    roll_windows=(3, 6, 12, 24, 48, 72, 168),
    add_exog_lag_for=("ws10", "ws100", "gust10", "temp_2m", "rh_2m"),
):
    df = df.copy()

    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for k in lags:
        df[f"{target}_lag_{k}"] = df[target].shift(k)

    df[f"{target}_diff_0"] = df[target].shift(1) - df[target].shift(2)
    df[f"{target}_diff_1"] = df[target].shift(2) - df[target].shift(3)

    for w in roll_windows:
        s = df[target].shift(1).rolling(window=w, min_periods=max(2, w//3))
        df[f"{target}_roll_mean_{w}"] = s.mean()
        df[f"{target}_roll_std_{w}"] = s.std()
        df[f"{target}_roll_max_{w}"] = s.max()
        df[f"{target}_roll_min_{w}"] = s.min()
        
        
    for c in add_exog_lag_for:
        if c not in df.columns:
            continue
        df[f"{c}_lag_1"] = df[c].shift(1)
    return df

def build_supervised(df: pd.DataFrame, horizon=1, feature_cols=None, target_col="Power"):
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["y"] = df[target_col].shift(-horizon)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    if feature_cols is None:
        drop_cols = {"date", target_col, "y"}
        feature_cols = [c for c in df.columns if c not in drop_cols]

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y, feature_cols, df[["date", "y"]].copy()

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=96, pred_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len
    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SeqForecastDataset(Dataset):
    """
    直接基于原始时间序列窗口构造样本：
    输入: [t-seq_len+1, ..., t]
    输出: [t+horizon, ..., t+horizon+pred_len-1]
    """
    def __init__(self, X, y, seq_len=96, horizon=1, pred_len=1):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.pred_len = pred_len

        self.n = len(self.X)
        self.valid_length = self.n - self.seq_len - self.horizon - self.pred_len + 2
        if self.valid_length <= 0:
            raise ValueError(
                f"序列长度不足: len={self.n}, seq_len={seq_len}, horizon={horizon}, pred_len={pred_len}"
            )

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]

        y_start = idx + self.seq_len - 1 + self.horizon
        y_end = y_start + self.pred_len
        y = self.y[y_start: y_end]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)