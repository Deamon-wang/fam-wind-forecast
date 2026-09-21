"""
Data preprocessing module for winf (wind farm) dataset format.

Converts raw WINF CSV data with the following columns:
    Time, lon, lat, u100, v100, u10, v10, d_temp, temp, prea, rian, obs_true, power_true

into the standard format expected by the existing feature engineering pipeline:
    date, temp_2m, rh_2m, dew_2m, ws10, ws100, wd10, wd100, p, rain, power, obs_lag_*

Key conversions:
    - u/v wind components  →  wind speed & direction (ws, wd)
    - temperature (K)      →  temp_2m (°C)
    - dew point (K)        →  dew_2m (°C)
    - pressure (Pa)        →  p (hPa)
    - Magnus formula       →  rh_2m (%) from temp & dew point
    - obs_true             →  lagged features only (future values NOT leaked)
    - power_true           →  power (target)

Usage:
    from src.data_processing_winf import preprocess_winf, build_feature_cols_winf
    from src.data_processing import add_time_features, encode_wd, add_power_features

    df = preprocess_winf("dataset/winf_fam1.csv")
    df = add_time_features(df)
    df = encode_wd(df)
    df = add_power_features(df)
    feature_cols = build_feature_cols_winf(df)
"""

import numpy as np
import pandas as pd
import logging

from src.data_processing import normalize_freq

logger = logging.getLogger(__name__)


# =========================
# 1) Low-level helpers
# =========================

def load_winf(csv_path: str) -> pd.DataFrame:
    """
    Load a winf-format CSV and perform basic column renaming.

    Input columns expected:
        Time, lon, lat, u100, v100, u10, v10,
        d_temp, temp, prea, rian, obs_true, power_true

    Returns DataFrame with:
        date, u100, v100, u10, v10, d_temp, temp, prea, rian, obs_true, power_true
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Rename Time -> date for datetime parsing
    if "Time" in df.columns:
        df = df.rename(columns={"Time": "date"})

    # Parse datetime (format: "2021/1/1 8:00")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop constant location columns (single wind farm)
    for col in ["lon", "lat"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Rename target column for downstream compatibility
    if "power_true" in df.columns:
        df = df.rename(columns={"power_true": "power"})

    return df


def uv_to_wind(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert u/v wind vector components to wind speed (ws) and
    meteorological wind direction (wd).

    Wind speed:  ws = sqrt(u^2 + v^2)
    Wind direction: wd = arctan2(v, u) converted to degrees [0, 360)

    Processes both 10m and 100m levels.
    Drops the original u/v component columns after conversion.
    """
    df = df.copy()

    for level in ["10", "100"]:
        u_col = f"u{level}"
        v_col = f"v{level}"
        ws_col = f"ws{level}"
        wd_col = f"wd{level}"

        if u_col in df.columns and v_col in df.columns:
            u = df[u_col].values.astype(float)
            v = df[v_col].values.astype(float)

            df[ws_col] = np.sqrt(u**2 + v**2)
            # arctan2(v, u): meteorological convention, 0° = wind FROM north
            # Standard: wd = atan2(u, v) for "wind FROM" direction, but
            # here we use atan2(v, u) which gives "wind TO" direction.
            # For wind power, consistency matters more than convention.
            wd_rad = np.arctan2(v, u)
            wd_deg = np.degrees(wd_rad)
            df[wd_col] = wd_deg % 360.0

            # Drop raw components
            df = df.drop(columns=[u_col, v_col])

    return df


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert meteorological variables to units expected by downstream pipeline.

    temp (K)   → temp_2m (°C)   :  subtract 273.15
    d_temp (K) → dew_2m (°C)    :  subtract 273.15
    prea (Pa)  → p (hPa / mbar) :  divide by 100
    rian       → rain           :  rename only
    """
    df = df.copy()

    if "temp" in df.columns:
        df["temp_2m"] = df["temp"].astype(float) - 273.15
        df = df.drop(columns=["temp"])

    if "d_temp" in df.columns:
        df["dew_2m"] = df["d_temp"].astype(float) - 273.15
        df = df.drop(columns=["d_temp"])

    if "prea" in df.columns:
        df["p"] = df["prea"].astype(float) / 100.0  # Pa → hPa
        df = df.drop(columns=["prea"])

    if "rian" in df.columns:
        df = df.rename(columns={"rian": "rain"})

    return df


def compute_rh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative humidity (%) from temperature and dew point
    using the Magnus formula (August-Roche-Magnus approximation).

        RH = 100 * exp(a * Td / (b + Td)) / exp(a * T / (b + T))

    where:
        T  = temperature (°C)
        Td = dew point (°C)
        a  = 17.625
        b  = 243.04 (°C)

    Output rh_2m is clamped to [0, 100].
    """
    df = df.copy()

    if "temp_2m" not in df.columns or "dew_2m" not in df.columns:
        return df

    T = df["temp_2m"].values.astype(float)
    Td = df["dew_2m"].values.astype(float)

    a = 17.625
    b = 243.04  # °C

    # Compute saturation vapor pressure ratio via Magnus formula
    # Use np.clip on inputs to avoid overflow in exp
    # For typical Earth-surface temps (-40 to +50 °C), this is well-behaved
    num = np.exp(a * Td / (b + Td))
    den = np.exp(a * T / (b + T))

    rh = 100.0 * num / den
    rh = np.clip(rh, 0.0, 100.0) 

    df["rh_2m"] = rh
    return df


def clean_winf(
    df: pd.DataFrame,
    freq: str = "1hour",
    power_clip_high: float | None = None,
    ffill_limit: int = 6,
) -> pd.DataFrame:
    """
    Clean and align winf data after unit conversion.

    Mirrors align_and_clean() from data_processing.py but with adjusted
    thresholds suitable for the winf dataset.

    Steps:
        1. Sort by date, convert all non-date columns to numeric
        2. Handle duplicate timestamps (mean aggregation)
        3. Reindex to regular frequency, marking gaps as NaN
        4. Clip columns to physically plausible ranges
        5. Forward-fill exogenous features (limited), fill rain with 0
        6. Reset index (date back to column)

    Parameters
    ----------
    power_clip_high : float or None
        Upper bound for power values (MW). When None (default), automatically
        determined as max(power) * 1.15 to adapt to different wind farm capacities.
    """
    df = df.copy()

    # --- 1) Sort & ensure numeric ---
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in df.columns:
        if c == "date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 2) Deduplicate timestamps ---
    dup_cnt = df["date"].duplicated().sum()
    if dup_cnt > 0:
        logger.warning(
            f"Found {dup_cnt} duplicate timestamps. Aggregating with mean."
        )
        df = df.groupby("date", as_index=False).mean(numeric_only=True)

    # --- 3) Reindex to regular frequency ---
    df = df.set_index("date").sort_index()
    full_index = pd.date_range(
        df.index.min(), df.index.max(), freq=normalize_freq(freq)
    )
    df = df.reindex(full_index)
    df.index.name = "date"

    # --- 4) Clip to physically plausible ranges ---
    def _clip(s, low=None, high=None):
        if low is not None:
            s = s.where(s >= low, np.nan)
        if high is not None:
            s = s.where(s <= high, np.nan)
        return s

    # Wind speed
    if "ws10" in df.columns:
        df["ws10"] = _clip(df["ws10"], low=0.0, high=60.0)
    if "ws100" in df.columns:
        df["ws100"] = _clip(df["ws100"], low=0.0, high=70.0)

    # Wind direction
    for wd_col in ["wd10", "wd100"]:
        if wd_col in df.columns:
            df[wd_col] = _clip(df[wd_col], low=0.0, high=360.0)

    # Temperature / humidity
    if "temp_2m" in df.columns:
        df["temp_2m"] = _clip(df["temp_2m"], low=-60.0, high=60.0)
    if "dew_2m" in df.columns:
        df["dew_2m"] = _clip(df["dew_2m"], low=-60.0, high=60.0)
    if "rh_2m" in df.columns:
        df["rh_2m"] = _clip(df["rh_2m"], low=0.0, high=100.0)

    # Pressure (hPa): wide range to accommodate diverse weather
    if "p" in df.columns:
        df["p"] = _clip(df["p"], low=500.0, high=1100.0)

    # Precipitation
    if "rain" in df.columns:
        df["rain"] = _clip(df["rain"], low=0.0)  # non-negative

    # Observed wind speed (hub height)
    if "obs_true" in df.columns:
        df["obs_true"] = _clip(df["obs_true"], low=0.0, high=80.0)

    # Power: auto-detect upper bound when not specified
    if "power" in df.columns:
        if power_clip_high is None:
            power_max = df["power"].max(skipna=True)
            power_clip_high = float(np.ceil(power_max * 1.15 / 5) * 5)  # round up to nearest 5 MW
            logger.info(f"  Auto power_clip_high: {power_clip_high:.0f} MW (data max={power_max:.1f} MW)")
        df["power"] = _clip(df["power"], low=0.0, high=power_clip_high)

    # --- 5) Fill gaps ---
    # Forward-fill exogenous columns (everything except power and rain)
    exog_ffill_cols = [
        c for c in df.columns if c not in ["power", "rain"]
    ]
    for c in exog_ffill_cols:
        df[c] = df[c].ffill(limit=ffill_limit)

    # Precipitation NaN → 0
    if "rain" in df.columns:
        df["rain"] = df["rain"].fillna(0.0)

    # --- 6) Reset index ---
    df = df.reset_index()
    return df


def detect_stuck_sensor(
    df: pd.DataFrame,
    min_hours: int = 6,
    nwp_std_threshold: float = 2.0,
    eps: float = 0.01,
) -> pd.DataFrame:
    """
    Detect and NaN out obs_true values where the anemometer sensor appears stuck.

    A sensor is flagged as "stuck" when ALL of the following hold for a
    consecutive run of >= min_hours:

    1. obs_true is nearly constant (absolute pairwise diff < eps) across the run
    2. Meanwhile, NWP ws100 varies significantly (std > nwp_std_threshold),
       confirming that real wind fluctuations exist during the same period

    This catches faults such as the May 2021 event where obs_true was stuck at
    2.3 m/s for 34 hours while ws100 ranged 7–11 m/s.

    Parameters
    ----------
    df : DataFrame, must have 'obs_true' and 'ws100' columns, sorted by date
    min_hours : int, minimum consecutive hours to trigger detection
    nwp_std_threshold : float, minimum std of ws100 (m/s) over the run to
        confirm real wind variability
    eps : float, tolerance for "constant" obs_true (m/s)

    Returns
    -------
    DataFrame with stuck obs_true values replaced by NaN
    """
    if "obs_true" not in df.columns or "ws100" not in df.columns:
        return df

    df = df.copy()
    obs = df["obs_true"].values
    nwp = df["ws100"].values
    n = len(df)

    # Find runs of constant obs_true
    stuck = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if np.isnan(obs[i]):
            i += 1
            continue
        # Grow forward while obs_true stays within eps of obs[i]
        j = i + 1
        while j < n and not np.isnan(obs[j]) and abs(obs[j] - obs[i]) < eps:
            j += 1
        run_len = j - i
        if run_len >= min_hours:
            nwp_std = np.nanstd(nwp[i:j])
            if nwp_std > nwp_std_threshold:
                stuck[i:j] = True
                logger.warning(
                    f"Sensor stuck detected: {run_len}h from "
                    f"{df['date'].iloc[i]} to {df['date'].iloc[j-1]}, "
                    f"obs_true≈{obs[i]:.2f} m/s, ws100_std={nwp_std:.2f} m/s"
                )
        i = j

    if stuck.any():
        n_stuck = stuck.sum()
        df.loc[stuck, "obs_true"] = np.nan
        logger.info(f"  Flagged {n_stuck} stuck-sensor samples as NaN ({n_stuck / n * 100:.1f}%)")

    return df


def add_obs_lag_features(
    df: pd.DataFrame,
    lags: tuple = (1, 2, 3, 6, 12, 24),
) -> pd.DataFrame:
    """
    Create lagged features from obs_true (observed hub-height wind speed).

    IMPORTANT: Only historical lags are created. obs_true at time t is NOT
    used as a future input — it is an observation that is unavailable at
    prediction time. The original obs_true column is DROPPED after lag
    creation to prevent accidental data leakage.

    Parameters
    ----------
    df : DataFrame with an 'obs_true' column
    lags : tuple of int, lag offsets (in time steps)

    Returns
    -------
    DataFrame with obs_lag_{k} columns added and obs_true removed
    """
    df = df.copy()

    if "obs_true" not in df.columns:
        return df

    for lag in lags:
        df[f"obs_lag_{lag}"] = df["obs_true"].shift(lag)

    # Drop the original obs_true — it is a future measurement,
    # not available at prediction time
    df = df.drop(columns=["obs_true"])

    return df


# =========================
# 2) Main entry points
# =========================

def preprocess_winf(
    csv_path: str,
    freq: str = "1hour",
    use_obs_lag: bool = True,
    obs_lags: tuple = (1, 2, 3, 6, 12, 24),
    ffill_limit: int = 6,
    power_clip_high: float | None = None,
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for winf-format wind farm data.

    Converts raw winf CSV into a clean DataFrame with standard column names
    ready for add_time_features(), encode_wd(), and add_power_features().

    Parameters
    ----------
    csv_path : str
        Path to the winf CSV file (e.g. "dataset/winf_fam1.csv")
    freq : str
        Time-series frequency, e.g. "1hour", "10min"
    use_obs_lag : bool
        Whether to create obs_lag features from obs_true
    obs_lags : tuple of int
        Lag offsets for obs_true (default: 1,2,3,6,12,24)
    ffill_limit : int
        Max consecutive NaN steps to forward-fill
    power_clip_high : float or None
        Upper bound for power values (MW). When None, auto-detected from data.

    Returns
    -------
    pd.DataFrame with columns:
        date, temp_2m, rh_2m, dew_2m, ws10, ws100, wd10, wd100,
        p, rain, power, [obs_lag_1, obs_lag_2, ...]

    Example
    -------
    >>> df = preprocess_winf("dataset/winf_fam1.csv")
    >>> df.columns.tolist()
    ['date', 'temp_2m', 'dew_2m', 'rh_2m', 'ws10', 'ws100',
     'wd10', 'wd100', 'p', 'rain', 'power',
     'obs_lag_1', 'obs_lag_2', 'obs_lag_3', 'obs_lag_6', 'obs_lag_12', 'obs_lag_24']
    """
    logger.info(f"Loading winf data from: {csv_path}")

    # Step 1: Load & basic rename
    df = load_winf(csv_path)
    logger.info(f"  Raw rows: {len(df)}")

    # Step 2: u/v → ws/wd
    df = uv_to_wind(df)

    # Step 3: Unit conversion (K→°C, Pa→hPa)
    df = convert_units(df)

    # Step 4: Compute relative humidity
    df = compute_rh(df)

    # Step 5: Clean & align
    df = clean_winf(df, freq=freq, power_clip_high=power_clip_high,
                    ffill_limit=ffill_limit)
    logger.info(f"  After clean & align: {len(df)} rows")

    # Step 5b: Detect stuck sensor (before lag creation to avoid propagation)
    df = detect_stuck_sensor(df)

    # Step 6: Observed wind speed lag features
    if use_obs_lag and "obs_true" in df.columns:
        df = add_obs_lag_features(df, lags=obs_lags)
        logger.info(f"  Added {len(obs_lags)} obs_lag features")

    logger.info(f"  Final columns ({len(df.columns)}): {df.columns.tolist()}")
    return df


def build_feature_cols_winf(
    df: pd.DataFrame,
    use_obs_lag: bool = True,
    obs_lags: tuple = (1, 2, 3, 6, 12, 24),
) -> list:
    """
    Build the ordered list of feature column names for model input.

    Must be called AFTER add_time_features(), encode_wd(), and
    add_power_features() have been applied to the DataFrame.

    Parameters
    ----------
    df : DataFrame after full feature engineering
    use_obs_lag : bool, whether to include obs_lag features
    obs_lags : tuple, which lag offsets were created

    Returns
    -------
    list of column name strings in order:
        base_exog + obs_lags + time_feats + [power]
    """
    # ---- Base exogenous (NWP + derived) ----
    base_exog = [c for c in [
        "temp_2m", "rh_2m", "dew_2m",
        "ws10", "ws100",
        "wd10_sin", "wd10_cos", "wd100_sin", "wd100_cos",
        "ws10_sq", "ws10_cb", "ws100_sq", "ws100_cb",
        "shear_100_10", "wd_diff",
    ] if c in df.columns]

    # ---- Observed wind speed lags ----
    obs_lag_cols = []
    if use_obs_lag:
        obs_lag_cols = [
            f"obs_lag_{lag}"
            for lag in obs_lags
            if f"obs_lag_{lag}" in df.columns
        ]

    # ---- Cyclical time features ----
    time_feats = [c for c in [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "doy_sin", "doy_cos",
    ] if c in df.columns]

    # ---- Target sequence feature (for autoregressive / persistence) ----
    seq_power_feats = [c for c in ["power"] if c in df.columns]

    feature_cols = base_exog + obs_lag_cols + time_feats + seq_power_feats

    logger.info(
        f"Built {len(feature_cols)} feature columns: "
        f"base_exog={len(base_exog)}, obs_lag={len(obs_lag_cols)}, "
        f"time={len(time_feats)}, power={len(seq_power_feats)}"
    )

    return feature_cols
