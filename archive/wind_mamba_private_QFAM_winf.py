import os
import math
import json
import random
import argparse
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from src.models import WindPowerTransformer
from src.mamba import WindPowerMambaFlowGRU,WindPowerMambaFlowLiteV5
from src.baseline_models import WindPowerMLP, WindPowerLSTM, WindPowerSimpleTransformer
from src.data_processing import add_time_features,add_power_features,encode_wd
from src.data_processing_winf import preprocess_winf, build_feature_cols_winf

from src.data_processing import SeqForecastDataset
from src.model_qmambaformer import WindQMambaformer
from torch.utils.data import Dataset, DataLoader, Subset
import time
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Seed
try:
    from src.utils import save_forecast_plot
except Exception:
    save_forecast_plot = None

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def filter_state_dict(state_dict):
    """Remove transient VQC state buffers whose shape varies with batch size.
    These are re-initialized by QMachine.reset_states() on every forward pass
    and do not need to be checkpointed.
    """
    return {k: v for k, v in state_dict.items() if ".qm.states" not in k}

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y")
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# =========================
# 1) 指标
# =========================
def safe_mape(y_true, y_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def smape(y_true, y_pred, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

 
# Train
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # print(x.shape)
        # exit()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        # total_loss += loss.item()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# Evaluate

@torch.no_grad()
def evaluate_scaled(model, loader, device):
    model.eval()
    preds = []
    trues = []
    # with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    yp = y_pred.reshape(-1)
    yt = y_true.reshape(-1)

    mse = mean_squared_error(yt, yp)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    mape = safe_mape(yt, yp)
    smape_v = smape(yt, yp)

    return {"MSE": mse,"RMSE": rmse,"MAE": mae,"R2": r2,"MAPE": mape,"SMAPE": smape_v,}, y_pred, y_true

def fallback_save_plot(
    out_df,
    out_dir,
    model_tag,
    ylabel="power",
    max_points=1000,
    plot_mode="first_step",
):
    """
    plot_mode:
        first_step: 只画 pred_step=1 的滚动预测结果，适合连续时间曲线
        first_origin: 只画第一个 forecast origin 的完整多步预测
        mean_by_target: 同一 target_date 的多个预测取平均
    """
    path = os.path.join(out_dir, f"{model_tag}.png")

    if out_df.empty:
        raise ValueError("out_df 为空，无法绘图")

    required_cols = {"power_true", "power_pred"}
    missing_cols = required_cols - set(out_df.columns)

    if missing_cols:
        raise KeyError(
            f"绘图数据缺少必要字段: {sorted(missing_cols)}，"
            f"当前字段为: {list(out_df.columns)}"
        )

    dfp = out_df.copy()

    # 新版输出优先使用 target_date；兼容旧版 date
    if "target_date" in dfp.columns:
        time_col = "target_date"
    elif "date" in dfp.columns:
        time_col = "date"
    else:
        raise KeyError(
            "out_df 中既没有 'target_date'，也没有 'date'，"
            f"当前字段为: {list(dfp.columns)}"
        )

    dfp[time_col] = pd.to_datetime(dfp[time_col])

    if plot_mode == "first_step":
        # 连续滚动预测曲线：每个 forecast origin 只取第一预测步
        if "pred_step" in dfp.columns:
            dfp = dfp[dfp["pred_step"] == 1].copy()

        dfp = (
            dfp.sort_values(time_col)
            .drop_duplicates(subset=[time_col], keep="first")
        )

        title = f"{model_tag} - one-step rolling forecast"

    elif plot_mode == "first_origin":
        # 展示某一个预测起点的全部多步预测
        if "forecast_origin_id" not in dfp.columns:
            raise KeyError(
                "plot_mode='first_origin' 需要 forecast_origin_id 字段"
            )

        first_origin = dfp["forecast_origin_id"].min()

        dfp = dfp[
            dfp["forecast_origin_id"] == first_origin
        ].copy()

        dfp = dfp.sort_values(time_col)
        title = f"{model_tag} - origin {first_origin}"

    elif plot_mode == "mean_by_target":
        # 同一目标时刻可能有多个预测，将它们平均
        dfp = (
            dfp.groupby(time_col, as_index=False)
            .agg(
                power_true=("power_true", "first"),
                power_pred=("power_pred", "mean"),
            )
            .sort_values(time_col)
        )

        title = f"{model_tag} - mean prediction by target time"

    else:
        raise ValueError(f"未知 plot_mode: {plot_mode}")

    dfp = dfp.iloc[:max_points].copy()

    if dfp.empty:
        raise ValueError(
            f"按照 plot_mode={plot_mode} 筛选后没有可绘制数据"
        )

    plt.figure(figsize=(14, 5))
    plt.plot(
        dfp[time_col].values,
        dfp["power_true"].values,
        label="true",
    )
    plt.plot(
        dfp[time_col].values,
        dfp["power_pred"].values,
        label="pred",
    )

    plt.xticks(rotation=30)
    plt.xlabel("target time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    return path

class HorizonWeightedMSE(nn.Module):
    def __init__(self, pred_len, mode="near"):
        super().__init__()
        if mode == "near":
            # 更重视近一步
            w = torch.tensor([1.00, 1.00, 0.95, 0.90, 0.85, 0.80,0.80,0.80,0.75,0.75,0.7,0.7][:pred_len], dtype=torch.float32)
        elif mode == "flat":
            w = torch.ones(pred_len, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        w = w / w.mean()
        self.register_buffer("weights", w)


    def forward(self, pred, target):
        # pred/target: [B, pred_len]
        loss = (pred - target) ** 2
        return (loss * self.weights.view(1, -1)).mean()
# ============================================================
# Main
# ============================================================

def main(csv_path="./dataset/Location1.csv",freq="1hour",horizon=1,seq_len=96,
    pred_len=1,batch_size=32,lr=1e-3,epochs=100,patience=20,optimizer_name="adamw",
    scheduler_name="plateau",d_model=128,n_heads=4,num_layers=3,dropout=0.1,
    train_ratio=0.7,val_ratio=0.15,seed=42,train_mode=True,model_type="wind_mamba",exp_id=None,max_eval_pred_len=12):
    # max_eval_pred_len=12 # 所有实验统一按照最大预测长度裁剪测试起点

    seed_everything(seed)
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
 
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") if train_mode else exp_id
    model_tag = f"{model_type}_h{horizon}_L{seq_len}_P{pred_len}_{dataset_name}_{exp_id}"

    # ---- 输出目录：results_opt/{model_type}/{exp_id}/ ----
    out_dir = os.path.join("results_opt", model_type, exp_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 日志 ----
    log_path = os.path.join(out_dir, f"train_{model_tag}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info("STARTING TRAINING...")
    logger.info(f"Experiment ID: {exp_id}")
    logger.info(f"csv_path={csv_path}, freq={freq}")
    logger.info(f"model_type={model_type}, horizon={horizon}, seq_len={seq_len}, pred_len={pred_len}")
    logger.info(f"epochs={epochs}, patience={patience}, batch_size={batch_size}, lr={lr}")
    logger.info(f"d_model={d_model}, n_heads={n_heads}, num_layers={num_layers}, dropout={dropout}")
    logger.info(f"train_ratio={train_ratio}, val_ratio={val_ratio}, seed={seed}")

    checkpoint_path = os.path.join(out_dir, f"{model_tag}_best.pt")
    # final_model_path = os.path.join(out_dir, f"{model_tag}.pt")

    # ---- 读入 & 清洗对齐 (winf format) ----
    df = preprocess_winf(csv_path, freq=freq, use_obs_lag=True)

    # ---- 特征工程 (reuse existing functions) ----
    df = add_time_features(df)
    df = encode_wd(df)
    df = add_power_features(df)

    feature_cols = build_feature_cols_winf(df, use_obs_lag=True)
    # 只丢弃当前时刻特征缺失，目标不提前 shift
    required_cols = ["date", "power"] + feature_cols
    df = df.dropna(subset=required_cols).sort_values("date").reset_index(drop=True)

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["power"].to_numpy(dtype=np.float32)
    date_all = df["date"].reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    min_required = seq_len + horizon + pred_len - 1
    if n_train < min_required:
        raise ValueError(f"训练集过短: n_train={n_train}, 至少需要 {min_required}")
    if n_val < pred_len:
        raise ValueError(f"验证集过短: n_val={n_val}")
    if n_test < pred_len:
        raise ValueError(f"测试集过短: n_test={n_test}")

    y_meta = df[["date", "power"]].copy()
    meta_test_full = y_meta.iloc[n_train + n_val:].reset_index(drop=True)  # 通常含 date + y_true(shift后)

    # 先按时间切 raw，再为 val/test 补左侧上下文
    train_start, train_end = 0, n_train
    val_start, val_end = n_train, n_train + n_val

    # test_start, test_end = n_train + n_val, n
    test_start, test_end = n_train + n_val, n_train + n_val + n
    # test_start, test_end = n_train + n_val , n_train + n_val +24

    X_train_raw = X_all[train_start:train_end]
    y_train_raw = y_all[train_start:train_end]

    ctx = seq_len + horizon - 1
    val_ctx_start = max(0, val_start - ctx)
    X_val_raw = X_all[val_ctx_start:val_end]
    y_val_raw = y_all[val_ctx_start:val_end]
    date_val_raw = date_all[val_ctx_start:val_end].reset_index(drop=True)
 

    test_ctx_start = max(0, test_start - ctx)
    X_test_raw = X_all[test_ctx_start:test_end]
    y_test_raw = y_all[test_ctx_start:test_end]
    date_test_raw = date_all[test_ctx_start:test_end].reset_index(drop=True)

    # 3) 标准化：只在训练集 fit
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    # 构造窗口数据集
    train_dataset = SeqForecastDataset(X_train, y_train, seq_len=seq_len, horizon=horizon, pred_len=pred_len)
    val_dataset = SeqForecastDataset(X_val, y_val, seq_len=seq_len, horizon=horizon, pred_len=pred_len)
    test_dataset_full = SeqForecastDataset(X_test, y_test, seq_len=seq_len, horizon=horizon, pred_len=pred_len)
    if max_eval_pred_len < pred_len:
        raise ValueError(
            f"max_eval_pred_len={max_eval_pred_len} "
            f"不能小于当前 pred_len={pred_len}"
        )

    common_n_test = (  len(X_test_raw) - seq_len - horizon - max_eval_pred_len + 2)
    test_dataset = Subset(  test_dataset_full, list(range(common_n_test)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 推理模式：如果存在历史checkpoint，可强制使用保存时的feature_cols（防止列漂移）
    if (not train_mode):
        logger.info("Inference mode: will use current feature_cols (unless you load a specific checkpoint).")
 
    # ---- 模型 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # from optimized_wind_mamba import WindPowerMamba
    power_idx = feature_cols.index("power") if "power" in feature_cols else None
    if model_type == "wind_mamba":
        # model = WindPowerMamba(input_dim=65, d_model=96, n_heads=4, num_layers=2, dropout=0.15).to(device)
        model = WindPowerTransformer(
            input_dim=len(feature_cols),
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            pred_len=pred_len,dropout=dropout,
            power_idx=power_idx,
            recent_window=min(6, seq_len),
            use_persistence=True,
            local_kernel_sizes=(3, 5),
            mamba_kernel_size=3).to(device)
 
    
    elif model_type == "FAM":
        model =  WindPowerMambaFlowLiteV5(
            input_dim=len(feature_cols),
            d_model=d_model,
            n_heads=n_heads,
            num_mamba_layers=num_layers,
            pred_len=pred_len,
            dropout=dropout,
            recent_windows=[6, 12, 24],
            local_kernel_sizes=(3, 5),
            mamba_kernel_size=5,
            flow_expand=2,
            use_pos_enc=True,
            use_shortcut=True,
            shortcut_scale_init=0.1,
            gru_hidden=d_model,
        ).to(device)

    
    elif model_type == "QFAM":
        model = WindQMambaformer(input_dim=len(feature_cols),
                model_dim=d_model,
                mamba_d_state=64,  # 增加状态维度
                mamba_d_conv=4,
                mamba_expand=3,
                flow_attn_heads=8,  # 增加注意力头数
                pred_horizon=pred_len,
                dropout=dropout,
                use_mamba_ssm=True,
                num_layers=2,  # 多层堆叠
                qbits=6,
                qlayer=1).to(device)

    elif model_type == "mlp":
        model = WindPowerMLP(
            input_dim=len(feature_cols),
            d_model=512,
            pred_len=pred_len,
            num_layers=4,
            dropout=0.3,
        ).to(device)

    elif model_type == "lstm":
        model = WindPowerLSTM(
            input_dim=len(feature_cols),
            d_model=d_model,
            pred_len=pred_len,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

    elif model_type == "transformer":
        model = WindPowerSimpleTransformer(
            input_dim=len(feature_cols),
            d_model=d_model,
            pred_len=pred_len,
            num_layers=num_layers,
            dropout=dropout,
            n_heads=n_heads,
        ).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数量: total={total_params:,}, trainable={trainable_params:,}")
 
    # ---- 优化器 ----
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # ---- LR调度器 ----
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-3)
        scheduler_is_plateau = False
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
        scheduler_is_plateau = True
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        scheduler_is_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-3)
        scheduler_is_plateau = False

    criterion = nn.MSELoss()
    # criterion = HorizonWeightedMSE(pred_len=pred_len, mode="near").to(device)
    # criterion = nn.HuberLoss(delta=0.5)
    # ---- 训练（early stopping + 保存最佳）----
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad = 0

    if train_mode:
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0)
            val_metrics, _, _ = evaluate_scaled(model, val_loader, device)
            val_mse = val_metrics["MSE"]

            if scheduler_is_plateau:
                scheduler.step(val_mse)
            else:
                scheduler.step()

            if val_mse < best_val - 1e-9:
                best_val = val_mse
                best_epoch = epoch
                bad = 0
                best_state = copy.deepcopy(model.state_dict())

                torch.save({"model_state_dict": best_state,
                        "feature_cols": feature_cols,
                        "horizon": horizon,
                        "seq_len": seq_len,
                        "pred_len": pred_len,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "num_layers": num_layers,
                        "optimizer_name": optimizer_name,
                        "scheduler_name": scheduler_name,
                        "lr": lr,
                        "epoch": epoch,
                        "val_mse": best_val,
                        "exp_id": exp_id,
                        "dataset_name": dataset_name,
                        "freq": freq,
                        "model_type": model_type,}, checkpoint_path,)
                logger.info(f"💾 Saved best model at epoch {epoch} with val_MSE={best_val:.6f}")
            else:
                bad += 1

            logger.info(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} | "
                f"val_MSE={val_metrics['MSE']:.6f} val_RMSE={val_metrics['RMSE']:.6f} "
                f"val_MAE={val_metrics['MAE']:.6f} val_R2={val_metrics['R2']:.4f} "
                f"bad={bad}/{patience}"
            )

            if bad >= patience:
                logger.info(f"🛑 Early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val_MSE={best_val:.6f}")
                break
        # model.load_state_dict(best_state)
        # ckpt = torch.load(checkpoint_path, map_location=device)
        # model.load_state_dict(ckpt["model_state_dict"])# best_state
        model.load_state_dict(filter_state_dict(best_state), strict=False)
        logger.info(f"✅ Training completed. Best epoch={best_epoch}, best_val_MSE={best_val:.6f}")
        # ---- 测试评估（scaled）----
        test_metrics_scaled, y_pred_scaled, y_true_scaled = evaluate_scaled(model, test_loader, device)
    
    else:
        # ---- 加载最佳 ----
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(filter_state_dict(ckpt["model_state_dict"]), strict=False)
        logger.info(f"📂 Loaded best checkpoint from disk: {checkpoint_path}")

        
        # ---- 测试评估（scaled）----
        test_metrics_scaled, y_pred_scaled, y_true_scaled = evaluate_scaled(model, test_loader, device)

    # ---- 反标准化并重新评估（真实尺度）----
    y_pred_2d = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_true_2d = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(y_true_scaled.shape)
    y_pred = y_pred_2d.reshape(-1)
    y_true = y_true_2d.reshape(-1)
 
   
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    smape_v = smape(y_true, y_pred)

    test_metrics = {"MSE": mse, "MAPE": mape, "SMAPE": smape_v, "RMSE": rmse, "MAE": mae, "R2": r2}
 
    logger.info("===== TEST METRICS (ORIGINAL SCALE) =====")
    for k, v in test_metrics.items():
        logger.info(f"{k}: {v:.6f}")

    # ---- 对齐预测日期（关键：序列窗口偏移）----
    
    ctx = seq_len + horizon - 1
    n_test_samples = y_true_2d.shape[0]

    rows = []

    for sample_idx in range(n_test_samples):
        # 输入窗口最后一个时间点
        origin_idx = sample_idx + seq_len - 1
        forecast_origin_date = date_test_raw.iloc[origin_idx]

        for step_idx in range(pred_len):
            # 第 step_idx 个目标的真实时间位置
            target_idx = (
                sample_idx
                + seq_len
                + horizon
                - 1
                + step_idx
            )

            target_date = date_test_raw.iloc[target_idx]

            rows.append(
                {
                    # 不同 pred_len 模型共同的预测起点编号
                    "forecast_origin_id": sample_idx,

                    # 模型看到的最后一个时间点
                    "forecast_origin_date": forecast_origin_date,

                    # 实际被预测的未来时间点
                    "target_date": target_date,

                    # 1, 2, ..., pred_len
                    "pred_step": step_idx + 1,

                    # horizon, horizon+1, ...
                    "horizon_step": horizon + step_idx,

                    "power_true": float(
                        y_true_2d[sample_idx, step_idx]
                    ),

                    "power_pred": float(
                        y_pred_2d[sample_idx, step_idx]
                    ),
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"test_pred_{model_tag}.csv")
    out_df.to_csv(out_path, index=False)
 
    logger.info(f"Saved predictions (long): {out_path}")
    # logger.info(f"Saved predictions (wide): {out_path_wide}")

    # ---- 保存图 ----
    if save_forecast_plot is not None:
        try:
            plot_path = save_forecast_plot(
                out_df=out_df,
                out_dir=out_dir,
                model_tag=model_tag,
                horizon=horizon,
                y_true_col="power_true",
                y_pred_col="power_pred",
                ylabel="power",
                logger=logger,
                max_points=1000,
            )
        except Exception:
            plot_path = fallback_save_plot(out_df, out_dir, model_tag, ylabel="power", max_points=1000)
    else:
        plot_path = fallback_save_plot(out_df, out_dir, model_tag, ylabel="power", max_points=1000)

    # ---- 保存 feature_cols ----
    with open(os.path.join(out_dir, f"feature_cols_{model_tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))

    # ---- 保存 config ----
    config_summary = {
        "exp_id": exp_id,
        "dataset_name": dataset_name,
        "model_type": model_type,
        "freq": freq,
        "horizon": horizon,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "feature_cols": feature_cols,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(out_dir, f"config_{model_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(config_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Saved model & artifacts to: {out_dir}")
    logger.info(f"  - Best checkpoint: {checkpoint_path}")
 
    # logger.info(f"  - Predictions: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WindPowerTransformer with template-style pipeline")
    parser.add_argument("--csv_path", type=str, default="./dataset/winf_fam1.csv")
    parser.add_argument("--freq", type=str, default="1hour")
    parser.add_argument("--horizon", type=int, default=1)  # 从输入窗口末尾往后偏移多少步开始预测
    parser.add_argument("--seq_len", type=int, default=96)  
    parser.add_argument("--pred_len", type=int, default=3) # 从这个起点开始，连续预测多少步
    parser.add_argument(  "--max_eval_pred_len", type=int, default=12,help="不同 pred_len 实验统一使用的最大评估预测长度"
        )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--optimizer_name", type=str, default="adam", choices=["adamw", "adam", "sgd", "rmsprop"])
    parser.add_argument("--scheduler_name", type=str, default="plateau", choices=["plateau", "cosine", "step"])

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_mode", type=str2bool, default=True) # True False
    parser.add_argument("--model_type", type=str, default="QFAM",
                        choices=["wind_mamba", "wind_qmamba", "FAM", "QFAM",
                                 "mlp", "lstm", "transformer"])
    parser.add_argument("--exp_id", type=str, default='20260727_150901', help="Experiment ID to load for inference (e.g. 20240224_123456)")
    
    args = parser.parse_args()
    main(csv_path=args.csv_path,freq=args.freq,horizon=args.horizon,
        seq_len=args.seq_len,pred_len=args.pred_len,batch_size=args.batch_size,
        lr=args.lr,epochs=args.epochs,patience=args.patience,optimizer_name=args.optimizer_name,
        scheduler_name=args.scheduler_name,d_model=args.d_model,n_heads=args.n_heads,num_layers=args.num_layers,
        dropout=args.dropout,train_ratio=args.train_ratio,val_ratio=args.val_ratio,seed=args.seed,
        train_mode=args.train_mode,model_type=args.model_type,exp_id = args.exp_id,max_eval_pred_len=args.max_eval_pred_len,)

"""
python3 wind_mamba_private_QFAM_winf.py --csv_path ./dataset/winf_fam1.csv --model_type QFAM --train_mode true --epochs 100

python3 wind_mamba_private_QFAM_winf.py --csv_path ./dataset/winf_fam1.csv     --model_type QFAM --train_mode False 
--epochs 100 --exp_id 20260703_110652
 模型参数量: total=394,525
2026-07-03 11:15:33,042 - INFO - MSE: 23.130022
2026-07-03 11:15:33,042 - INFO - MAPE: 245.870002
2026-07-03 11:15:33,042 - INFO - SMAPE: 0.694904
2026-07-03 11:15:33,042 - INFO - RMSE: 4.809368
2026-07-03 11:15:33,043 - INFO - MAE: 3.419440
2026-07-03 11:15:33,043 - INFO - R2: 0.912495

python3 wind_mamba_private_QFAM_winf.py --csv_path ./dataset/winf_fam2.csv  --model_type QFAM --train_mode true --epochs 100
2026-07-03 13:52:24,790 - INFO - MSE: 148.748093
2026-07-03 13:52:24,790 - INFO - MAPE: 33.377811
2026-07-03 13:52:24,791 - INFO - SMAPE: 0.196551
2026-07-03 13:52:24,791 - INFO - RMSE: 12.196233
2026-07-03 13:52:24,791 - INFO - MAE: 9.441045
2026-07-03 13:52:24,791 - INFO - R2: 0.912348
2026-07-03 13:52:24,843 - INFO - Saved predictions (long): results_opt/QFAM/20260703_134531/test_pred_QFAM_h1_L96_P1_winf_fam2_20260703_134531.csv
2026-07-03 13:52:29,247 - INFO - 💾 Saved model & artifacts to: results_opt/QFAM/20260703_134531
2026-07-03 13:52:29,247 - INFO -   - Best checkpoint: results_opt/QFAM/20260703_134531/QFAM_h1_L96_P1_winf_fam2_20260703_134531_best.pt


使用更新后的mamba

模型参数量: total=652,573
2026-07-03 16:04:57,042 - INFO - MSE: 140.609558
2026-07-03 16:04:57,042 - INFO - MAPE: 19.175668
2026-07-03 16:04:57,043 - INFO - SMAPE: 0.183322
2026-07-03 16:04:57,044 - INFO - RMSE: 11.857890
2026-07-03 16:04:57,044 - INFO - MAE: 8.944383
2026-07-03 16:04:57,045 - INFO - R2: 0.917143
2026-07-03 16:04:57,125 - INFO - Saved predictions (long): results_opt/QFAM/20260703_153848/test_pred_QFAM_h1_L96_P1_winf_fam2_20260703_153848.csv
2026-07-03 16:05:02,262 - INFO - 💾 Saved model & artifacts to: results_opt/QFAM/20260703_153848
2026-07-03 16:05:02,264 - INFO -   - Best checkpoint: results_opt/QFAM/20260703_153848/QFAM_h1_L96_P1_winf_fam2_20260703_153848_best.pt

2026-07-03 17:46:08,321 - INFO - MSE: 23.252848
2026-07-03 17:46:08,322 - INFO - MAPE: 206.167479
2026-07-03 17:46:08,323 - INFO - SMAPE: 0.706919
2026-07-03 17:46:08,323 - INFO - RMSE: 4.822121
2026-07-03 17:46:08,324 - INFO - MAE: 3.401577
2026-07-03 17:46:08,325 - INFO - R2: 0.912338
2026-07-03 17:46:08,419 - INFO - Saved predictions (long): results_opt/QFAM/20260703_171110/test_pred_QFAM_h1_L96_P1_winf_fam1_20260703_171110.csv
2026-07-03 17:46:13,461 - INFO - 💾 Saved model & artifacts to: results_opt/QFAM/20260703_171110
2026-07-03 17:46:13,463 - INFO -   - Best checkpoint: results_opt/QFAM/20260703_171110/QFAM_h1_L96_P1_winf_fam1_20260703_171110_best.pt
"""