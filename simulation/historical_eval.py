"""
Historical Dataset STGNN Evaluation Engine
==========================================
Evaluates the trained STGNN and baseline models on the empirical held-out test partition
derived from the 5-room campus dataset (dataset/combined_Room1-5.csv).

Guarantees:
1. Strictly isolated from the synthetic digital twin simulation state.
2. Uses the exact chronological test boundary (rows 7099 to 8352, N = 1251 sequences).
3. Computes STGNN (continuous & integer), Persistence baseline, and Recent-Trend baseline.
4. Provides full per-room metrics, signed errors, and responsiveness diagnostics.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd

from stgnn_model import (
    STGNNInference,
    ROOMS,
    INPUT_FEATURES,
    TIMESTAMP_COL,
    TARGET_COL,
    SEQ_LEN,
    HORIZON,
)
from reconstruct_scalers import load_room, add_time_features


@dataclass
class HistoricalEvaluationResult:
    """Holds full historical evaluation outputs, DataFrames, and metrics."""
    test_sample_count: int
    test_start_timestamp: str
    test_end_timestamp: str
    evaluation_dataframe: pd.DataFrame
    building_metrics: dict[str, float]
    room_metrics: dict[str, dict[str, Any]]
    responsiveness: dict[str, dict[str, Any]]


_CACHED_HISTORICAL_RESULT: Optional[HistoricalEvaluationResult] = None


def get_historical_evaluation(
    dataset_dir: str = "dataset",
    force_recompute: bool = False,
) -> HistoricalEvaluationResult:
    """
    Computes or retrieves cached historical evaluation on the held-out test set.
    """
    global _CACHED_HISTORICAL_RESULT
    if _CACHED_HISTORICAL_RESULT is not None and not force_recompute:
        return _CACHED_HISTORICAL_RESULT

    # Check that dataset files exist
    for i, room in enumerate(ROOMS):
        path = os.path.join(dataset_dir, f"combined_Room{i+1}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Historical dataset file not found: {path}")

    infer = STGNNInference()

    # Load and process raw room datasets
    room_dfs = {
        r: add_time_features(load_room(r, os.path.join(dataset_dir, f"combined_Room{i+1}.csv")))
        for i, r in enumerate(ROOMS)
    }
    indexed = {
        r: room_dfs[r][[TIMESTAMP_COL] + INPUT_FEATURES].set_index(TIMESTAMP_COL).sort_index().drop_duplicates()
        for r in ROOMS
    }
    common_index = indexed[ROOMS[0]].index
    for r in ROOMS[1:]:
        common_index = common_index.intersection(indexed[r].index)
    common_index = common_index.sort_values()

    X_all = np.stack([indexed[r].reindex(common_index)[INPUT_FEATURES].to_numpy(dtype=np.float32) for r in ROOMS], axis=1)
    Y_all = np.stack([indexed[r].reindex(common_index)[TARGET_COL].to_numpy(dtype=np.float32) for r in ROOMS], axis=1)

    n = len(common_index)
    val_end = int(n * 0.85)  # 7099
    n_sensor_features = len(INPUT_FEATURES) - 1

    # Standardize
    X_scaled = np.zeros_like(X_all, dtype=np.float32)
    Y_scaled = np.zeros_like(Y_all, dtype=np.float32)
    for node, r in enumerate(ROOMS):
        X_scaled[:, node, :n_sensor_features] = infer.sensor_scalers[r].transform(X_all[:, node, :n_sensor_features]).astype(np.float32)
        X_scaled[:, node, -1] = infer.target_scalers[r].transform(Y_all[:, node].reshape(-1, 1)).reshape(-1).astype(np.float32)
        Y_scaled[:, node] = infer.target_scalers[r].transform(Y_all[:, node].reshape(-1, 1)).reshape(-1).astype(np.float32)

    # Form test sequences (12-step lookback, 3-step forecast horizon)
    Xs, Ys, origins, targets = [], [], [], []
    first_t = max(val_end, SEQ_LEN)
    last_t = n - HORIZON + 1
    for t in range(first_t, last_t):
        target_t = t + HORIZON - 1
        Xs.append(X_scaled[t - SEQ_LEN:t])
        Ys.append(Y_scaled[target_t])
        origins.append(common_index[t - 1])
        targets.append(common_index[target_t])

    X_test = np.asarray(Xs, dtype=np.float32)
    Y_test = np.asarray(Ys, dtype=np.float32)

    # Model inference
    pred_int, pred_cont = infer.predict(X_test)
    actual_cont = infer.inverse_transform_targets(Y_test)
    actual_int = infer.postprocess_integer_counts(actual_cont)

    # Persistence baseline (t - 1 occupancy)
    last_scaled_occ = X_test[:, -1, :, -1]
    pers_cont = infer.inverse_transform_targets(last_scaled_occ)
    pers_int = infer.postprocess_integer_counts(pers_cont)

    # Recent-Trend baseline: extrapolate 15m trend (from t-4 to t-1, i.e. 3 steps prior)
    prev_scaled_occ = X_test[:, -4, :, -1]
    prev_cont = infer.inverse_transform_targets(prev_scaled_occ)
    delta_recent = pers_cont - prev_cont
    trend_cont = np.maximum(pers_cont + delta_recent, 0.0)
    trend_int = infer.postprocess_integer_counts(trend_cont)

    # Assemble comprehensive evaluation DataFrame
    rows = []
    num_samples = len(X_test)
    for sample_idx in range(num_samples):
        orig_ts_str = str(origins[sample_idx])
        targ_ts_str = str(targets[sample_idx])
        for room_idx, room in enumerate(ROOMS):
            act_i = int(actual_int[sample_idx, room_idx])
            pred_i = int(pred_int[sample_idx, room_idx])
            pred_c = float(pred_cont[sample_idx, room_idx])
            pers_i = int(pers_int[sample_idx, room_idx])
            pers_c = float(pers_cont[sample_idx, room_idx])
            trnd_i = int(trend_int[sample_idx, room_idx])
            trnd_c = float(trend_cont[sample_idx, room_idx])

            err_stgnn_int = pred_i - act_i
            err_pers = pers_i - act_i
            err_trend = trnd_i - act_i

            rows.append({
                "origin_timestamp": orig_ts_str,
                "target_timestamp": targ_ts_str,
                "room_id": room,
                "actual": act_i,
                "predicted_continuous": pred_c,
                "predicted_integer": pred_i,
                "persistence": pers_i,
                "recent_trend": trnd_i,
                "forecast_delta_raw": pred_c - pers_c,
                "forecast_delta_int": pred_i - pers_i,
                "actual_15m_change": act_i - pers_i,
                "error_stgnn_int": err_stgnn_int,
                "abs_error_stgnn_int": abs(err_stgnn_int),
                "error_persistence": err_pers,
                "abs_error_persistence": abs(err_pers),
                "error_trend": err_trend,
                "abs_error_trend": abs(err_trend),
            })

    eval_df = pd.DataFrame(rows)

    # Compute overall building and per-room metrics
    room_metrics = {}
    responsiveness = {}

    all_y_true = eval_df["actual"].to_numpy(dtype=float)
    all_y_stgnn = eval_df["predicted_integer"].to_numpy(dtype=float)
    all_y_cont = eval_df["predicted_continuous"].to_numpy(dtype=float)
    all_y_pers = eval_df["persistence"].to_numpy(dtype=float)
    all_y_trend = eval_df["recent_trend"].to_numpy(dtype=float)

    building_metrics = {
        "stgnn_integer_mae": float(np.mean(np.abs(all_y_stgnn - all_y_true))),
        "stgnn_integer_rmse": float(np.sqrt(np.mean((all_y_stgnn - all_y_true) ** 2))),
        "stgnn_integer_medae": float(np.median(np.abs(all_y_stgnn - all_y_true))),
        "stgnn_integer_bias": float(np.mean(all_y_stgnn - all_y_true)),
        "stgnn_continuous_mae": float(np.mean(np.abs(all_y_cont - all_y_true))),
        "stgnn_continuous_rmse": float(np.sqrt(np.mean((all_y_cont - all_y_true) ** 2))),
        "persistence_mae": float(np.mean(np.abs(all_y_pers - all_y_true))),
        "persistence_rmse": float(np.sqrt(np.mean((all_y_pers - all_y_true) ** 2))),
        "persistence_medae": float(np.median(np.abs(all_y_pers - all_y_true))),
        "persistence_bias": float(np.mean(all_y_pers - all_y_true)),
        "trend_mae": float(np.mean(np.abs(all_y_trend - all_y_true))),
        "trend_rmse": float(np.sqrt(np.mean((all_y_trend - all_y_true) ** 2))),
        "trend_medae": float(np.median(np.abs(all_y_trend - all_y_true))),
        "trend_bias": float(np.mean(all_y_trend - all_y_true)),
    }

    for room in ROOMS:
        rdf = eval_df[eval_df["room_id"] == room]
        r_true = rdf["actual"].to_numpy(dtype=float)
        r_stgnn = rdf["predicted_integer"].to_numpy(dtype=float)
        r_cont = rdf["predicted_continuous"].to_numpy(dtype=float)
        r_pers = rdf["persistence"].to_numpy(dtype=float)
        r_trnd = rdf["recent_trend"].to_numpy(dtype=float)

        stgnn_mae = float(np.mean(np.abs(r_stgnn - r_true)))
        pers_mae = float(np.mean(np.abs(r_pers - r_true)))
        trnd_mae = float(np.mean(np.abs(r_trnd - r_true)))

        room_metrics[room] = {
            "samples": len(rdf),
            "stgnn_int_mae": stgnn_mae,
            "stgnn_int_rmse": float(np.sqrt(np.mean((r_stgnn - r_true) ** 2))),
            "stgnn_int_medae": float(np.median(np.abs(r_stgnn - r_true))),
            "stgnn_int_bias": float(np.mean(r_stgnn - r_true)),
            "stgnn_cont_mae": float(np.mean(np.abs(r_cont - r_true))),
            "stgnn_cont_rmse": float(np.sqrt(np.mean((r_cont - r_true) ** 2))),
            "persistence_mae": pers_mae,
            "persistence_rmse": float(np.sqrt(np.mean((r_pers - r_true) ** 2))),
            "persistence_medae": float(np.median(np.abs(r_pers - r_true))),
            "persistence_bias": float(np.mean(r_pers - r_true)),
            "trend_mae": trnd_mae,
            "trend_rmse": float(np.sqrt(np.mean((r_trnd - r_true) ** 2))),
            "trend_medae": float(np.median(np.abs(r_trnd - r_true))),
            "trend_bias": float(np.mean(r_trnd - r_true)),
            "improvement_vs_persistence": round(pers_mae - stgnn_mae, 4),
            "improvement_vs_trend": round(trnd_mae - stgnn_mae, 4),
        }

        # Responsiveness statistics
        raw_deltas = rdf["forecast_delta_raw"].to_numpy(dtype=float)
        int_deltas = rdf["forecast_delta_int"].to_numpy(dtype=float)
        act_changes = rdf["actual_15m_change"].to_numpy(dtype=float)

        responsiveness[room] = {
            "raw_std": float(np.std(r_cont)),
            "raw_delta_mean": float(np.mean(raw_deltas)),
            "raw_delta_std": float(np.std(raw_deltas)),
            "raw_delta_min": float(np.min(raw_deltas)),
            "raw_delta_max": float(np.max(raw_deltas)),
            "zero_rounded_change_pct": float((int_deltas == 0).mean() * 100.0),
            "change_ge_1_pct": float((np.abs(int_deltas) >= 1).mean() * 100.0),
            "actual_zero_change_pct": float((act_changes == 0).mean() * 100.0),
            "prediction_persistence_pct": float((np.diff(r_stgnn) == 0).mean() * 100.0) if len(r_stgnn) > 1 else 100.0,
        }

    res = HistoricalEvaluationResult(
        test_sample_count=num_samples,
        test_start_timestamp=str(origins[0]),
        test_end_timestamp=str(targets[-1]),
        evaluation_dataframe=eval_df,
        building_metrics=building_metrics,
        room_metrics=room_metrics,
        responsiveness=responsiveness,
    )
    _CACHED_HISTORICAL_RESULT = res
    return res
