"""
End-to-End Digital Twin Inference Pipeline
==========================================
Integrates the 5-room virtual building, virtual environmental sensor engine,
and verified STGNN inference engine into a unified real-time forecasting pipeline.

Core Guarantees:
1. Strict 5-room spatial topology matching model node ordering (Room 1 to Room 5).
2. Rolling 12-step lookback buffer across all 13 features (60-minute historical context).
3. Exact feature ordering: 7 sensors + 5 cyclic time features + 1 observed occupancy count.
4. 15-minute ahead predictive forecasting (horizon: 3 steps x 5 min = 15 min).
5. Strict temporal causality: zero future ground-truth leakage during inference.
6. Robust warm-up handling, input validation, and explicit persistence baseline comparison.
"""

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import numpy as np
import pandas as pd

from simulation.config import EXPECTED_ROOM_IDS
from simulation.engine import SimulationEngine
from stgnn_model import (
    STGNNInference,
    COMMON_SENSOR_FEATURES,
    TIME_FEATURES,
    INPUT_FEATURES,
    SEQ_LEN,
    HORIZON,
    CADENCE_MINUTES,
    compute_time_features_df,
)


@dataclass
class PredictionRecord:
    """
    Standardized forecast record maintaining both continuous and discrete representations.
    Retains explicit separation between raw model output and display estimates.
    """
    forecast_made_at: str
    target_timestamp: str
    room_id: str
    predicted_occupants: Optional[int]
    predicted_continuous: Optional[float]
    persistence_occupants: Optional[int]
    actual_occupants_at_target: Optional[int] = None
    status: str = "SUCCESS"
    forecast_id: Optional[str] = None
    trend_occupants: Optional[int] = None
    trend_continuous: Optional[float] = None
    forecast_delta: Optional[float] = None
    model_version: str = "STGNN_v1_ResNet"

    def __post_init__(self):
        if self.forecast_id is None:
            self.forecast_id = f"{self.forecast_made_at}_{self.room_id}"
        if self.forecast_delta is None and self.predicted_continuous is not None and self.persistence_occupants is not None:
            self.forecast_delta = round(float(self.predicted_continuous) - float(self.persistence_occupants), 4)

    @property
    def raw_prediction(self) -> Optional[float]:
        """Continuous numeric model output after inverse transformation."""
        return self.predicted_continuous

    @property
    def display_prediction(self) -> Optional[int]:
        """Clearly documented integer occupancy estimate (clamped non-negative rounded count)."""
        return self.predicted_occupants

    @property
    def current_occupancy(self) -> Optional[int]:
        """Observed ground truth at the time the forecast was generated."""
        return self.persistence_occupants

    @property
    def forecast_origin_timestamp(self) -> str:
        """Timestamp of observations when forecast was initiated."""
        return self.forecast_made_at

    @property
    def model_status(self) -> str:
        """Status string indicating if prediction is valid, warmup, invalid, or unavailable."""
        return self.status


class DigitalTwinPipeline:
    """
    Coordinates building simulation, virtual sensors, rolling lookback window,
    and STGNN neural network inference for real-time 15-minute occupancy forecasting.
    """

    def __init__(
        self,
        engine: Optional[SimulationEngine] = None,
        inference: Optional[STGNNInference] = None,
        seed: int = 42,
        model_path: str = "joint_lstm_gnn_all_5_rooms.keras",
        scalers_path: str = "artifacts/scalers.joblib",
    ):
        self.seed = seed
        self.engine = engine or SimulationEngine(seed=self.seed)

        # Initialize STGNN inference engine safely
        if inference is not None:
            self.inference = inference
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"STGNN model archive not found: {model_path}")
            if not os.path.exists(scalers_path):
                raise FileNotFoundError(f"Scalers artifact not found: {scalers_path}")
            self.inference = STGNNInference(
                model_path=model_path,
                scalers_path=scalers_path,
            )

        self.seq_len = SEQ_LEN # 12
        self.horizon_steps = HORIZON # 3 steps = 15 minutes
        self.cadence_minutes = CADENCE_MINUTES # 5 minutes

        # Rolling history buffer: list of dicts mapping room_id -> feature row
        self.rolling_buffer: list[dict[str, dict[str, Any]]] = []

        # Completed and pending prediction logs
        self.predictions: list[PredictionRecord] = []
        self.pending_evaluations: list[PredictionRecord] = []

        # Record initial t=0 state into buffer
        self._record_step_to_buffer(self.engine.initial_time)

    def _record_step_to_buffer(self, dt: datetime):
        """Extracts current sensor and occupancy state for all 5 rooms and adds to rolling buffer."""
        ts_str = dt.isoformat()
        latest_sensors = self.engine.sensor_manager.get_latest_readings()

        # Compute cyclic time features for this timestamp
        minutes = dt.hour * 60 + dt.minute
        dow = dt.weekday()
        time_dict = {
            "sin_hour": float(np.sin(2 * np.pi * minutes / (24 * 60))),
            "cos_hour": float(np.cos(2 * np.pi * minutes / (24 * 60))),
            "sin_week": float(np.sin(2 * np.pi * dow / 7)),
            "cos_week": float(np.cos(2 * np.pi * dow / 7)),
            "is_weekend": 1.0 if dow >= 5 else 0.0,
        }

        step_data = {}
        for room_id in EXPECTED_ROOM_IDS:
            room = self.engine.building.get_room(room_id)
            reading = latest_sensors.get(room_id)
            if reading is None:
                raise ValueError(f"Missing sensor telemetry for {room_id} at {ts_str}")

            # Validate reading numeric integrity (no NaNs or Infs)
            for feat in COMMON_SENSOR_FEATURES:
                val = getattr(reading, feat)
                if not np.isfinite(val):
                    raise ValueError(f"Non-finite sensor reading {feat}={val} in {room_id} at {ts_str}")

            curr_occ = room.occupant_count
            if curr_occ < 0 or not np.isfinite(curr_occ):
                raise ValueError(f"Invalid occupant count {curr_occ} in {room_id} at {ts_str}")

            row = {
                "timestamp": ts_str,
                "room_id": room_id,
                "occupant_count": float(curr_occ),
            }
            # Add 7 environmental sensor features
            for feat in COMMON_SENSOR_FEATURES:
                row[feat] = getattr(reading, feat)
            # Add 5 cyclic time features
            row.update(time_dict)

            step_data[room_id] = row

        self.rolling_buffer.append(step_data)

        # Maintain buffer size at most seq_len
        if len(self.rolling_buffer) > self.seq_len:
            self.rolling_buffer.pop(0)

    def _backfill_actuals(self, current_ts_str: str):
        """Backfills ground-truth occupant counts for predictions reaching target timestamp."""
        remaining_pending = []
        for record in self.pending_evaluations:
            if record.target_timestamp == current_ts_str:
                room = self.engine.building.get_room(record.room_id)
                record.actual_occupants_at_target = room.occupant_count
            elif record.target_timestamp < current_ts_str and record.actual_occupants_at_target is None:
                # If target was in the past (e.g. after jump or skip), resolve from history
                hist_df = self.engine.recorder.to_dataframe()
                match = hist_df[
                    (hist_df["timestamp"] == record.target_timestamp) &
                    (hist_df["room_id"] == record.room_id)
                ]
                if not match.empty:
                    record.actual_occupants_at_target = int(match["actual_occupants"].iloc[0])
                else:
                    remaining_pending.append(record)
            else:
                remaining_pending.append(record)
        self.pending_evaluations = remaining_pending

    def step(self) -> dict[str, Any]:
        """
        Advances the entire pipeline by one 5-minute logical step:
        1. Advances building simulation & virtual sensors.
        2. Backfills ground truth for prior forecasts reaching target time.
        3. Appends current step observations to rolling buffer.
        4. If buffer is full (12 steps), invokes STGNN model to forecast 15 min ahead.
        """
        # 1. Advance simulation engine
        step_result = self.engine.step()
        curr_dt = self.engine.current_time
        curr_ts = curr_dt.isoformat()

        # 2. Backfill target actuals for past forecasts
        self._backfill_actuals(curr_ts)

        # 3. Append current observation to rolling buffer
        self._record_step_to_buffer(curr_dt)

        # Target forecast timestamp (+15 minutes ahead)
        target_dt = curr_dt + timedelta(minutes=self.horizon_steps * self.cadence_minutes)
        target_ts = target_dt.isoformat()

        # 4. Check warm-up status
        if len(self.rolling_buffer) < self.seq_len:
            # Warm-up phase: buffer does not have 12 steps yet
            existing_ids = {r.forecast_id for r in self.predictions}
            for room_id in EXPECTED_ROOM_IDS:
                fid = f"{curr_ts}_{room_id}"
                rec = PredictionRecord(
                    forecast_made_at=curr_ts,
                    target_timestamp=target_ts,
                    room_id=room_id,
                    predicted_occupants=None,
                    predicted_continuous=None,
                    persistence_occupants=self.engine.building.get_room(room_id).occupant_count,
                    actual_occupants_at_target=None,
                    status="WARMUP",
                    forecast_id=fid,
                )
                if fid not in existing_ids:
                    self.predictions.append(rec)
                    existing_ids.add(fid)
            return {
                "step": self.engine.step_count,
                "timestamp": curr_ts,
                "status": "WARMUP",
                "buffer_length": len(self.rolling_buffer),
                "predictions": None,
            }

        # 5. Construct 12-step lookback DataFrames per room
        window_dfs = {}
        for room_id in EXPECTED_ROOM_IDS:
            room_rows = [b[room_id] for b in self.rolling_buffer]
            df = pd.DataFrame(room_rows)
            window_dfs[room_id] = df

        # 6. Transform and validate tensor shape (1, 12, 5, 13)
        X_norm = self.inference.transform_raw_window(window_dfs)
        assert X_norm.shape == (1, self.seq_len, len(EXPECTED_ROOM_IDS), len(INPUT_FEATURES)), (
            f"Invalid input tensor shape: {X_norm.shape}"
        )

        # 7. Model forward pass with explicit finite validation
        pred_int, pred_cont = self.inference.predict(X_norm)
        is_finite = bool(np.all(np.isfinite(pred_cont)))

        # 8. Log predictions and baselines
        step_predictions = {}
        existing_ids = {r.forecast_id for r in self.predictions}

        for i, room_id in enumerate(EXPECTED_ROOM_IDS):
            pers_val = self.engine.building.get_room(room_id).occupant_count
            fid = f"{curr_ts}_{room_id}"

            # Calculate recent-trend momentum (extrapolating 15m change from t-3 to t)
            room_hist = [b[room_id]["occupant_count"] for b in self.rolling_buffer]
            cap = self.engine.building.get_room(room_id).capacity
            if len(room_hist) >= 4:
                delta_recent = room_hist[-1] - room_hist[-4]
                trend_c = max(0.0, min(float(cap), float(room_hist[-1] + delta_recent)))
                trend_i = int(round(trend_c))
            else:
                trend_c = float(pers_val)
                trend_i = int(pers_val)

            if is_finite:
                int_val = int(pred_int[0, i])
                cont_val = float(pred_cont[0, i])
                rec_status = "SUCCESS"
                raw_delta = cont_val - pers_val
            else:
                int_val = None
                cont_val = None
                rec_status = "INVALID"
                raw_delta = None

            rec = PredictionRecord(
                forecast_made_at=curr_ts,
                target_timestamp=target_ts,
                room_id=room_id,
                predicted_occupants=int_val,
                predicted_continuous=cont_val,
                persistence_occupants=pers_val,
                actual_occupants_at_target=None,
                status=rec_status,
                forecast_id=fid,
                trend_occupants=trend_i,
                trend_continuous=round(trend_c, 4),
                forecast_delta=round(raw_delta, 4) if raw_delta is not None else None,
            )

            if fid not in existing_ids:
                self.predictions.append(rec)
                self.pending_evaluations.append(rec)
                existing_ids.add(fid)

            step_predictions[room_id] = {
                "forecast_id": fid,
                "forecast_origin_timestamp": curr_ts,
                "target_timestamp": target_ts,
                "room_id": room_id,
                "current_occupancy": pers_val,
                "raw_prediction": round(cont_val, 4) if cont_val is not None else None,
                "display_prediction": int_val,
                "forecast_delta": round(raw_delta, 4) if raw_delta is not None else None,
                "display_delta": (int_val - pers_val) if int_val is not None else None,
                "persistence_occupants": pers_val,
                "trend_occupants": trend_i,
                "trend_continuous": round(trend_c, 4),
                "model_status": rec_status,
                # Legacy keys for existing test backward compatibility
                "predicted": int_val,
                "predicted_occupants": int_val,
                "predicted_continuous": round(cont_val, 2) if cont_val is not None else None,
                "persistence": pers_val,
            }

        return {
            "step": self.engine.step_count,
            "timestamp": curr_ts,
            "status": "SUCCESS" if is_finite else "INVALID",
            "buffer_length": len(self.rolling_buffer),
            "target_timestamp": target_ts,
            "predictions": step_predictions,
        }

    def run(self, num_steps: int) -> pd.DataFrame:
        """Runs the pipeline for a specified number of steps and returns prediction history."""
        for _ in range(num_steps):
            self.step()
        return self.get_predictions_df()

    def get_predictions_df(self) -> pd.DataFrame:
        """Returns the full prediction history as a pandas DataFrame."""
        if not self.predictions:
            return pd.DataFrame(columns=[
                "forecast_made_at", "target_timestamp", "room_id",
                "predicted_occupants", "predicted_continuous",
                "persistence_occupants", "actual_occupants_at_target",
                "status"
            ])
        data = [asdict(r) for r in self.predictions]
        df = pd.DataFrame(data)
        for col in ["predicted_occupants", "persistence_occupants", "actual_occupants_at_target"]:
            if col in df.columns:
                df[col] = df[col].astype("Int64")
        return df


    def export_predictions(self, filepath: str = "artifacts/occupancy_predictions.csv", full: bool = False):
        """Exports recorded predictions and evaluations to CSV."""
        df = self.get_predictions_df()
        if not full:
            legacy_cols = [
                "forecast_made_at", "target_timestamp", "room_id",
                "predicted_occupants", "predicted_continuous",
                "persistence_occupants", "actual_occupants_at_target",
                "status"
            ]
            cols = [c for c in legacy_cols if c in df.columns]
            df = df[cols]
        os_dir = os.path.dirname(filepath)
        if os_dir:
            os.makedirs(os_dir, exist_ok=True)
        df.to_csv(filepath, index=False)

    def compute_evaluation_metrics(self) -> dict[str, Any]:
        """
        Calculates MAE, RMSE, MedAE, bias, and baseline comparative metrics between
        predicted occupancy, persistence baseline, recent-trend baseline, and ground truth.
        """
        df = self.get_predictions_df()
        eval_df = df[
            (df["status"] == "SUCCESS") &
            (df["actual_occupants_at_target"].notna()) &
            (df["predicted_occupants"].notna())
        ].copy()

        if eval_df.empty:
            return {
                "total_evaluated_samples": 0,
                "building_metrics": {},
                "room_metrics": {},
            }

        y_true = eval_df["actual_occupants_at_target"].astype(float).values
        y_pred_int = eval_df["predicted_occupants"].astype(float).values
        y_pred_cont = eval_df["predicted_continuous"].astype(float).values
        y_pers = eval_df["persistence_occupants"].astype(float).values

        # Trend baseline values (fallback to persistence if not recorded)
        if "trend_occupants" in eval_df.columns and eval_df["trend_occupants"].notna().any():
            y_trend = eval_df["trend_occupants"].fillna(eval_df["persistence_occupants"]).astype(float).values
        else:
            y_trend = y_pers.copy()

        # Building-wide metrics
        mae_stgnn_int = float(np.mean(np.abs(y_pred_int - y_true)))
        rmse_stgnn_int = float(np.sqrt(np.mean((y_pred_int - y_true) ** 2)))
        medae_stgnn_int = float(np.median(np.abs(y_pred_int - y_true)))
        bias_stgnn_int = float(np.mean(y_pred_int - y_true))

        mae_stgnn_cont = float(np.mean(np.abs(y_pred_cont - y_true)))
        rmse_stgnn_cont = float(np.sqrt(np.mean((y_pred_cont - y_true) ** 2)))

        mae_pers = float(np.mean(np.abs(y_pers - y_true)))
        rmse_pers = float(np.sqrt(np.mean((y_pers - y_true) ** 2)))
        medae_pers = float(np.median(np.abs(y_pers - y_true)))
        bias_pers = float(np.mean(y_pers - y_true))

        mae_trend = float(np.mean(np.abs(y_trend - y_true)))
        rmse_trend = float(np.sqrt(np.mean((y_trend - y_true) ** 2)))
        medae_trend = float(np.median(np.abs(y_trend - y_true)))
        bias_trend = float(np.mean(y_trend - y_true))

        unchanged_frac = float(np.mean(y_pred_int == y_pers))
        change_rate = float(np.mean(y_pred_int != y_pers))

        room_metrics = {}
        for room_id in EXPECTED_ROOM_IDS:
            rdf = eval_df[eval_df["room_id"] == room_id]
            if rdf.empty:
                continue
            r_true = rdf["actual_occupants_at_target"].astype(float).values
            r_pred_int = rdf["predicted_occupants"].astype(float).values
            r_pred_cont = rdf["predicted_continuous"].astype(float).values
            r_pers = rdf["persistence_occupants"].astype(float).values

            if "trend_occupants" in rdf.columns and rdf["trend_occupants"].notna().any():
                r_trend = rdf["trend_occupants"].fillna(rdf["persistence_occupants"]).astype(float).values
            else:
                r_trend = r_pers.copy()

            r_stgnn_mae = float(np.mean(np.abs(r_pred_int - r_true)))
            r_pers_mae = float(np.mean(np.abs(r_pers - r_true)))
            r_trend_mae = float(np.mean(np.abs(r_trend - r_true)))

            room_metrics[room_id] = {
                "samples": len(rdf),
                "stgnn_int_mae": r_stgnn_mae,
                "stgnn_int_rmse": float(np.sqrt(np.mean((r_pred_int - r_true) ** 2))),
                "stgnn_int_medae": float(np.median(np.abs(r_pred_int - r_true))),
                "stgnn_int_bias": float(np.mean(r_pred_int - r_true)),
                "stgnn_cont_mae": float(np.mean(np.abs(r_pred_cont - r_true))),
                "stgnn_cont_rmse": float(np.sqrt(np.mean((r_pred_cont - r_true) ** 2))),
                "persistence_mae": r_pers_mae,
                "persistence_rmse": float(np.sqrt(np.mean((r_pers - r_true) ** 2))),
                "persistence_medae": float(np.median(np.abs(r_pers - r_true))),
                "persistence_bias": float(np.mean(r_pers - r_true)),
                "trend_mae": r_trend_mae,
                "trend_rmse": float(np.sqrt(np.mean((r_trend - r_true) ** 2))),
                "trend_medae": float(np.median(np.abs(r_trend - r_true))),
                "trend_bias": float(np.mean(r_trend - r_true)),
                "improvement_vs_persistence": round(r_pers_mae - r_stgnn_mae, 4),
                "improvement_vs_trend": round(r_trend_mae - r_stgnn_mae, 4),
                "unchanged_forecast_fraction": float(np.mean(r_pred_int == r_pers)),
                "forecast_change_rate": float(np.mean(r_pred_int != r_pers)),
            }

        return {
            "total_evaluated_samples": len(eval_df),
            "building_metrics": {
                "stgnn_integer_mae": mae_stgnn_int,
                "stgnn_integer_rmse": rmse_stgnn_int,
                "stgnn_integer_medae": medae_stgnn_int,
                "stgnn_integer_bias": bias_stgnn_int,
                "stgnn_continuous_mae": mae_stgnn_cont,
                "stgnn_continuous_rmse": rmse_stgnn_cont,
                "persistence_mae": mae_pers,
                "persistence_rmse": rmse_pers,
                "persistence_medae": medae_pers,
                "persistence_bias": bias_pers,
                "trend_mae": mae_trend,
                "trend_rmse": rmse_trend,
                "trend_medae": medae_trend,
                "trend_bias": bias_trend,
                "unchanged_forecast_fraction": unchanged_frac,
                "forecast_change_rate": change_rate,
                "improvement_vs_persistence": round(mae_pers - mae_stgnn_int, 4),
                "improvement_vs_trend": round(mae_trend - mae_stgnn_int, 4),
            },
            "room_metrics": room_metrics,
        }

    def compute_responsiveness_metrics(self) -> dict[str, Any]:
        """
        Calculates prediction responsiveness diagnostics across all rooms
        from the recorded prediction history.
        """
        df = self.get_predictions_df()
        success_df = df[
            (df["status"] == "SUCCESS") &
            (df["predicted_continuous"].notna()) &
            (df["persistence_occupants"].notna())
        ].copy()

        if success_df.empty:
            return {"building": {}, "rooms": {}}

        room_resp = {}
        for room_id in EXPECTED_ROOM_IDS:
            rdf = success_df[success_df["room_id"] == room_id]
            if rdf.empty:
                continue
            r_cont = rdf["predicted_continuous"].to_numpy(dtype=float)
            r_int = rdf["predicted_occupants"].to_numpy(dtype=float)
            r_pers = rdf["persistence_occupants"].to_numpy(dtype=float)
            r_raw_delta = r_cont - r_pers
            r_int_delta = r_int - r_pers

            room_resp[room_id] = {
                "samples": len(rdf),
                "raw_mean": float(np.mean(r_cont)),
                "raw_std": float(np.std(r_cont)),
                "raw_min": float(np.min(r_cont)),
                "raw_max": float(np.max(r_cont)),
                "raw_delta_mean": float(np.mean(r_raw_delta)),
                "raw_delta_std": float(np.std(r_raw_delta)),
                "raw_delta_min": float(np.min(r_raw_delta)),
                "raw_delta_max": float(np.max(r_raw_delta)),
                "zero_rounded_change_pct": float((r_int_delta == 0).mean() * 100.0),
                "change_ge_1_pct": float((np.abs(r_int_delta) >= 1).mean() * 100.0),
                "prediction_persistence_pct": float((np.diff(r_int) == 0).mean() * 100.0) if len(r_int) > 1 else 100.0,
            }

        all_cont = success_df["predicted_continuous"].to_numpy(dtype=float)
        all_int = success_df["predicted_occupants"].to_numpy(dtype=float)
        all_pers = success_df["persistence_occupants"].to_numpy(dtype=float)
        all_raw_delta = all_cont - all_pers
        all_int_delta = all_int - all_pers

        bld_resp = {
            "samples": len(success_df),
            "raw_mean": float(np.mean(all_cont)),
            "raw_std": float(np.std(all_cont)),
            "raw_min": float(np.min(all_cont)),
            "raw_max": float(np.max(all_cont)),
            "raw_delta_mean": float(np.mean(all_raw_delta)),
            "raw_delta_std": float(np.std(all_raw_delta)),
            "zero_rounded_change_pct": float((all_int_delta == 0).mean() * 100.0),
            "change_ge_1_pct": float((np.abs(all_int_delta) >= 1).mean() * 100.0),
        }

        return {"building": bld_resp, "rooms": room_resp}

    def _update_latest_buffer_step(self):
        """Refreshes the latest entry in the rolling buffer with current engine state."""
        if not self.rolling_buffer:
            return
        curr_dt = self.engine.current_time
        latest_sensors = self.engine.sensor_manager.get_latest_readings()

        step_data = self.rolling_buffer[-1]
        for room_id in EXPECTED_ROOM_IDS:
            room = self.engine.building.get_room(room_id)
            reading = latest_sensors.get(room_id)
            if reading and room_id in step_data:
                step_data[room_id]["occupant_count"] = float(room.occupant_count)
                for feat in COMMON_SENSOR_FEATURES:
                    step_data[room_id][feat] = getattr(reading, feat)

    def add_person(self, count: int = 1, destination: Optional[str] = None) -> dict[str, Any]:
        """Interactively adds persons through engine and refreshes rolling buffer."""
        res = self.engine.add_person(count=count, destination=destination)
        self._update_latest_buffer_step()
        return res

    def exit_person(self, count: int = 1, room_id: Optional[str] = None) -> dict[str, Any]:
        """Interactively exits persons through engine and refreshes rolling buffer."""
        res = self.engine.exit_person(count=count, room_id=room_id)
        self._update_latest_buffer_step()
        return res

    def move_people(
        self,
        source_room: str,
        destination_room: str,
        person_ids: Optional[list[str]] = None,
        count: Optional[int] = None,
    ) -> dict[str, Any]:
        """Manually moves occupants via engine and refreshes rolling buffer telemetry."""
        res = self.engine.move_people(
            source_room=source_room,
            destination_room=destination_room,
            person_ids=person_ids,
            count=count,
        )
        self._update_latest_buffer_step()
        return res

    def reset(self, new_seed: Optional[int] = None, keep_session_counter: bool = True):
        """Resets the simulation and clears rolling buffers and prediction logs."""
        if new_seed is not None:
            self.seed = new_seed
        self.engine.reset(self.seed, keep_session_counter=keep_session_counter)
        self.rolling_buffer.clear()
        self.predictions.clear()
        self.pending_evaluations.clear()
        self._record_step_to_buffer(self.engine.initial_time)

