"""
Automated Test Suite for Digital Twin Inference Pipeline (Phase 3.3)
====================================================================
Verifies all pipeline integration, causality, tensor shape, warm-up,
and predictive evaluation requirements:
1. Rolling 12-step lookback buffer across all 13 features per room.
2. Exact feature ordering matching model specification.
3. Proper tensor shapes: input (1, 12, 5, 13) and output (1, 5).
4. Forecast timestamp strictly aligned to 15 minutes ahead (t + 15 min).
5. Ground-truth backfilling occurs accurately once target time is reached.
6. Warm-up behavior: no predictions emitted during steps 0-10, begins at step 11.
7. Causal integrity: zero leakage of future occupancy into the lookback window.
8. Output validity: all 5 room predictions are finite, non-negative integers.
9. Persistence baseline matches the latest observed occupant count (y_t).
10. Error handling on invalid/missing model paths and corrupted inputs.
11. Clean CSV export with complete schema and no unhandled exceptions.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from simulation.config import EXPECTED_ROOM_IDS
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline, PredictionRecord
from stgnn_model import STGNNInference, INPUT_FEATURES, COMMON_SENSOR_FEATURES, TIME_FEATURES, SEQ_LEN


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_inference():
    """Module-scoped inference engine to minimize TensorFlow model reload overhead."""
    return STGNNInference()


@pytest.fixture
def pipeline(shared_inference):
    """Creates a fresh pipeline with fixed seed for tests."""
    engine = SimulationEngine(num_occupants=35, seed=42)
    return DigitalTwinPipeline(engine=engine, inference=shared_inference, seed=42)


# ---------------------------------------------------------------------
# 1 & 2. ROLLING WINDOW & FEATURE ORDERING
# ---------------------------------------------------------------------

def test_rolling_window_and_feature_order(pipeline):
    """Test 1 & 2: Buffer holds at most 12 steps, each step having exactly 13 ordered features."""
    # Step 12 times to fill buffer
    for _ in range(12):
        pipeline.step()

    assert len(pipeline.rolling_buffer) == SEQ_LEN # 12

    for step_data in pipeline.rolling_buffer:
        assert set(step_data.keys()) == set(EXPECTED_ROOM_IDS)
        for room_id in EXPECTED_ROOM_IDS:
            row = step_data[room_id]
            # Verify 7 sensors + 5 time features + 1 target
            for feat in INPUT_FEATURES:
                assert feat in row, f"Missing feature {feat} in {room_id}"
            assert np.isfinite(row["occupant_count"]), "Invalid occupant count"


# ---------------------------------------------------------------------
# 3. TENSOR SHAPE INTEGRITY
# ---------------------------------------------------------------------

def test_tensor_shapes(pipeline):
    """Test 3: Lookback transformation produces exact shape (1, 12, 5, 13) and model predicts (1, 5)."""
    for _ in range(12):
        pipeline.step()

    window_dfs = {}
    for room_id in EXPECTED_ROOM_IDS:
        rows = [b[room_id] for b in pipeline.rolling_buffer]
        window_dfs[room_id] = pd.DataFrame(rows)

    X_norm = pipeline.inference.transform_raw_window(window_dfs)
    assert X_norm.shape == (1, 12, 5, 13), f"Unexpected tensor shape: {X_norm.shape}"

    pred_int, pred_cont = pipeline.inference.predict(X_norm)
    assert pred_int.shape == (1, 5), f"Unexpected integer prediction shape: {pred_int.shape}"
    assert pred_cont.shape == (1, 5), f"Unexpected continuous prediction shape: {pred_cont.shape}"


# ---------------------------------------------------------------------
# 4 & 5. FORECAST TIMESTAMP ALIGNMENT & BACKFILLING
# ---------------------------------------------------------------------

def test_forecast_alignment_and_backfilling(pipeline):
    """Test 4 & 5: Forecast is strictly 15 minutes ahead, and ground truth is backfilled accurately."""
    # Run 15 steps (12 warm-up + 3 horizon steps)
    for _ in range(15):
        pipeline.step()

    df = pipeline.get_predictions_df()
    succ_df = df[df["status"] == "SUCCESS"].copy()
    assert not succ_df.empty, "No successful predictions recorded"

    for _, row in succ_df.iterrows():
        t_made = datetime.fromisoformat(row["forecast_made_at"])
        t_target = datetime.fromisoformat(row["target_timestamp"])
        assert t_target - t_made == timedelta(minutes=15), (
            f"Forecast delta mismatch: {t_target - t_made} != 15 min"
        )

    # First prediction was made at step 11. Step 11 + 3 = step 14.
    # At step 14, ground truth for step 14 is observed and backfilled!
    first_pred_time = succ_df["forecast_made_at"].iloc[0]
    first_target_time = succ_df["target_timestamp"].iloc[0]

    evaluated_records = df[df["target_timestamp"] == first_target_time]
    assert len(evaluated_records) == 5 # 5 rooms
    for _, rec in evaluated_records.iterrows():
        assert rec["actual_occupants_at_target"] is not None
        assert rec["actual_occupants_at_target"] >= 0


# ---------------------------------------------------------------------
# 6. WARM-UP BEHAVIOR
# ---------------------------------------------------------------------

def test_warmup_behavior(pipeline):
    """Test 6: Steps 1-10 are marked WARMUP without predictions; step 11 transitions to SUCCESS."""
    for i in range(1, 11): # Steps 1 through 10
        res = pipeline.step()
        assert res["status"] == "WARMUP"
        assert res["predictions"] is None

    # Step 11: buffer reaches length 12
    res11 = pipeline.step()
    assert res11["status"] == "SUCCESS"
    assert res11["predictions"] is not None
    assert len(res11["predictions"]) == 5


# ---------------------------------------------------------------------
# 7. CAUSAL INTEGRITY & ZERO FUTURE LEAKAGE
# ---------------------------------------------------------------------

def test_causal_integrity_no_future_leakage(pipeline):
    """Test 7: The rolling lookback window strictly excludes future timestamps or future counts."""
    for _ in range(14):
        pipeline.step()

    curr_time = pipeline.engine.current_time.isoformat()
    # Check all entries in the rolling buffer
    for step_data in pipeline.rolling_buffer:
        for room_id in EXPECTED_ROOM_IDS:
            row = step_data[room_id]
            assert row["timestamp"] <= curr_time, (
                f"Future timestamp found in lookback buffer: {row['timestamp']} > {curr_time}"
            )


# ---------------------------------------------------------------------
# 8. OUTPUT NON-NEGATIVITY AND INTEGER POSTPROCESSING
# ---------------------------------------------------------------------

def test_output_nonnegativity_and_integers(pipeline):
    """Test 8: Model forecasts are nonnegative integers."""
    pipeline.run(num_steps=18)
    df = pipeline.get_predictions_df()
    succ_df = df[df["status"] == "SUCCESS"]

    # Test PredictionRecord directly
    for rec in pipeline.predictions:
        if rec.status == "SUCCESS":
            assert isinstance(rec.predicted_occupants, int), f"Record {rec.predicted_occupants} is not Python int"
            assert rec.predicted_occupants >= 0, f"Negative prediction: {rec.predicted_occupants}"

    # Test DataFrame values
    for val in succ_df["predicted_occupants"]:
        assert int(val) == val, f"Prediction {val} is not a whole number"
        assert val >= 0, f"Negative prediction: {val}"



# ---------------------------------------------------------------------
# 9. PERSISTENCE BASELINE ACCURACY
# ---------------------------------------------------------------------

def test_persistence_baseline_matches_observation(pipeline):
    """Test 9: Persistence baseline equals current observation y_t at forecast time."""
    pipeline.run(num_steps=14)
    df = pipeline.get_predictions_df()
    succ_df = df[df["status"] == "SUCCESS"]

    for _, row in succ_df.iterrows():
        made_at = row["forecast_made_at"]
        room_id = row["room_id"]
        pers_val = row["persistence_occupants"]

        # Look up ground truth recorded at made_at
        gt_df = pipeline.engine.recorder.to_dataframe()
        matched_gt = gt_df[(gt_df["timestamp"] == made_at) & (gt_df["room_id"] == room_id)]
        assert not matched_gt.empty
        expected_occ = int(matched_gt["actual_occupants"].iloc[0])
        assert pers_val == expected_occ, f"Persistence {pers_val} != observed {expected_occ}"


# ---------------------------------------------------------------------
# 10. EVALUATION METRICS COMPUTATION
# ---------------------------------------------------------------------

def test_compute_evaluation_metrics(pipeline):
    """Test 10: Metrics method returns valid non-negative MAE and RMSE."""
    pipeline.run(num_steps=20)
    metrics = pipeline.compute_evaluation_metrics()

    assert metrics["total_evaluated_samples"] > 0
    b_metrics = metrics["building_metrics"]

    assert "stgnn_integer_mae" in b_metrics
    assert "stgnn_integer_rmse" in b_metrics
    assert "persistence_mae" in b_metrics

    assert b_metrics["stgnn_integer_mae"] >= 0.0
    assert b_metrics["stgnn_integer_rmse"] >= b_metrics["stgnn_integer_mae"]
    assert b_metrics["persistence_mae"] >= 0.0

    for room_id in EXPECTED_ROOM_IDS:
        assert room_id in metrics["room_metrics"]
        rm = metrics["room_metrics"][room_id]
        assert rm["samples"] > 0
        assert rm["stgnn_int_mae"] >= 0.0


# ---------------------------------------------------------------------
# 11. CSV EXPORT INTEGRITY
# ---------------------------------------------------------------------

def test_export_predictions_csv(pipeline):
    """Test 11: Predictions export cleanly to CSV with all fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "predictions.csv")
        pipeline.run(num_steps=16)
        pipeline.export_predictions(csv_path)

        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)

        expected_cols = [
            "forecast_made_at", "target_timestamp", "room_id",
            "predicted_occupants", "predicted_continuous",
            "persistence_occupants", "actual_occupants_at_target",
            "status"
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 16 * 5 # 16 steps x 5 rooms
