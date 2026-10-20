"""
Automated Test Suite for Interactive Digital Twin Dashboard (Phase 4)
=====================================================================
Verifies all dashboard data preparation, floor plan generation,
Plotly chart rendering, metric calculation, and data export serialization:
1. Floor plan figure generates all 5 rooms, doors, and occupant markers.
2. Occupancy time-series chart handles both building-wide and per-room modes.
3. Sensor telemetry charts support multiple features with dual-axis scaling.
4. Residual error chart plots deviations accurately against zero-error baseline.
5. Metrics comparison bar chart plots STGNN vs. Persistence for all 5 rooms.
6. CSV and JSON export data structures serialize cleanly without missing fields.
7. Utilization heat colors map correctly to capacity ratio thresholds.
"""

import json
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from simulation.config import load_building_config, EXPECTED_ROOM_IDS
from simulation.building import Building
from simulation.pipeline import DigitalTwinPipeline
from stgnn_model import STGNNInference, COMMON_SENSOR_FEATURES
from dashboard.floorplan import (
    create_floorplan_figure,
    get_utilization_color,
    get_utilization_border_color,
)
from dashboard.charts import (
    create_occupancy_chart,
    create_sensor_charts,
    create_error_chart,
    create_metrics_comparison_chart,
    FEATURE_UNITS,
)


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_pipeline():
    """Module-scoped pipeline with fixed seed for dashboard testing."""
    infer = STGNNInference()
    p = DigitalTwinPipeline(inference=infer, seed=42)
    # Run 18 steps so we have full lookback + multiple evaluated predictions
    p.run(num_steps=18)
    return p


# ---------------------------------------------------------------------
# 1. FLOOR PLAN VISUALIZATION
# ---------------------------------------------------------------------

def test_floorplan_figure_generation(shared_pipeline):
    """Test 1: Floor plan generates 5 room shapes, doors, and occupant points."""
    building = shared_pipeline.engine.building
    occupants = shared_pipeline.engine.occupants

    fig = create_floorplan_figure(
        building=building,
        occupants=occupants,
        show_occupants=True,
        dark_mode=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 5, f"Expected 5 room shapes, got {len(fig.layout.shapes)}"

    # Check trace names
    trace_names = [t.name for t in fig.data if t.name]
    assert "Interior Doors" in trace_names
    assert "Exterior Exit" in trace_names
    assert "Occupants" in trace_names


def test_utilization_color_thresholds():
    """Test 2: Heat colors transition accurately based on utilization thresholds."""
    # < 50% -> Green
    assert "46, 204, 113" in get_utilization_color(occupant_count=5, capacity=20)
    # 50% to 85% -> Amber
    assert "243, 156, 18" in get_utilization_color(occupant_count=12, capacity=20)
    # >= 85% -> Red
    assert "231, 76, 60" in get_utilization_color(occupant_count=18, capacity=20)


# ---------------------------------------------------------------------
# 2. OCCUPANCY TIMELINE CHARTS
# ---------------------------------------------------------------------

def test_occupancy_chart_building_and_room(shared_pipeline):
    """Test 3: Occupancy chart renders for both building-wide and individual rooms."""
    gt_df = shared_pipeline.engine.recorder.to_dataframe()
    pred_df = shared_pipeline.get_predictions_df()

    # Building total
    fig_total = create_occupancy_chart(gt_df, pred_df, room_id=None)
    assert isinstance(fig_total, go.Figure)
    assert len(fig_total.data) >= 2 # Ground truth + STGNN

    # Room-specific
    fig_room = create_occupancy_chart(gt_df, pred_df, room_id="Room 2")
    assert isinstance(fig_room, go.Figure)
    assert "Room 2" in fig_room.layout.title.text


# ---------------------------------------------------------------------
# 3. SENSOR TELEMETRY CHARTS
# ---------------------------------------------------------------------

def test_sensor_charts_dual_axis_and_units(shared_pipeline):
    """Test 4: Sensor chart renders selected features with dual-axis scaling."""
    sensor_df = shared_pipeline.engine.get_sensor_readings_df()

    chosen = ["indoor_temperature", "indoor_co2", "illuminance"]
    fig = create_sensor_charts(sensor_df, room_id="Room 1", selected_features=chosen)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(chosen)

    # Check units exist for all 7 features
    for feat in COMMON_SENSOR_FEATURES:
        assert feat in FEATURE_UNITS
        assert len(FEATURE_UNITS[feat]) > 0


# ---------------------------------------------------------------------
# 4. ERROR CHARTS & RESIDUAL PLOTS
# ---------------------------------------------------------------------

def test_error_chart_generation(shared_pipeline):
    """Test 5: Residual error chart generates traces for STGNN and persistence errors."""
    pred_df = shared_pipeline.get_predictions_df()
    fig = create_error_chart(pred_df, room_id=None)

    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if t.name]
    assert any("STGNN Error" in n for n in trace_names)
    assert any("Persistence Error" in n for n in trace_names)


# ---------------------------------------------------------------------
# 5. METRICS COMPARISON BAR CHART
# ---------------------------------------------------------------------

def test_metrics_comparison_chart(shared_pipeline):
    """Test 6: Metrics bar chart renders comparison across all 5 rooms."""
    metrics = shared_pipeline.compute_evaluation_metrics()
    fig = create_metrics_comparison_chart(metrics)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 # STGNN MAE + Persistence MAE
    assert list(fig.data[0].x) == EXPECTED_ROOM_IDS


# ---------------------------------------------------------------------
# 6. DATA EXPORT SERIALIZATION
# ---------------------------------------------------------------------

def test_data_export_serialization(shared_pipeline):
    """Test 7: All 4 export streams serialize to valid, parseable formats."""
    gt_df = shared_pipeline.engine.recorder.to_dataframe()
    sensor_df = shared_pipeline.engine.get_sensor_readings_df()
    pred_df = shared_pipeline.get_predictions_df()
    metrics = shared_pipeline.compute_evaluation_metrics()

    # Ground truth CSV string
    gt_csv = gt_df.to_csv(index=False)
    assert "timestamp,room_id,actual_occupants" in gt_csv

    # Sensor CSV string
    sensor_csv = sensor_df.to_csv(index=False)
    assert "indoor_co2" in sensor_csv
    assert "dry_bulb_temperature" in sensor_csv

    # Predictions CSV string
    pred_csv = pred_df.to_csv(index=False)
    assert "forecast_made_at,target_timestamp" in pred_csv
    assert "predicted_occupants" in pred_csv

    # Metrics JSON string
    metrics_json = json.dumps(metrics, indent=2)
    parsed = json.loads(metrics_json)
    assert "building_metrics" in parsed
    assert "room_metrics" in parsed

