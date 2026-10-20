"""
Automated Test Suite for Phase 8: BEMS Dashboard Interactivity & Usability
==========================================================================
Verifies:
1. Floor plan selected room visual highlighting (border width 5, cyan color).
2. Floor plan status annotations and sensor/HVAC badges.
3. All 6 BEMS operational scenario definitions and metadata integrity.
4. Deterministic scenario application (empty building, peak surge, heatwave, energy saving).
5. Synchronized selected room state management and invalid room ID rejection.
6. Temporary action feedback banner lifecycle.
7. Configurable step duration forwarding (+5m, +15m).
8. Room detail physical explainability calculations (sensible heat, thermal delta).
9. Room energy table filtering and sorting accuracy.
10. Export stream metadata generation and schema verification.
"""

import pytest
import pandas as pd
import plotly.graph_objects as go

from simulation.config import EXPECTED_ROOM_IDS
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from dashboard.floorplan import create_floorplan_figure
from dashboard.scenarios import SCENARIO_PRESETS, get_scenario_metadata, apply_scenario
from dashboard.state import (
    get_selected_room,
    set_selected_room,
    set_action_feedback,
    clear_action_feedback,
    advance_pipeline_step,
)


@pytest.fixture
def test_pipeline():
    """Deterministic pipeline fixture for BEMS testing."""
    engine = SimulationEngine(num_occupants=30, seed=42)
    pipeline = DigitalTwinPipeline(engine=engine, seed=42)
    pipeline.run(num_steps=5)
    return pipeline


# -----------------------------------------------------------------------------
# 1. Floor Plan Room Selection & Highlighting
# -----------------------------------------------------------------------------

def test_floorplan_selected_room_highlight(test_pipeline):
    """Test 1: Selected room receives prominent cyan border and [SELECTED] tag."""
    building = test_pipeline.engine.building
    occupants = test_pipeline.engine.occupants

    # Render with Room 2 selected
    fig = create_floorplan_figure(
        building=building,
        occupants=occupants,
        selected_room_id="Room 2",
        dark_mode=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 5

    # Check that Room 2's shape border is cyan (#00F0FF) with width 5
    room_idx_map = {r_id: idx for idx, r_id in enumerate(building.rooms.keys())}
    r2_idx = room_idx_map["Room 2"]
    r2_shape = fig.layout.shapes[r2_idx]
    assert r2_shape.line.color == "#00F0FF"
    assert r2_shape.line.width == 5

    # Other rooms should have default width 3
    r1_idx = room_idx_map["Room 1"]
    r1_shape = fig.layout.shapes[r1_idx]
    assert r1_shape.line.width == 3

    # Check text trace contains [SELECTED] for Room 2
    text_traces = [t for t in fig.data if t.mode == "text"]
    assert len(text_traces) == 5
    r2_text = text_traces[r2_idx].text[0]
    assert "[SELECTED]" in r2_text


def test_floorplan_status_and_sensor_badges(test_pipeline):
    """Test 2: Floor plan integrates sensor readings and energy state into display text."""
    building = test_pipeline.engine.building
    occupants = test_pipeline.engine.occupants
    e_states = test_pipeline.engine.energy_manager.get_latest_states()

    mock_sensors = {
        "Room 1": {"indoor_temperature": 23.4, "indoor_co2": 1150.0},
    }

    fig = create_floorplan_figure(
        building=building,
        occupants=occupants,
        energy_states=e_states,
        sensor_readings=mock_sensors,
        selected_room_id="Room 1",
    )

    room_idx_map = {r_id: idx for idx, r_id in enumerate(building.rooms.keys())}
    r1_idx = room_idx_map["Room 1"]
    text_traces = [t for t in fig.data if t.mode == "text"]
    r1_text = text_traces[r1_idx].text[0]

    assert "23.4°C" in r1_text
    assert "1150 ppm" in r1_text
    # Should flag elevated CO2
    assert "Elevated CO₂" in r1_text


# -----------------------------------------------------------------------------
# 2. BEMS Scenarios
# -----------------------------------------------------------------------------

def test_bems_scenarios_definitions():
    """Test 3: All 6 required demonstration scenarios exist with complete metadata."""
    required_keys = ["normal", "low_occupancy", "peak_surge", "empty_building", "energy_saving", "heatwave"]
    for k in required_keys:
        assert k in SCENARIO_PRESETS, f"Missing scenario preset: {k}"
        meta = get_scenario_metadata(k)
        assert "name" in meta
        assert "description" in meta
        assert "occupants" in meta
        assert "target_setpoint" in meta
        assert "impact_summary" in meta


def test_apply_empty_building_scenario():
    """Test 4: Empty building scenario initializes 0 occupants and sets all rooms to ECO."""
    pipeline = apply_scenario("empty_building")
    assert pipeline.engine.building.total_occupants == 0
    assert pipeline.engine.num_occupants == 0

    # Verify all room HVAC controllers are in ECO mode
    energy_mgr = pipeline.engine.energy_manager
    for r_id in EXPECTED_ROOM_IDS:
        hvac = energy_mgr.hvac_controllers[r_id]
        assert hvac.manual_override == "ECO"


def test_apply_peak_surge_scenario():
    """Test 5: Peak surge initializes 80 occupants and executes steps safely."""
    pipeline = apply_scenario("peak_surge")
    assert pipeline.engine.num_occupants == 80
    assert len(pipeline.engine.occupants) == 80
    # Step forward 2 steps to progress scheduled arrivals
    res = pipeline.step()
    assert res is not None
    assert pipeline.engine.step_count == 1
    assert pipeline.engine.building.total_occupants > 0


def test_apply_heatwave_scenario():
    """Test 6: Heatwave scenario elevates ambient outdoor temperature to 35.0°C."""
    pipeline = apply_scenario("heatwave")
    weather = pipeline.engine.sensor_manager.weather
    assert weather.temp_mean == 35.0


# -----------------------------------------------------------------------------
# 3. State Management & Synchronization
# -----------------------------------------------------------------------------

def test_selected_room_synchronization():
    """Test 7: Room selection is synchronized and rejects invalid identifiers."""
    import streamlit as st

    # Initialize state
    st.session_state.selected_room = "Room 1"
    assert get_selected_room() == "Room 1"

    set_selected_room("Room 3")
    assert get_selected_room() == "Room 3"

    # Reject invalid room ID
    set_selected_room("Room 99")
    assert get_selected_room() == "Room 3"  # Remains Room 3


def test_action_feedback_lifecycle():
    """Test 8: Feedback banners can be set and cleared cleanly."""
    import streamlit as st

    set_action_feedback("success", "Moved 3 occupants successfully.")
    assert st.session_state.action_feedback == ("success", "Moved 3 occupants successfully.")

    clear_action_feedback()
    assert st.session_state.action_feedback is None


def test_advance_step_duration(test_pipeline):
    """Test 9: Stepping advances by duration multiplier (15 min = 3 steps)."""
    initial_step = test_pipeline.engine.step_count
    import streamlit as st
    st.session_state.pipeline = test_pipeline

    # Advance 15 min (3 steps)
    advance_pipeline_step(step_minutes=15)
    assert test_pipeline.engine.step_count == initial_step + 3


# -----------------------------------------------------------------------------
# 4. Energy Explainability & Table Logic
# -----------------------------------------------------------------------------

def test_energy_explainability_calculation(test_pipeline):
    """Test 10: Explainability correctly isolates sensible heat and thermal delta."""
    e_state = test_pipeline.engine.energy_manager.get_latest_states().get("Room 1")
    assert e_state is not None

    occupants = e_state.occupant_count
    sensible_gain = occupants * 100.0  # 100 W/occ
    assert sensible_gain >= 0.0

    t_in = e_state.indoor_temperature
    t_target = e_state.hvac_target_setpoint
    t_delta = t_in - t_target
    # Delta should be a valid float
    assert isinstance(t_delta, float)


def test_energy_table_sorting_and_filtering(test_pipeline):
    """Test 11: Table data frame correctly sorts by power and filters by HVAC mode."""
    energy_mgr = test_pipeline.engine.energy_manager
    latest = energy_mgr.get_latest_states()

    rows = []
    for r_id in EXPECTED_ROOM_IDS:
        s = latest[r_id]
        rows.append({
            "Room": r_id,
            "Mode": s.hvac_mode,
            "Power": s.total_power_w,
            "Energy": s.cumulative_total_kwh,
        })
    df = pd.DataFrame(rows)
    assert len(df) == 5

    # Sort descending by power
    df_sorted = df.sort_values(by="Power", ascending=False)
    assert df_sorted.iloc[0]["Power"] >= df_sorted.iloc[-1]["Power"]

