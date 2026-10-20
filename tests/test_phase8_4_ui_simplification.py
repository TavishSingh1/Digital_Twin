"""
Phase 8.4 Automated Test Suite: UI Simplification, Stability & Information Hierarchy
===================================================================================
Verifies:
1. Simplified Plotly blueprint floor plan rendering (clean annotations, no clutter on non-selected rooms).
2. Selected room highlighting and selective telemetry display.
3. Live Canvas HTML stability (safe coordinate clamping, scheduleFrame, beforeunload, ResizeObserver).
4. Missing, invalid, or NaN occupant coordinate resilience without crashes.
5. Canvas pause, resume, speed, and replay script integrity.
6. Absolute simulation invariance: rendering floor plan or canvas causes zero clock or energy advancement.
7. Occupant selection synchronization remains regression-free.
"""

import pytest
import math
import json
import plotly.graph_objects as go

from simulation.config import EXPECTED_ROOM_IDS, load_building_config
from simulation.building import Building
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from simulation.occupant import Occupant
from simulation.schedules import OccupantSchedule
from dashboard.animation import AnimationManager, MovementEvent
from dashboard.floorplan import (
    create_floorplan_figure,
    validate_building_geometry,
    get_overlay_color_and_text,
    OVERLAY_OPTIONS,
)
from dashboard.live_canvas import (
    generate_live_digital_twin_html,
    render_live_floorplan_component,
)


@pytest.fixture
def sim_pipeline():
    """Reproducible simulation pipeline for Phase 8.4 testing."""
    engine = SimulationEngine(num_occupants=25, seed=42)
    pipeline = DigitalTwinPipeline(engine=engine, seed=42)
    pipeline.run(num_steps=6)
    return pipeline


@pytest.fixture
def anim_mgr():
    """AnimationManager instance for testing."""
    return AnimationManager()


# -----------------------------------------------------------------------------
# 1. Simplified Blueprint Floor Plan Tests
# -----------------------------------------------------------------------------

def test_simplified_blueprint_rendering(sim_pipeline, anim_mgr):
    """Verifies that the blueprint displays clean room outlines, names, and occupancy counts."""
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants
    e_states = sim_pipeline.engine.energy_manager.get_latest_states()

    fig = create_floorplan_figure(
        building=building,
        occupants=occupants,
        selected_room_id="Room 2",
        dark_mode=True,
        animation_manager=anim_mgr,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 5

    # Check text annotations
    text_traces = [t for t in fig.data if t.mode == "text"]
    assert len(text_traces) == 5

    room_keys = list(building.rooms.keys())
    for r_id in room_keys:
        idx = room_keys.index(r_id)
        r_text = text_traces[idx].text[0]
        rm = building.get_room(r_id)

        # Must display room ID, name, and capacity
        assert r_id in r_text
        assert rm.name in r_text
        assert str(rm.capacity) in r_text

        # For non-selected rooms, verify absence of persistent secondary telemetry
        if r_id != "Room 2":
            assert "[SELECTED]" not in r_text
            assert "Status:" not in r_text

    # Selected room (Room 2) must have [SELECTED]
    r2_idx = room_keys.index("Room 2")
    assert "[SELECTED]" in text_traces[r2_idx].text[0]


def test_selected_room_highlight_border(sim_pipeline):
    """Verifies subtle cyan glowing border on selected room."""
    building = sim_pipeline.engine.building
    fig = create_floorplan_figure(building=building, selected_room_id="Room 3")

    room_keys = list(building.rooms.keys())
    r3_idx = room_keys.index("Room 3")
    r3_shape = fig.layout.shapes[r3_idx]

    assert r3_shape.line.color == "#00F0FF"
    assert r3_shape.line.width == 5

    # Other room shapes must have default width 3
    r1_idx = room_keys.index("Room 1")
    assert fig.layout.shapes[r1_idx].line.width == 3


# -----------------------------------------------------------------------------
# 2. Selected Room Details & Telemetry Tests
# -----------------------------------------------------------------------------

def test_selected_room_details_telemetry_availability(sim_pipeline):
    """Verifies that all 7 sensor dimensions and energy state are available for selected room."""
    selected_room = "Room 1"
    sensor_df = sim_pipeline.engine.get_sensor_readings_df()
    assert not sensor_df.empty

    rdf = sensor_df[sensor_df["room_id"] == selected_room]
    assert not rdf.empty
    last_sensor = rdf.iloc[-1].to_dict()

    # All 7 physical variables must be present and valid numbers
    assert "indoor_temperature" in last_sensor and isinstance(last_sensor["indoor_temperature"], (int, float))
    assert "indoor_co2" in last_sensor and isinstance(last_sensor["indoor_co2"], (int, float))
    assert "indoor_relative_humidity" in last_sensor and isinstance(last_sensor["indoor_relative_humidity"], (int, float))
    assert "illuminance" in last_sensor and isinstance(last_sensor["illuminance"], (int, float))
    assert "sound_pressure_level" in last_sensor and isinstance(last_sensor["sound_pressure_level"], (int, float))
    assert "dry_bulb_temperature" in last_sensor and isinstance(last_sensor["dry_bulb_temperature"], (int, float))
    assert "outdoor_relative_humidity" in last_sensor and isinstance(last_sensor["outdoor_relative_humidity"], (int, float))

    # Energy state
    e_states = sim_pipeline.engine.energy_manager.get_latest_states()
    e_st = e_states.get(selected_room)
    assert e_st is not None
    assert hasattr(e_st, "hvac_mode")
    assert hasattr(e_st, "total_power_w")
    assert hasattr(e_st, "cumulative_total_kwh")


# -----------------------------------------------------------------------------
# 3. Live Canvas HTML Stability & Cleanup Tests
# -----------------------------------------------------------------------------

def test_canvas_html_structure_and_controls(sim_pipeline, anim_mgr):
    """Verifies that generated Canvas HTML contains proper lifecycle guards and controls."""
    html = generate_live_digital_twin_html(
        building=sim_pipeline.engine.building,
        occupants=sim_pipeline.engine.occupants,
        anim_mgr=anim_mgr,
        selected_room_id="Room 1",
        active_overlay="Occupancy",
        height=490,
    )

    # Core canvas elements
    assert "<canvas id=\"canvas\"" in html
    assert "id=\"dt-container\"" in html
    assert "490px" in html

    # Lifecycle guards
    assert "scheduleFrame()" in html
    assert "cancelAnimationFrame" in html
    assert "beforeunload" in html
    assert "ResizeObserver" in html

    # Numerical clamping helper
    assert "function clamp(" in html

    # Controls
    assert "btn-play-pause" in html
    assert "btn-replay" in html
    assert "scrub-slider" in html
    assert "spd-btn" in html


def test_canvas_missing_and_invalid_coordinates_resilience(sim_pipeline, anim_mgr):
    """Verifies that Canvas HTML generation handles NaN, None, and out-of-bound occupant coordinates safely."""
    building = sim_pipeline.engine.building
    # Inject corrupted/extreme coordinates into real occupants
    corrupted_occupants = dict(sim_pipeline.engine.occupants)
    pids = [pid for pid, o in corrupted_occupants.items() if o.is_inside_building]
    assert len(pids) >= 3

    corrupted_occupants[pids[0]].position = (float("nan"), 10.0)
    corrupted_occupants[pids[1]].position = (-999.0, 999.0)
    corrupted_occupants[pids[2]].position = (100.0, -50.0)

    html = generate_live_digital_twin_html(
        building=building,
        occupants=corrupted_occupants,
        anim_mgr=anim_mgr,
        selected_room_id="Room 1",
    )

    # Must generate valid HTML without raising errors
    assert "<canvas" in html
    assert pids[0] in html or pids[1] in html or pids[2] in html


def test_canvas_degenerate_waypoints_handling(sim_pipeline, anim_mgr):
    """Verifies that empty or single-point movement event waypoints do not cause crashes."""
    building = sim_pipeline.engine.building
    anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T10:00:00",
        person_ids=["P0001"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    # Manually truncate waypoints to test degenerate edge cases
    anim_mgr.events[-1].waypoints = [(7.0, 3.5)]

    html = generate_live_digital_twin_html(
        building=building,
        occupants=sim_pipeline.engine.occupants,
        anim_mgr=anim_mgr,
    )
    assert "<canvas" in html


# -----------------------------------------------------------------------------
# 4. Simulation Invariance Tests
# -----------------------------------------------------------------------------

def test_rendering_does_not_advance_simulation_or_energy(sim_pipeline, anim_mgr):
    """Verifies that rendering Blueprint and Canvas 20 times causes 0 simulation state mutations."""
    engine = sim_pipeline.engine
    clock_start = engine.current_time
    step_start = engine.step_count
    kwh_start = sum(s.cumulative_total_kwh for s in engine.energy_manager.get_latest_states().values())

    building = engine.building
    occupants = engine.occupants

    for _ in range(20):
        # Render Blueprint
        create_floorplan_figure(
            building=building,
            occupants=occupants,
            animation_manager=anim_mgr,
            active_overlay="Occupancy",
        )
        # Render Canvas
        generate_live_digital_twin_html(
            building=building,
            occupants=occupants,
            anim_mgr=anim_mgr,
            active_overlay="Indoor Temperature",
        )

    # Strict equality assertions
    assert engine.current_time == clock_start
    assert engine.step_count == step_start
    kwh_end = sum(s.cumulative_total_kwh for s in engine.energy_manager.get_latest_states().values())
    assert kwh_start == pytest.approx(kwh_end, abs=1e-9)


# -----------------------------------------------------------------------------
# 5. Occupant Selection Synchronization Non-Regression
# -----------------------------------------------------------------------------

def test_occupant_selection_synchronization_integrity(sim_pipeline, anim_mgr):
    """Verifies that occupant selection updates anim_mgr without corrupting state."""
    inside_pids = [
        pid for pid, occ in sim_pipeline.engine.occupants.items()
        if occ.is_inside_building
    ]
    assert len(inside_pids) > 0
    chosen = inside_pids[0]

    anim_mgr.selected_occupant_id = chosen
    valid = anim_mgr.validate_selected_occupant(sim_pipeline.engine.occupants)
    assert valid == chosen

    # Clearing selection sets None
    anim_mgr.selected_occupant_id = None
    assert anim_mgr.validate_selected_occupant(sim_pipeline.engine.occupants) is None