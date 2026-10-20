"""
Phase 8.1 Automated Test Suite: Interactive 2D Floor Plan and Room Selection
===========================================================================
Verifies:
1. All 5 configured rooms render in correct geometrical layout.
2. Room names, dimensions, and capacities match configuration strictly.
3. Room selection updates selected room state and floor plan highlighting.
4. Selection persists across ordinary state accesses.
5. All 6 multi-physics overlays (Occupancy, Temp, CO2, Lux, HVAC Power, Total Power)
   update visual presentation and legends cleanly without mutating simulation.
6. Missing/unavailable telemetry values are handled gracefully (no crashes, clean N/A).
7. Room occupancy and power values in floor plan match underlying simulation state.
8. Invalid building geometry raises clear, informative ValueError.
9. Manual people movement and entrance queue rendering remain topologically accurate.
10. Rendering floor plans causes zero energy meter mutations or double-counting.
11. STGNN inference buffer and prediction horizons are completely unaffected by rendering.
12. Legend metadata is complete with units, physical descriptions, and color ranges.
"""

import pytest
import pandas as pd
import plotly.graph_objects as go

from simulation.config import load_building_config, EXPECTED_ROOM_IDS
from simulation.building import Building
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from dashboard.floorplan import (
    create_floorplan_figure,
    validate_building_geometry,
    get_utilization_color,
    get_utilization_border_color,
    get_overlay_color_and_text,
    get_overlay_legend_info,
    OVERLAY_OPTIONS,
)
from dashboard.state import (
    get_selected_room,
    set_selected_room,
    get_selected_overlay,
    set_selected_overlay,
)


@pytest.fixture
def sim_pipeline():
    """Deterministic simulation pipeline for Phase 8.1 testing."""
    engine = SimulationEngine(num_occupants=35, seed=42)
    pipeline = DigitalTwinPipeline(engine=engine, seed=42)
    pipeline.run(num_steps=6)
    return pipeline


# -----------------------------------------------------------------------------
# 1. Geometry & Room Layout Rendering
# -----------------------------------------------------------------------------

def test_all_five_configured_rooms_render(sim_pipeline):
    """Requirement 1: All 5 configured rooms render with bounding polygons."""
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants

    fig = create_floorplan_figure(
        building=building,
        occupants=occupants,
        show_occupants=True,
        dark_mode=True,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 5, f"Expected 5 room shapes, found {len(fig.layout.shapes)}"

    # Check axis properties for 1:1 isometric aspect ratio
    assert fig.layout.xaxis.scaleratio == 1
    assert fig.layout.xaxis.scaleanchor == "y"


def test_room_names_and_capacities_match_config(sim_pipeline):
    """Requirement 2: Rendered room labels and capacities strictly match configuration."""
    building = sim_pipeline.engine.building
    cfg = load_building_config()

    fig = create_floorplan_figure(building=building, show_occupants=False)
    text_traces = [t for t in fig.data if t.mode == "text"]
    assert len(text_traces) == 5

    for rc in cfg.rooms:
        r_id = rc.id
        room_idx = list(building.rooms.keys()).index(r_id)
        room_text = text_traces[room_idx].text[0]

        # Room ID and Name must appear
        assert r_id in room_text
        assert rc.name in room_text
        assert str(rc.capacity) in room_text


# -----------------------------------------------------------------------------
# 2. Room Selection & Synchronization
# -----------------------------------------------------------------------------

def test_room_selection_highlight(sim_pipeline):
    """Requirement 3: Selected room receives prominent cyan highlight border."""
    building = sim_pipeline.engine.building

    # Select Room 4
    fig = create_floorplan_figure(building=building, selected_room_id="Room 4")

    room_keys = list(building.rooms.keys())
    r4_idx = room_keys.index("Room 4")
    r4_shape = fig.layout.shapes[r4_idx]

    # Selected room border must be #00F0FF with width 5
    assert r4_shape.line.color == "#00F0FF"
    assert r4_shape.line.width == 5

    # Non-selected rooms must have width 3
    r1_idx = room_keys.index("Room 1")
    assert fig.layout.shapes[r1_idx].line.width == 3

    # Text annotation must include [SELECTED]
    text_traces = [t for t in fig.data if t.mode == "text"]
    assert "[SELECTED]" in text_traces[r4_idx].text[0]
    assert "[SELECTED]" not in text_traces[r1_idx].text[0]


def test_selection_persists_across_state(sim_pipeline):
    """Requirement 4: Room and overlay selection persist in state manager."""
    set_selected_room("Room 5")
    assert get_selected_room() == "Room 5"

    set_selected_overlay("Indoor Temperature")
    assert get_selected_overlay() == "Indoor Temperature"

    # Reset to default
    set_selected_room("Room 1")
    set_selected_overlay("Occupancy")
    assert get_selected_room() == "Room 1"
    assert get_selected_overlay() == "Occupancy"


# -----------------------------------------------------------------------------
# 3. Multi-Physics Overlays & Pure Presentation
# -----------------------------------------------------------------------------

def test_all_overlays_render_cleanly(sim_pipeline):
    """Requirement 5: All 6 overlays render valid figures without modifying state."""
    building = sim_pipeline.engine.building
    e_states = sim_pipeline.engine.energy_manager.get_latest_states()

    sensor_df = sim_pipeline.engine.get_sensor_readings_df()
    latest_sensors = {r: sensor_df[sensor_df["room_id"] == r].iloc[-1].to_dict() for r in EXPECTED_ROOM_IDS}

    for overlay in OVERLAY_OPTIONS:
        fig = create_floorplan_figure(
            building=building,
            energy_states=e_states,
            sensor_readings=latest_sensors,
            active_overlay=overlay,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.shapes) == 5
        assert overlay in fig.layout.title.text

        # Verify legend metadata is populated
        leg = get_overlay_legend_info(overlay)
        assert "name" in leg
        assert "unit" in leg
        assert len(leg["ranges"]) >= 3


def test_overlay_switching_preserves_simulation(sim_pipeline):
    """Requirement 5b: Switching overlays does NOT advance clock, step count, or energy."""
    initial_step = sim_pipeline.engine.step_count
    initial_time = sim_pipeline.engine.current_time
    initial_energy = sim_pipeline.engine.energy_manager.get_building_kpis()["cumulative_total_kwh"]

    building = sim_pipeline.engine.building

    # Render all 6 overlays repeatedly
    for overlay in OVERLAY_OPTIONS:
        create_floorplan_figure(building=building, active_overlay=overlay)

    # Invariants must hold strictly
    assert sim_pipeline.engine.step_count == initial_step
    assert sim_pipeline.engine.current_time == initial_time
    assert sim_pipeline.engine.energy_manager.get_building_kpis()["cumulative_total_kwh"] == initial_energy


# -----------------------------------------------------------------------------
# 4. Missing / Unavailable Data Handling
# -----------------------------------------------------------------------------

def test_missing_telemetry_graceful_handling(sim_pipeline):
    """Requirement 6: Missing sensor or energy data produces clean N/A rather than crashing."""
    building = sim_pipeline.engine.building
    r1 = building.get_room("Room 1")

    # Call overlay helper with empty sensor and energy dictionaries
    for ov in OVERLAY_OPTIONS:
        fill, border, metric_str, badge, tip = get_overlay_color_and_text(
            room_id="Room 1",
            room=r1,
            overlay=ov,
            energy_states={},
            sensor_readings={},
        )
        assert fill is not None
        assert border is not None
        assert metric_str is not None
        if ov != "Occupancy":
            # Must reflect Unavailable / Standby / No Reading
            assert any(term in metric_str for term in ["Unavailable", "Standby", "N/A"])


# -----------------------------------------------------------------------------
# 5. Simulation State Parity
# -----------------------------------------------------------------------------

def test_occupancy_and_power_match_simulation(sim_pipeline):
    """Requirement 7: Values in floor plan strictly reflect underlying simulation."""
    building = sim_pipeline.engine.building
    e_states = sim_pipeline.engine.energy_manager.get_latest_states()

    for r_id in EXPECTED_ROOM_IDS:
        rm = building.get_room(r_id)
        e_st = e_states.get(r_id)

        _, _, metric_str, _, tip = get_overlay_color_and_text(
            room_id=r_id,
            room=rm,
            overlay="Occupancy",
            energy_states=e_states,
        )
        assert f"{rm.occupant_count} / {rm.capacity}" in metric_str

        _, _, pwr_str, _, _ = get_overlay_color_and_text(
            room_id=r_id,
            room=rm,
            overlay="Total Room Power",
            energy_states=e_states,
        )
        expected_w = f"{e_st.total_power_w:.0f} W"
        assert expected_w in pwr_str


# -----------------------------------------------------------------------------
# 6. Geometry Validation & Error Handling
# -----------------------------------------------------------------------------

def test_invalid_geometry_raises_value_error():
    """Requirement 8: Invalid room geometry raises clean ValueError."""
    cfg = load_building_config()
    b = Building(cfg)

    # Corrupt a room's width to negative
    b.rooms["Room 1"].width = -10.0
    with pytest.raises(ValueError, match="non-positive width"):
        validate_building_geometry(b)

    # Restore and corrupt height
    b.rooms["Room 1"].width = 14.0
    b.rooms["Room 2"].height = 0.0
    with pytest.raises(ValueError, match="non-positive height"):
        validate_building_geometry(b)

    # Restore and corrupt coordinates
    b.rooms["Room 2"].height = 10.0
    b.rooms["Room 3"].x = -5.0
    with pytest.raises(ValueError, match="negative coordinates"):
        validate_building_geometry(b)


# -----------------------------------------------------------------------------
# 7. Entrance Queue & Topological Doorway Connections
# -----------------------------------------------------------------------------

def test_entrance_queue_and_doorways(sim_pipeline):
    """Requirement 9: Doorways and entrance queue render accurately."""
    building = sim_pipeline.engine.building

    # Render with 12 queued entrants
    fig = create_floorplan_figure(
        building=building,
        entrance_queue_count=12,
        show_occupants=False,
    )

    trace_names = [t.name for t in fig.data if t.name]
    assert "Interior Doors" in trace_names
    assert "Exterior Exit" in trace_names
    assert "Entrance Queue" in trace_names

    # Check queue text
    q_trace = next(t for t in fig.data if t.name == "Entrance Queue")
    assert "WAITING QUEUE: 12" in q_trace.text[0]


# -----------------------------------------------------------------------------
# 8. STGNN Pipeline Invariance
# -----------------------------------------------------------------------------

def test_stgnn_pipeline_unaffected_by_rendering(sim_pipeline):
    """Requirement 11: Floor plan rendering leaves rolling buffer and STGNN intact."""
    initial_buffer_len = len(sim_pipeline.rolling_buffer)
    initial_predictions_len = len(sim_pipeline.predictions)

    # Render floor plans
    create_floorplan_figure(building=sim_pipeline.engine.building)

    assert len(sim_pipeline.rolling_buffer) == initial_buffer_len
    assert len(sim_pipeline.predictions) == initial_predictions_len

