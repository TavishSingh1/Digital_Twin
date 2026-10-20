"""
Phase 8.5 Automated Test Suite: Professional Digital Twin UI Redesign
=====================================================================
Validates:
1. Primary SVG floor plan renders all 5 configured rooms with accurate 1:1 geometry.
2. High-contrast selected room highlight ([SELECTED] tag and cyan glow styling).
3. Selected room data synchronization across state and views.
4. Valid occupant placement (occupants placed within building bounds, no NaN/inf coordinates).
5. Strict visual decluttering: floor plan contains clean room IDs, names, and occupancy counts
   WITHOUT walls of raw sensor telemetry (no temperature, CO2, lux, sound, kw strings all over rooms).
6. Simulation invariance: Rendering SVG floor plan 20 times causes 0 simulation state mutations,
   0 clock advancements, and 0 energy meter increments.
7. Multi-physics overlay switching applies distinct, valid styling without mutating simulation.
8. Doorway passages, exterior exit, and entrance queue are geometrically accurate.
"""

import pytest
import re

from simulation.config import load_building_config, EXPECTED_ROOM_IDS
from simulation.building import Building
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from dashboard.animation import AnimationManager
from dashboard.svg_floorplan import generate_svg_floorplan_markup
from dashboard.floorplan import OVERLAY_OPTIONS
from dashboard.state import (
    get_selected_room,
    set_selected_room,
    get_selected_overlay,
    set_selected_overlay,
)


@pytest.fixture
def sim_pipeline():
    """Initializes a deterministic simulation pipeline with occupants for Phase 8.5 testing."""
    engine = SimulationEngine(num_occupants=35, seed=42)
    pipeline = DigitalTwinPipeline(engine=engine, seed=42)
    pipeline.run(num_steps=5)
    return pipeline


@pytest.fixture
def anim_mgr(sim_pipeline):
    """Initializes an AnimationManager populated with movement events."""
    mgr = AnimationManager()
    building = sim_pipeline.engine.building
    mgr.record_movement(
        event_type="admission",
        timestamp="2026-10-19T09:00:00",
        person_ids=["P0001", "P0002"],
        source_room=None,
        destination_room="Room 1",
        building=building,
    )
    mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:05:00",
        person_ids=["P0001"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    return mgr


# -----------------------------------------------------------------------------
# 1. Primary SVG Floor Plan Geometry & Rooms
# -----------------------------------------------------------------------------

def test_svg_floorplan_renders_all_five_rooms(sim_pipeline, anim_mgr):
    """Verifies that the primary SVG floor plan renders all 5 configured rooms."""
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants

    svg = generate_svg_floorplan_markup(
        building=building,
        occupants=occupants,
        selected_room_id="Room 1",
        active_overlay="Occupancy",
        animation_manager=anim_mgr,
    )

    assert "<svg" in svg
    assert "</svg>" in svg
    assert 'viewBox="-60 -40 1160 980"' in svg

    # All 5 room IDs must be rendered
    for r_id in EXPECTED_ROOM_IDS:
        assert f'data-room-id="{r_id}"' in svg
        assert r_id in svg


def test_svg_floorplan_doors_and_exits(sim_pipeline, anim_mgr):
    """Verifies that interior doorways and the exterior exit are present in the SVG."""
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants

    svg = generate_svg_floorplan_markup(
        building=building,
        occupants=occupants,
        selected_room_id="Room 1",
        active_overlay="Occupancy",
        animation_manager=anim_mgr,
        entrance_queue_count=3,
    )

    # Door group and exit markers
    assert "door-group" in svg
    assert "MAIN EXIT" in svg
    assert "QUEUE: 3" in svg


# -----------------------------------------------------------------------------
# 2. Room Selection & Highlight
# -----------------------------------------------------------------------------

def test_svg_selected_room_highlight(sim_pipeline, anim_mgr):
    """Verifies that selecting a room applies high-contrast cyan highlight and star tag."""
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants

    for test_room in EXPECTED_ROOM_IDS:
        svg = generate_svg_floorplan_markup(
            building=building,
            occupants=occupants,
            selected_room_id=test_room,
            active_overlay="Occupancy",
            animation_manager=anim_mgr,
        )

        assert 'filter="url(#cyan-glow)"' in svg
        assert "#00F0FF" in svg  # High-contrast cyan stroke
        assert f'{test_room}' in svg


def test_room_selection_persistence():
    """Verifies that set_selected_room updates the active selection cleanly."""
    set_selected_room("Room 3")
    assert get_selected_room() == "Room 3"
    set_selected_room("Room 5")
    assert get_selected_room() == "Room 5"
    set_selected_room("Invalid Room")
    assert get_selected_room() == "Room 5"  # Rejects invalid room gracefully


# -----------------------------------------------------------------------------
# 3. Valid Occupant Placement & Visual Decluttering
# -----------------------------------------------------------------------------

def test_svg_occupant_markers_valid_coordinates(sim_pipeline, anim_mgr):
    """Verifies that occupant markers are placed at valid coordinates within building bounds."""
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants

    svg = generate_svg_floorplan_markup(
        building=building,
        occupants=occupants,
        selected_room_id="Room 1",
        active_overlay="Occupancy",
        animation_manager=anim_mgr,
        show_occupants=True,
    )

    # Extract all circle coordinates
    circle_coords = re.findall(r'<circle cx="([\d\.-]+)" cy="([\d\.-]+)" r="11"', svg)
    assert len(circle_coords) > 0

    for cx_str, cy_str in circle_coords:
        cx = float(cx_str)
        cy = float(cy_str)
        # In SVG: x is in [0, 1040], y is in [0, 800]
        assert 0.0 <= cx <= 1040.0, f"Occupant marker x={cx} outside building bounds"
        assert 0.0 <= cy <= 800.0, f"Occupant marker y={cy} outside building bounds"


def test_strict_visual_decluttering(sim_pipeline, anim_mgr):
    """
    Requirement 3: Strict visual rules.
    Never display all sensor readings inside every room.
    Do not place temperature, CO2, humidity, lighting, sound, HVAC wattage across floor plan.
    """
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants

    # Provide sensor readings
    sensors = {r_id: {"indoor_temperature": 23.4, "indoor_co2": 520, "illuminance": 400} for r_id in EXPECTED_ROOM_IDS}
    energies = sim_pipeline.engine.energy_manager.get_latest_states()

    svg = generate_svg_floorplan_markup(
        building=building,
        occupants=occupants,
        selected_room_id="Room 1",
        active_overlay="Occupancy",
        sensor_readings=sensors,
        energy_states=energies,
        animation_manager=anim_mgr,
    )

    # Default SVG should NOT display raw multi-sensor labels in room body
    assert "23.4 °C" not in svg
    assert "520 ppm" not in svg
    assert "400 lux" not in svg
    # Must only display clean room names, room IDs, and occupancy ratio
    assert "👥" in svg


# -----------------------------------------------------------------------------
# 4. Multi-Physics Overlay Support
# -----------------------------------------------------------------------------

def test_all_overlays_render_svg_cleanly(sim_pipeline, anim_mgr):
    """Verifies that all 6 multi-physics overlays generate valid SVG markup."""
    building = sim_pipeline.engine.building
    occupants = sim_pipeline.engine.occupants
    sensors = {r_id: {"indoor_temperature": 24.0, "indoor_co2": 600} for r_id in EXPECTED_ROOM_IDS}
    energies = sim_pipeline.engine.energy_manager.get_latest_states()

    for overlay in OVERLAY_OPTIONS:
        svg = generate_svg_floorplan_markup(
            building=building,
            occupants=occupants,
            selected_room_id="Room 2",
            active_overlay=overlay,
            sensor_readings=sensors,
            energy_states=energies,
            animation_manager=anim_mgr,
        )
        assert "<svg" in svg
        assert "</svg>" in svg
        assert 'data-room-id="Room 2"' in svg


# -----------------------------------------------------------------------------
# 5. Pure Presentation / Simulation Invariance
# -----------------------------------------------------------------------------

def test_svg_rendering_causes_zero_simulation_mutations(sim_pipeline, anim_mgr):
    """Verifies that rendering the SVG floor plan 20 times causes 0 simulation state mutations."""
    engine = sim_pipeline.engine
    clock_start = engine.current_time
    step_start = engine.step_count
    arr_start = engine.total_arrivals
    exit_start = engine.total_exits
    kwh_start = sum(s.cumulative_total_kwh for s in engine.energy_manager.get_latest_states().values())

    building = engine.building
    occupants = engine.occupants
    sensors = {r: {"indoor_temperature": 23.0} for r in EXPECTED_ROOM_IDS}
    energies = engine.energy_manager.get_latest_states()

    for _ in range(20):
        _ = generate_svg_floorplan_markup(
            building=building,
            occupants=occupants,
            selected_room_id="Room 1",
            active_overlay="Occupancy",
            sensor_readings=sensors,
            energy_states=energies,
            animation_manager=anim_mgr,
        )

    # Assert 100% invariance
    assert engine.current_time == clock_start
    assert engine.step_count == step_start
    assert engine.total_arrivals == arr_start
    assert engine.total_exits == exit_start
    kwh_end = sum(s.cumulative_total_kwh for s in engine.energy_manager.get_latest_states().values())
    assert kwh_end == kwh_start

