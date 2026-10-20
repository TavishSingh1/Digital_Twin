"""
Phase 5 Interactive Person Entry & Live Sensor Monitoring Test Suite
=====================================================================
Validates interactive person additions, arbitrary capacity / batching,
entrance queueing at Room 1 capacity, FIFO queue processing, sensor
coupling (immediate acoustics vs continuous CO2/thermal dynamics),
lifecycle metrics, and STGNN inference pipeline integration.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from simulation.building import Building
from simulation.config import load_building_config, EXPECTED_ROOM_IDS
from stgnn_model import COMMON_SENSOR_FEATURES


@pytest.fixture
def clean_engine():
    """Initializes a deterministic simulation engine with zero initial occupants."""
    return SimulationEngine(num_occupants=0, seed=42)


@pytest.fixture
def populated_engine():
    """Initializes a simulation engine with 20 initial scheduled occupants."""
    return SimulationEngine(num_occupants=20, seed=42)


@pytest.fixture
def clean_pipeline():
    """Initializes an end-to-end pipeline with zero initial occupants."""
    engine = SimulationEngine(num_occupants=0, seed=42)
    return DigitalTwinPipeline(engine=engine, seed=42)


# ---------------------------------------------------------------------
# TEST 1: Single Person Entry Through Room 1
# ---------------------------------------------------------------------
def test_single_person_entry(clean_engine):
    """Test adding a single person through Room 1 entrance."""
    res = clean_engine.add_person(count=1)
    assert res["requested"] == 1
    assert res["admitted_count"] == 1
    assert res["queued_count"] == 0
    assert len(res["admitted_ids"]) == 1
    
    pid = res["admitted_ids"][0]
    assert pid == "P0001"
    assert clean_engine.building.get_room("Room 1").occupant_count == 1
    assert clean_engine.current_occupants == 1
    assert clean_engine.total_arrivals == 1


# ---------------------------------------------------------------------
# TEST 2: Batch Person Entry of 25 Persons
# ---------------------------------------------------------------------
def test_batch_person_entry_25(clean_engine):
    """Test batch entry of 25 occupants simultaneously into Room 1."""
    res = clean_engine.add_person(count=25)
    assert res["requested"] == 25
    assert res["admitted_count"] == 25
    assert res["queued_count"] == 0
    assert len(res["admitted_ids"]) == 25
    
    # Check that Room 1 has 25 occupants
    r1 = clean_engine.building.get_room("Room 1")
    assert r1.occupant_count == 25
    assert clean_engine.current_occupants == 25
    assert clean_engine.total_arrivals == 25


# ---------------------------------------------------------------------
# TEST 3: Repeated Person Additions
# ---------------------------------------------------------------------
def test_repeated_person_additions(clean_engine):
    """Test repeated manual additions across multiple actions."""
    res1 = clean_engine.add_person(count=5)
    res2 = clean_engine.add_person(count=10)
    
    assert res1["admitted_count"] == 5
    assert res2["admitted_count"] == 10
    assert clean_engine.current_occupants == 15
    assert clean_engine.total_arrivals == 15


# ---------------------------------------------------------------------
# TEST 4: Sequential Unique IDs
# ---------------------------------------------------------------------
def test_sequential_unique_ids(clean_engine):
    """Test that all created occupants receive strictly unique, sequential IDs."""
    res = clean_engine.add_person(count=10)
    ids = res["admitted_ids"]
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert ids == [f"P{i:04d}" for i in range(1, 11)]


# ---------------------------------------------------------------------
# TEST 5: Occupants Persist Across Simulation Steps
# ---------------------------------------------------------------------
def test_occupants_persist_across_steps(clean_engine):
    """Test that manually added occupants remain inside the building as steps advance."""
    clean_engine.add_person(count=5)
    assert clean_engine.current_occupants == 5
    
    # Step 3 times (15 minutes)
    for _ in range(3):
        clean_engine.step()
    
    # Total occupants inside building should remain 5 (unless manually exited)
    assert clean_engine.current_occupants == 5


# ---------------------------------------------------------------------
# TEST 6: Entrance Queueing at Room 1 Capacity
# ---------------------------------------------------------------------
def test_entrance_queueing_at_capacity(clean_engine):
    """Test that when Room 1 capacity (40) is reached, excess occupants queue outside."""
    r1 = clean_engine.building.get_room("Room 1")
    cap = r1.capacity  # 40
    
    res = clean_engine.add_person(count=55)
    assert res["requested"] == 55
    assert res["admitted_count"] == cap  # 40 admitted
    assert res["queued_count"] == 15     # 15 queued
    
    assert r1.occupant_count == cap
    assert clean_engine.current_occupants == cap
    assert clean_engine.waiting_at_entrance == 15
    assert len(clean_engine.entrance_queue) == 15
    
    # Verify queued occupants have state 'waiting_at_entrance'
    for occ in clean_engine.entrance_queue:
        assert occ.movement_state == "waiting_at_entrance"
        assert occ.current_room_id is None


# ---------------------------------------------------------------------
# TEST 7: FIFO Admission from Entrance Queue
# ---------------------------------------------------------------------
def test_fifo_admission_from_entrance_queue(clean_engine):
    """Test that queued occupants are admitted FIFO when Room 1 capacity frees up."""
    clean_engine.add_person(count=45)  # 40 admitted, 5 queued
    assert clean_engine.waiting_at_entrance == 5
    queued_first_id = clean_engine.entrance_queue[0].occupant_id
    
    # Free up 3 spots in Room 1 by exiting 3 people
    clean_engine.exit_person(count=3, room_id="Room 1")
    assert clean_engine.building.get_room("Room 1").occupant_count == 37
    
    # Step the simulation: engine should admit 3 queued people FIFO into Room 1
    clean_engine.step()
    
    r1 = clean_engine.building.get_room("Room 1")
    assert clean_engine.waiting_at_entrance == 2
    # Verify the first queued person was admitted into Room 1
    assert queued_first_id in r1.occupants


# ---------------------------------------------------------------------
# TEST 8: No Silent Discarding Under High Load
# ---------------------------------------------------------------------
def test_no_silent_discarding_under_high_load(clean_engine):
    """Test that adding 100 people at once preserves all occupants (inside or queued)."""
    res = clean_engine.add_person(count=100)
    assert res["requested"] == 100
    assert res["admitted_count"] + res["queued_count"] == 100
    
    total_tracked = clean_engine.current_occupants + clean_engine.waiting_at_entrance
    assert total_tracked == 100
    assert len(clean_engine.occupants) == 100


# ---------------------------------------------------------------------
# TEST 9: Metrics Consistency (No Double Counting)
# ---------------------------------------------------------------------
def test_metrics_consistency_no_double_counting(clean_engine):
    """Test conservation of occupants: arrivals - exits == inside + queued."""
    clean_engine.add_person(count=45)
    clean_engine.exit_person(count=10, room_id="Room 1")
    
    inside = clean_engine.current_occupants
    queued = clean_engine.waiting_at_entrance
    arrivals = clean_engine.total_arrivals
    exits = clean_engine.total_exits
    
    assert arrivals == 40
    assert exits == 10
    assert arrivals - exits == inside
    assert queued == 5


# ---------------------------------------------------------------------
# TEST 10: Exit Person Functionality
# ---------------------------------------------------------------------
def test_exit_person_functionality(clean_engine):
    """Test manually removing occupants from a specific room or building."""
    clean_engine.add_person(count=10)
    assert clean_engine.current_occupants == 10
    
    exit_res = clean_engine.exit_person(count=4, room_id="Room 1")
    assert exit_res["requested"] == 4
    assert exit_res["exited_count"] == 4
    assert clean_engine.current_occupants == 6
    assert clean_engine.total_exits == 4


# ---------------------------------------------------------------------
# TEST 11: Immediate Acoustic Sensor Response
# ---------------------------------------------------------------------
def test_immediate_acoustic_sensor_response(clean_engine):
    """Test that sound pressure level in Room 1 immediately responds to occupant entry."""
    # Baseline acoustic level with 0 occupants
    base_readings = clean_engine.sensor_manager.get_latest_readings()
    base_spl = base_readings["Room 1"].sound_pressure_level
    
    # Add 20 occupants without advancing time
    clean_engine.add_person(count=20)
    
    # Immediate telemetry check
    new_readings = clean_engine.sensor_manager.get_latest_readings()
    new_spl = new_readings["Room 1"].sound_pressure_level
    
    # Acoustic level should rise immediately
    assert new_spl > base_spl


# ---------------------------------------------------------------------
# TEST 12: Continuous CO2 Accumulation Over Steps
# ---------------------------------------------------------------------
def test_continuous_co2_accumulation_over_steps(clean_engine):
    """Test that CO2 evolves continuously over 5-minute steps rather than jumping immediately."""
    clean_engine.step()  # Initial step
    co2_init = clean_engine.sensor_manager.get_latest_readings()["Room 1"].indoor_co2
    
    # Add 25 people
    clean_engine.add_person(count=25)
    
    # Immediate CO2 should not jump discontinuously
    co2_immediate = clean_engine.sensor_manager.get_latest_readings()["Room 1"].indoor_co2
    assert pytest.approx(co2_immediate, abs=1.0) == co2_init
    
    # Step forward 1 step (5 minutes)
    clean_engine.step()
    co2_step1 = clean_engine.sensor_manager.get_latest_readings()["Room 1"].indoor_co2
    assert co2_step1 > co2_init
    
    # Step forward another step (10 minutes)
    clean_engine.step()
    co2_step2 = clean_engine.sensor_manager.get_latest_readings()["Room 1"].indoor_co2
    assert co2_step2 > co2_step1


# ---------------------------------------------------------------------
# TEST 13: Outdoor Features Independent of Indoor Occupants
# ---------------------------------------------------------------------
def test_outdoor_environmental_features_independent(clean_engine):
    """Test that outdoor ambient temperature and outdoor humidity are not affected by indoor occupancy."""
    readings_before = clean_engine.sensor_manager.get_latest_readings()["Room 1"]
    out_temp_before = readings_before.dry_bulb_temperature
    out_rh_before = readings_before.outdoor_relative_humidity
    
    clean_engine.add_person(count=30)
    
    readings_after = clean_engine.sensor_manager.get_latest_readings()["Room 1"]
    assert readings_after.dry_bulb_temperature == out_temp_before
    assert readings_after.outdoor_relative_humidity == out_rh_before


# ---------------------------------------------------------------------
# TEST 14: STGNN Pipeline Compatibility and Forecasting
# ---------------------------------------------------------------------
def test_pipeline_compatibility_and_forecasting(clean_pipeline):
    """Test that adding people interactively preserves rolling buffer shape and enables STGNN inference."""
    # Warm up buffer with 11 steps
    for _ in range(11):
        clean_pipeline.step()
    
    # Interactively add 15 people
    clean_pipeline.add_person(count=15)
    
    # Step to 12th step (warmup complete -> inference triggered)
    res = clean_pipeline.step()
    assert res is not None
    assert res["status"] == "SUCCESS"
    assert "predictions" in res
    assert len(res["predictions"]) == 5
    
    # Verify predictions are nonnegative integers
    for r_id, p_rec in res["predictions"].items():
        assert p_rec["predicted_occupants"] >= 0
        assert isinstance(p_rec["predicted_occupants"], int)


# ---------------------------------------------------------------------
# TEST 15: Reset Preserves Person Counter
# ---------------------------------------------------------------------
def test_reset_preserves_person_counter(clean_engine):
    """Test that reset(keep_session_counter=True) maintains monotonic ID progression."""
    clean_engine.add_person(count=5)  # P0001 - P0005
    assert clean_engine.person_counter == 6
    
    clean_engine.reset(keep_session_counter=True)
    assert clean_engine.person_counter == 6
    
    res = clean_engine.add_person(count=1)
    assert res["admitted_ids"][0] == "P0006"


# ---------------------------------------------------------------------
# TEST 16: Reset Restarts Counter When Requested
# ---------------------------------------------------------------------
def test_reset_restarts_counter_when_requested(clean_engine):
    """Test that reset(keep_session_counter=False) restarts ID progression at P0001."""
    clean_engine.add_person(count=5)  # P0001 - P0005
    assert clean_engine.person_counter == 6
    
    clean_engine.reset(keep_session_counter=False)
    assert clean_engine.person_counter == 1
    
    res = clean_engine.add_person(count=1)
    assert res["admitted_ids"][0] == "P0001"

