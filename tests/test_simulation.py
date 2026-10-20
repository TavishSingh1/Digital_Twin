"""
Automated Test Suite for Virtual Building & Occupant Simulation (Phase 3.1)
==========================================================================
Verifies all 14 safety and correctness requirements:
1. Exactly five configured rooms exist.
2. Room IDs match the verified model room ordering.
3. Every occupant belongs to at most one room.
4. No room exceeds its configured capacity.
5. No room occupancy becomes negative.
6. Total population remains conserved during internal transitions.
7. Arrivals and departures update population correctly.
8. Disconnected rooms cannot be reached without a valid route.
9. Occupants cannot pass through walls or room boundaries.
10. Repeating a scenario with the same seed produces identical results (determinism).
11. Simulation timestamps advance by exactly five minutes per logical step.
12. Reset restores the initial state cleanly.
13. CSV export contains correct timestamps, room IDs, and occupancy counts.
14. Invalid configurations fail with clear, descriptive error messages.
"""

import os
import tempfile
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

from simulation.config import (
    BuildingConfig,
    RoomConfig,
    ConnectionConfig,
    load_building_config,
    EXPECTED_ROOM_IDS,
    ConfigurationError,
    InvalidRoomError,
)
from simulation.room import Room, RoomCapacityExceededError
from simulation.building import Building
from simulation.engine import SimulationEngine


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Loads the verified default building configuration."""
    return load_building_config("config/building.json")


@pytest.fixture
def engine(default_config):
    """Creates a standard simulation engine instance with 40 occupants."""
    return SimulationEngine(config=default_config, num_occupants=40, seed=42)


# ---------------------------------------------------------------------
# 1 & 2. ROOM EXISTENCE AND ORDERING
# ---------------------------------------------------------------------

def test_five_configured_rooms_exist(default_config):
    """Test 1: Exactly five configured rooms exist in building."""
    assert len(default_config.rooms) == 5, f"Expected 5 rooms, got {len(default_config.rooms)}"
    building = Building(default_config)
    assert len(building.rooms) == 5


def test_room_ids_match_model_order(default_config):
    """Test 2: Room IDs strictly match the model's verified ordering."""
    room_ids = [r.id for r in default_config.rooms]
    assert room_ids == EXPECTED_ROOM_IDS, (
        f"Room IDs mismatch: {room_ids} != {EXPECTED_ROOM_IDS}"
    )


# ---------------------------------------------------------------------
# 3, 4, 5. OCCUPANT MEMBERSHIP, CAPACITY, AND NON-NEGATIVITY
# ---------------------------------------------------------------------

def test_occupant_membership_single_room(engine):
    """Test 3: Every occupant belongs to exactly one room (or is outside)."""
    engine.run(num_steps=24) # Simulate 2 hours
    active_ids = {o.occupant_id for o in engine.occupants.values() if o.is_inside_building}
    
    # Building's internal invariant validator confirms disjoint room sets
    engine.building.validate_invariants(active_ids)

    # Double check manually
    seen_occupants = set()
    for room in engine.building.rooms.values():
        for occ_id in room.occupants:
            assert occ_id not in seen_occupants, f"Occupant {occ_id} found in multiple rooms"
            seen_occupants.add(occ_id)
            assert engine.occupants[occ_id].current_room_id == room.room_id


def test_room_capacity_not_exceeded(engine):
    """Test 4: No room exceeds its configured capacity under heavy load."""
    # Run with 100 occupants for 48 steps (4 hours)
    heavy_engine = SimulationEngine(config=engine.config, num_occupants=100, seed=101)
    for _ in range(48):
        heavy_engine.step()
        for room in heavy_engine.building.rooms.values():
            assert room.occupant_count <= room.capacity, (
                f"Room '{room.room_id}' occupancy {room.occupant_count} exceeded capacity {room.capacity}"
            )


def test_room_occupancy_non_negative(engine):
    """Test 5: No room occupancy ever becomes negative."""
    for _ in range(36):
        engine.step()
        for room in engine.building.rooms.values():
            assert room.occupant_count >= 0, (
                f"Room '{room.room_id}' has negative occupancy: {room.occupant_count}"
            )


# ---------------------------------------------------------------------
# 6 & 7. POPULATION CONSERVATION, ARRIVALS, AND DEPARTURES
# ---------------------------------------------------------------------

def test_population_conservation_during_internal_transitions(default_config):
    """Test 6: Total occupants are strictly conserved during internal room transitions."""
    building = Building(default_config)
    r1 = building.get_room("Room 1")
    r2 = building.get_room("Room 2")

    r1.add_occupant("test_occ_1")
    r1.add_occupant("test_occ_2")
    initial_total = building.total_occupants
    assert initial_total == 2

    # Move test_occ_1 from Room 1 to Room 2
    r1.remove_occupant("test_occ_1")
    r2.add_occupant("test_occ_1")

    assert building.total_occupants == initial_total
    assert r1.occupant_count == 1
    assert r2.occupant_count == 1


def test_arrivals_and_departures_update_population(engine):
    """Test 7: Entering and leaving explicitly updates total building population."""
    initial_pop = engine.building.total_occupants
    # Run through morning arrivals (08:00 to 10:00 = 24 steps)
    engine.run(num_steps=24)
    peak_pop = engine.building.total_occupants
    assert peak_pop > initial_pop, "Population should increase during morning arrivals"

    # Total inside building must equal active occupant count
    inside_count = sum(1 for o in engine.occupants.values() if o.is_inside_building)
    assert engine.building.total_occupants == inside_count


# ---------------------------------------------------------------------
# 8 & 9. TOPOLOGY, PATHFINDING, AND WALL BOUNDARIES
# ---------------------------------------------------------------------

def test_disconnected_rooms_unreachable():
    """Test 8: Disconnected rooms cannot be reached without a valid route."""
    # Create a config with Room 5 completely isolated
    cfg = load_building_config("config/building.json")
    cfg.connections = [
        c for c in cfg.connections if c.room_a != "Room 5" and c.room_b != "Room 5"
    ]
    building = Building(cfg)

    # Path from Room 1 to Room 5 should be None
    path = building.find_path("Room 1", "Room 5")
    assert path is None, "Path to disconnected room should be None"

    # Path between connected rooms still works
    path_1_2 = building.find_path("Room 1", "Room 2")
    assert path_1_2 == ["Room 1", "Room 2"]


def test_occupants_remain_within_room_boundaries(engine):
    """Test 9: Occupants cannot pass through walls or room boundaries."""
    engine.run(num_steps=36)
    for occ in engine.occupants.values():
        if occ.is_inside_building:
            room = engine.building.get_room(occ.current_room_id)
            px, py = occ.position
            # Verify coordinates are strictly inside room rectangle
            assert room.x <= px <= (room.x + room.width), (
                f"Occupant {occ.occupant_id} x-coordinate {px} is outside Room '{room.room_id}' bounds [{room.x}, {room.x+room.width}]"
            )
            assert room.y <= py <= (room.y + room.height), (
                f"Occupant {occ.occupant_id} y-coordinate {py} is outside Room '{room.room_id}' bounds [{room.y}, {room.y+room.height}]"
            )


# ---------------------------------------------------------------------
# 10. REPRODUCIBILITY & DETERMINISM
# ---------------------------------------------------------------------

def test_scenario_reproducibility(default_config):
    """Test 10: Repeating a scenario with the same seed produces identical results."""
    engine_a = SimulationEngine(config=default_config, num_occupants=30, seed=777)
    df_a = engine_a.run(num_steps=24)

    engine_b = SimulationEngine(config=default_config, num_occupants=30, seed=777)
    df_b = engine_b.run(num_steps=24)

    pd.testing.assert_frame_equal(df_a, df_b), "Simulation with identical seed must produce identical ground truth"


# ---------------------------------------------------------------------
# 11 & 12. LOGICAL TIMESTAMPS AND RESET
# ---------------------------------------------------------------------

def test_timestamps_advance_by_five_minutes(engine):
    """Test 11: Logical clock advances by exactly 5 minutes per step."""
    t0 = engine.current_time
    res1 = engine.step()
    t1 = engine.current_time
    assert t1 - t0 == timedelta(minutes=5)

    res2 = engine.step()
    t2 = engine.current_time
    assert t2 - t1 == timedelta(minutes=5)


def test_reset_restores_initial_state(engine):
    """Test 12: Reset restores the initial state cleanly."""
    t0 = engine.current_time
    engine.run(num_steps=30)
    assert engine.step_count == 30
    assert engine.building.total_occupants > 0

    engine.reset()
    assert engine.step_count == 0
    assert engine.current_time == t0
    assert engine.building.total_occupants == 0
    assert len(engine.recorder) == 5  # Initial t=0 record for 5 rooms


# ---------------------------------------------------------------------
# 13. CSV EXPORT INTEGRITY
# ---------------------------------------------------------------------

def test_csv_export_format(engine):
    """Test 13: Ground-truth export produces valid CSV with correct schema."""
    engine.run(num_steps=12)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv = tmp.name

    try:
        engine.export_ground_truth(tmp_csv)
        assert os.path.exists(tmp_csv)
        df = pd.read_csv(tmp_csv)

        expected_cols = [
            "timestamp", "room_id", "actual_occupants", "room_capacity",
            "occupancy_fraction", "arrivals", "departures",
            "incoming_transitions", "outgoing_transitions"
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == (12 + 1) * 5  # 13 timesteps * 5 rooms
        assert set(df["room_id"].unique()) == set(EXPECTED_ROOM_IDS)
        assert (df["actual_occupants"] >= 0).all()
        assert (df["actual_occupants"] <= df["room_capacity"]).all()
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


# ---------------------------------------------------------------------
# 14. CONFIGURATION VALIDATION & ERROR REJECTION
# ---------------------------------------------------------------------

def test_invalid_configurations_rejected():
    """Test 14: Invalid room IDs, negative capacities, and invalid dimensions raise clear errors."""
    # 1. Unknown room ID
    with pytest.raises(InvalidRoomError):
        rc = RoomConfig("Room 99", "Fake", 10.0, 10.0, 0.0, 0.0, 20)
        rc.validate()

    # 2. Non-positive capacity
    with pytest.raises(ConfigurationError):
        rc = RoomConfig("Room 1", "Room", 10.0, 10.0, 0.0, 0.0, 0)
        rc.validate()

    # 3. Non-positive dimensions
    with pytest.raises(ConfigurationError):
        rc = RoomConfig("Room 1", "Room", -5.0, 10.0, 0.0, 0.0, 20)
        rc.validate()

    # 4. Wrong number of rooms
    cfg = load_building_config("config/building.json")
    cfg.rooms = cfg.rooms[:4]
    with pytest.raises(ConfigurationError):
        cfg.validate()

