"""
Manual People Flow Control Test Suite
======================================
Validates manual movement of occupants between rooms:
- Direct topological connectivity validation via doors
- Source and destination validation
- Specific person IDs vs count-based movement
- Capacity limit enforcement
- Occupant exclusivity (no dual residency)
- Immediate sensor response to movement
- Preservation of building arrival and exit metrics
- Full pipeline integration and STGNN forecast compatibility
"""

import pytest
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from simulation.config import EXPECTED_ROOM_IDS


@pytest.fixture
def clean_engine():
    """Deterministic simulation engine with 0 initial occupants."""
    return SimulationEngine(num_occupants=0, seed=42)


@pytest.fixture
def clean_pipeline():
    """Deterministic pipeline with 0 initial occupants."""
    engine = SimulationEngine(num_occupants=0, seed=42)
    return DigitalTwinPipeline(engine=engine, seed=42)


# ---------------------------------------------------------------------
# TEST 1: Move Single Person Between Connected Rooms
# ---------------------------------------------------------------------
def test_move_single_person_between_connected_rooms(clean_engine):
    """Test moving a single occupant from Room 1 to directly connected Room 2."""
    clean_engine.add_person(count=5)
    r1 = clean_engine.building.get_room("Room 1")
    r2 = clean_engine.building.get_room("Room 2")
    
    assert r1.occupant_count == 5
    assert r2.occupant_count == 0
    
    res = clean_engine.move_people(source_room="Room 1", destination_room="Room 2", count=1)
    
    assert res["success"] is True
    assert res["moved_count"] == 1
    assert r1.occupant_count == 4
    assert r2.occupant_count == 1
    
    moved_id = res["moved_ids"][0]
    assert moved_id in r2.occupants
    assert moved_id not in r1.occupants
    assert clean_engine.occupants[moved_id].current_room_id == "Room 2"


# ---------------------------------------------------------------------
# TEST 2: Move Selected Group of People
# ---------------------------------------------------------------------
def test_move_selected_group_of_people(clean_engine):
    """Test moving a specific list of occupant IDs between connected rooms."""
    clean_engine.add_person(count=10)
    target_ids = ["P0002", "P0004", "P0007"]
    
    res = clean_engine.move_people(
        source_room="Room 1",
        destination_room="Room 2",
        person_ids=target_ids,
    )
    
    assert res["success"] is True
    assert res["moved_count"] == 3
    assert set(res["moved_ids"]) == set(target_ids)
    
    r1 = clean_engine.building.get_room("Room 1")
    r2 = clean_engine.building.get_room("Room 2")
    assert r1.occupant_count == 7
    assert r2.occupant_count == 3
    for pid in target_ids:
        assert pid in r2.occupants
        assert pid not in r1.occupants
        assert clean_engine.occupants[pid].current_room_id == "Room 2"


# ---------------------------------------------------------------------
# TEST 3: Reject Movement Between Unconnected Rooms
# ---------------------------------------------------------------------
def test_reject_movement_between_unconnected_rooms(clean_engine):
    """Test that movement across solid walls (non-adjacent rooms) is rejected."""
    clean_engine.add_person(count=5)
    
    # Room 1 is connected to Room 2 and Room 3, but NOT directly to Room 4
    with pytest.raises(ValueError) as excinfo:
        clean_engine.move_people(source_room="Room 1", destination_room="Room 4", count=2)
    
    err_msg = str(excinfo.value)
    assert "not directly connected" in err_msg
    assert "intermediate room" in err_msg
    
    # State remains unchanged
    assert clean_engine.building.get_room("Room 1").occupant_count == 5
    assert clean_engine.building.get_room("Room 4").occupant_count == 0


# ---------------------------------------------------------------------
# TEST 4: Reject Movement into a Full Room
# ---------------------------------------------------------------------
def test_reject_movement_into_full_room(clean_engine):
    """Test that moving occupants into a room exceeding capacity is rejected."""
    # Room 3 has capacity 15. Fill it.
    clean_engine.add_person(count=20)  # in Room 1
    # Move 15 people to Room 3 (directly connected to Room 1)
    clean_engine.move_people(source_room="Room 1", destination_room="Room 3", count=15)
    
    r3 = clean_engine.building.get_room("Room 3")
    assert r3.occupant_count == 15
    assert r3.capacity == 15
    
    # Try to move 1 more person into full Room 3
    with pytest.raises(ValueError) as excinfo:
        clean_engine.move_people(source_room="Room 1", destination_room="Room 3", count=1)
    
    assert "full capacity" in str(excinfo.value)
    assert r3.occupant_count == 15


# ---------------------------------------------------------------------
# TEST 5: Reject Nonexistent Occupant IDs
# ---------------------------------------------------------------------
def test_reject_nonexistent_occupant_ids(clean_engine):
    """Test that specifying invalid or mislocated occupant IDs raises an error."""
    clean_engine.add_person(count=5)
    
    # Nonexistent ID
    with pytest.raises(ValueError) as exc1:
        clean_engine.move_people("Room 1", "Room 2", person_ids=["P9999"])
    assert "does not exist" in str(exc1.value)
    
    # Exists, but not in source room
    with pytest.raises(ValueError) as exc2:
        clean_engine.move_people("Room 2", "Room 4", person_ids=["P0001"])
    assert "not currently in source room 'Room 2'" in str(exc2.value)


# ---------------------------------------------------------------------
# TEST 6: Occupant Exclusivity (No Dual Room Residency)
# ---------------------------------------------------------------------
def test_occupant_exclusivity_no_dual_residency(clean_engine):
    """Test that no occupant can exist in two rooms simultaneously after movement."""
    clean_engine.add_person(count=20)
    clean_engine.move_people("Room 1", "Room 2", count=8)
    clean_engine.move_people("Room 2", "Room 4", count=3)
    clean_engine.move_people("Room 1", "Room 3", count=5)
    
    rooms = [clean_engine.building.get_room(r) for r in EXPECTED_ROOM_IDS]
    
    for i, ra in enumerate(rooms):
        for rb in rooms[i + 1:]:
            overlap = ra.occupants.intersection(rb.occupants)
            assert len(overlap) == 0, f"Occupants {overlap} exist in both {ra.room_id} and {rb.room_id}"
            
    # Verify building total is conserved
    assert clean_engine.building.total_occupants == 20


# ---------------------------------------------------------------------
# TEST 7: Updating Source and Destination Occupancy Correctly
# ---------------------------------------------------------------------
def test_updating_source_and_destination_occupancy(clean_engine):
    """Test multi-hop sequence correctly updates counts at every step."""
    clean_engine.add_person(count=25)
    
    # 25 in Room 1
    # Move 10 to Room 2 -> R1: 15, R2: 10
    clean_engine.move_people("Room 1", "Room 2", count=10)
    assert clean_engine.building.get_room("Room 1").occupant_count == 15
    assert clean_engine.building.get_room("Room 2").occupant_count == 10
    
    # Move 4 from Room 2 to Room 5 (connected) -> R2: 6, R5: 4
    clean_engine.move_people("Room 2", "Room 5", count=4)
    assert clean_engine.building.get_room("Room 2").occupant_count == 6
    assert clean_engine.building.get_room("Room 5").occupant_count == 4
    
    # Total occupants remains 25
    assert clean_engine.building.total_occupants == 25


# ---------------------------------------------------------------------
# TEST 8: Updating Sensor State After Movement
# ---------------------------------------------------------------------
def test_sensor_state_updates_after_movement(clean_engine):
    """Test that acoustic sound pressure level responds immediately to occupant transfer."""
    clean_engine.add_person(count=20)  # All in Room 1
    
    readings_t0 = clean_engine.sensor_manager.get_latest_readings()
    spl_r1_t0 = readings_t0["Room 1"].sound_pressure_level
    spl_r2_t0 = readings_t0["Room 2"].sound_pressure_level
    
    # Move 15 people from Room 1 to Room 2
    clean_engine.move_people("Room 1", "Room 2", count=15)
    
    readings_t1 = clean_engine.sensor_manager.get_latest_readings()
    spl_r1_t1 = readings_t1["Room 1"].sound_pressure_level
    spl_r2_t1 = readings_t1["Room 2"].sound_pressure_level
    
    # Room 1 sound level drops as people leave
    assert spl_r1_t1 < spl_r1_t0
    # Room 2 sound level jumps as 15 people arrive
    assert spl_r2_t1 > spl_r2_t0


# ---------------------------------------------------------------------
# TEST 9: Preserving Existing Arrivals and Exits
# ---------------------------------------------------------------------
def test_preserving_arrivals_and_exits(clean_engine):
    """Test that internal room transfers do not alter cumulative building arrivals or exits."""
    clean_engine.add_person(count=12)
    assert clean_engine.total_arrivals == 12
    assert clean_engine.total_exits == 0
    
    clean_engine.move_people("Room 1", "Room 2", count=6)
    clean_engine.move_people("Room 2", "Room 4", count=3)
    
    assert clean_engine.total_arrivals == 12
    assert clean_engine.total_exits == 0


# ---------------------------------------------------------------------
# TEST 10: Reject Same Source and Destination Room
# ---------------------------------------------------------------------
def test_reject_same_source_and_destination(clean_engine):
    """Test that attempting to move people from a room to itself raises ValueError."""
    clean_engine.add_person(count=5)
    with pytest.raises(ValueError) as exc:
        clean_engine.move_people("Room 1", "Room 1", count=1)
    assert "cannot be the same" in str(exc.value)


# ---------------------------------------------------------------------
# TEST 11: Pipeline Integration and STGNN Lookback Update
# ---------------------------------------------------------------------
def test_pipeline_integration_and_stgnn_lookback(clean_pipeline):
    """Test moving people via pipeline updates rolling buffer and feeds valid tensor to STGNN."""
    # Warm up buffer with 11 steps
    for _ in range(11):
        clean_pipeline.step()
        
    clean_pipeline.add_person(count=20)
    # Move 8 people from Room 1 to Room 2
    res = clean_pipeline.move_people("Room 1", "Room 2", count=8)
    assert res["success"] is True
    
    # Step to 12th step (forecast triggered)
    step_res = clean_pipeline.step()
    assert step_res["status"] == "SUCCESS"
    assert "predictions" in step_res
    assert len(step_res["predictions"]) == 5
    for r_id, p_info in step_res["predictions"].items():
        assert p_info["predicted"] >= 0

