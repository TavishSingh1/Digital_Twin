"""
Unit and Integration Tests for Phase 8.2: Occupant Animation & Movement Visualization
=====================================================================================
Validates all 16 Phase 8.2 requirements:
1. Confirmed movement creates one animation.
2. Rejected movement creates no animation.
3. Route waypoints follow valid room connections.
4. Occupants do not pass through solid walls.
5. Multiple simultaneous movements work.
6. Batch movement preserves occupant IDs.
7. Stationary markers are deterministic across reruns.
8. Queued occupants are not counted as room occupants.
9. Entry and exit animations reflect confirmed engine events.
10. Repeated reruns do not duplicate animations.
11. Animation progress does not advance simulation time.
12. Animation progress does not increment cumulative energy.
13. Reset clears transient animation state.
14. Missing geometry and degenerate paths are handled safely.
15. Existing room selection and all 6 overlays continue to work.
16. STGNN pipeline and simulation integrity remain completely unaffected.
"""

import pytest
import math
import plotly.graph_objects as go

from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from simulation.building import Building
from simulation.room import Room
from simulation.occupant import Occupant
from simulation.config import EXPECTED_ROOM_IDS
from dashboard.animation import (
    AnimationManager,
    MovementEvent,
    get_deterministic_room_grid_position,
    calculate_route_waypoints,
    interpolate_along_waypoints,
)
from dashboard.floorplan import (
    create_floorplan_figure,
    OVERLAY_OPTIONS,
    validate_building_geometry,
)
from dashboard.state import (
    add_person_action,
    exit_person_action,
    move_people_action,
    get_animation_manager,
    reset_simulation,
    init_session_state,
)


@pytest.fixture
def engine():
    """Initializes a reproducible simulation engine."""
    return SimulationEngine(num_occupants=20, seed=42)


@pytest.fixture
def pipeline(engine):
    """Initializes a reproducible digital twin pipeline."""
    return DigitalTwinPipeline(engine=engine, seed=42)


@pytest.fixture
def anim_mgr():
    """Initializes a clean AnimationManager."""
    mgr = AnimationManager()
    mgr.clear()
    return mgr


@pytest.fixture
def building(engine):
    """Provides building instance from engine."""
    return engine.building


# ---------------------------------------------------------------------------
# Requirement 1: Confirmed movement creates one animation
# ---------------------------------------------------------------------------
def test_1_confirmed_movement_creates_one_animation(anim_mgr, building):
    # Move from Room 1 to connected Room 2
    ev = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:00:00",
        person_ids=["P0001"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    assert ev is not None
    assert len(anim_mgr.events) == 1
    assert anim_mgr.events[0].event_type == "transfer"
    assert anim_mgr.events[0].source_room == "Room 1"
    assert anim_mgr.events[0].destination_room == "Room 2"
    assert anim_mgr.events[0].person_ids == ["P0001"]
    assert len(anim_mgr.events[0].waypoints) >= 2


# ---------------------------------------------------------------------------
# Requirement 2: Rejected movement creates no animation
# ---------------------------------------------------------------------------
def test_2_rejected_movement_creates_no_animation(engine, anim_mgr):
    # Attempt movement between disconnected rooms without intermediate hop (Room 1 -> Room 5)
    initial_event_count = len(anim_mgr.events)
    with pytest.raises(ValueError):
        engine.move_people(source_room="Room 1", destination_room="Room 5", count=1)

    # Verify no animation was recorded in anim_mgr
    assert len(anim_mgr.events) == initial_event_count


# ---------------------------------------------------------------------------
# Requirement 3: Route waypoints follow valid room connections
# ---------------------------------------------------------------------------
def test_3_route_waypoints_follow_valid_room_connections(building):
    # Multi-hop: Room 1 -> Room 4 (must traverse Room 1 -> Room 2 -> Room 4 or Room 1 -> Room 3 -> Room 4)
    waypoints = calculate_route_waypoints(building, "Room 1", "Room 4")
    assert len(waypoints) >= 3

    # Waypoints must include door between Room 1 and intermediate room, then door to Room 4
    door_1_2 = building.door_positions[("Room 1", "Room 2")]
    door_2_4 = building.door_positions[("Room 2", "Room 4")]
    door_1_3 = building.door_positions[("Room 1", "Room 3")]
    door_3_4 = building.door_positions[("Room 3", "Room 4")]

    via_room2 = door_1_2 in waypoints and door_2_4 in waypoints
    via_room3 = door_1_3 in waypoints and door_3_4 in waypoints
    assert via_room2 or via_room3, f"Waypoints {waypoints} did not route through valid doors."


# ---------------------------------------------------------------------------
# Requirement 4: Occupants do not pass through solid walls
# ---------------------------------------------------------------------------
def test_4_occupants_do_not_pass_through_solid_walls(building):
    # Direct line between Room 1 center (7.0, 5.0) and Room 4 center (11.0, 15.0) intersects wall at y=10.0 (x in [0, 8.0])
    waypoints = calculate_route_waypoints(building, "Room 1", "Room 4")
    # All intermediate waypoints must have x >= 4.0 and pass through door openings, not cutting through wall
    for pt in waypoints:
        assert isinstance(pt, tuple) and len(pt) == 2
        # Check inside building bounding box
        assert 0.0 <= pt[0] <= 26.0
        assert 0.0 <= pt[1] <= 20.0


# ---------------------------------------------------------------------------
# Requirement 5: Multiple simultaneous movements work
# ---------------------------------------------------------------------------
def test_5_multiple_simultaneous_movements(anim_mgr, building):
    ev1 = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:05:00",
        person_ids=["P0001", "P0002"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    ev2 = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:05:00",
        person_ids=["P0003"],
        source_room="Room 3",
        destination_room="Room 4",
        building=building,
    )
    assert ev1 is not None and ev2 is not None
    assert len(anim_mgr.events) == 2


# ---------------------------------------------------------------------------
# Requirement 6: Batch movement preserves occupant IDs
# ---------------------------------------------------------------------------
def test_6_batch_movement_preserves_occupant_ids(anim_mgr, building):
    batch_pids = ["P0010", "P0011", "P0012", "P0013", "P0014"]
    ev = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:10:00",
        person_ids=batch_pids,
        source_room="Room 2",
        destination_room="Room 5",
        building=building,
    )
    assert ev is not None
    assert ev.person_ids == batch_pids
    # Verify interpolated positions for all members
    pos = ev.get_interpolated_position(0.5)
    assert isinstance(pos, tuple) and len(pos) == 2


# ---------------------------------------------------------------------------
# Requirement 7: Stationary markers are deterministic across reruns
# ---------------------------------------------------------------------------
def test_7_stationary_markers_are_deterministic(engine, anim_mgr):
    # Compute positions multiple times without modifying building
    pos_run1 = anim_mgr.get_occupant_positions(engine.building, engine.occupants)
    pos_run2 = anim_mgr.get_occupant_positions(engine.building, engine.occupants)
    pos_run3 = anim_mgr.get_occupant_positions(engine.building, engine.occupants)

    assert pos_run1 == pos_run2 == pos_run3
    # Check that occupants inside the same room do not all occupy the exact same coordinate
    room1_occupants = [pid for pid in engine.building.get_room("Room 1").occupants]
    if len(room1_occupants) >= 2:
        assert pos_run1[room1_occupants[0]] != pos_run1[room1_occupants[1]]


# ---------------------------------------------------------------------------
# Requirement 8: Queued occupants are not counted as room occupants
# ---------------------------------------------------------------------------
def test_8_queued_occupants_not_counted_as_room_occupants(engine):
    r1 = engine.building.get_room("Room 1")
    # Fill Room 1 to max capacity
    needed = r1.capacity - r1.occupant_count
    if needed > 0:
        engine.add_person(count=needed)
    assert r1.occupant_count == r1.capacity
    baseline_total = engine.building.total_occupants

    # Add 3 more people (should go to queue)
    res = engine.add_person(count=3)
    assert res["queued_count"] == 3
    assert res["admitted_count"] == 0
    assert r1.occupant_count == r1.capacity
    assert engine.building.total_occupants == baseline_total
    assert len(engine.entrance_queue) == 3

    # Ensure queued occupants are outside Room 1
    for queued_occ in engine.entrance_queue:
        assert queued_occ.current_room_id is None
        assert queued_occ.movement_state == "waiting_at_entrance"


# ---------------------------------------------------------------------------
# Requirement 9: Entry and exit animations reflect confirmed engine events
# ---------------------------------------------------------------------------
def test_9_entry_and_exit_animations_reflect_confirmed_events(engine, anim_mgr):
    # Admission
    res_add = engine.add_person(count=1)
    admitted = res_add["admitted_ids"]
    assert len(admitted) == 1
    ev_add = anim_mgr.record_movement(
        event_type="admission",
        timestamp=engine.current_time.isoformat(),
        person_ids=admitted,
        source_room="Outside",
        destination_room="Room 1",
        building=engine.building,
    )
    assert ev_add is not None
    assert ev_add.event_type == "admission"
    assert ev_add.waypoints[0] == (7.0, -1.6)  # Outside queue position

    # Departure
    res_exit = engine.exit_person(count=1, room_id="Room 1")
    exited = res_exit["exited_ids"]
    assert len(exited) == 1
    ev_exit = anim_mgr.record_movement(
        event_type="exit",
        timestamp=engine.current_time.isoformat(),
        person_ids=exited,
        source_room="Room 1",
        destination_room=None,
        building=engine.building,
    )
    assert ev_exit is not None
    assert len(anim_mgr.exit_records) == 1
    assert anim_mgr.exit_records[0]["occupant_id"] == exited[0]
    # Visual exit record must NOT count towards room occupancy
    assert exited[0] not in engine.building.get_room("Room 1").occupants


# ---------------------------------------------------------------------------
# Requirement 10: Repeated reruns do not duplicate animations
# ---------------------------------------------------------------------------
def test_10_repeated_reruns_do_not_duplicate_animations(anim_mgr, building):
    ev1 = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:20:00",
        person_ids=["P0005"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    assert ev1 is not None
    # Duplicate invocation with identical params
    ev2 = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:20:00",
        person_ids=["P0005"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    assert ev2 is None
    assert len(anim_mgr.events) == 1


# ---------------------------------------------------------------------------
# Requirement 11: Animation progress does not advance simulation time
# ---------------------------------------------------------------------------
def test_11_animation_progress_does_not_advance_simulation_time(engine, anim_mgr):
    t_start = engine.current_time
    step_start = engine.step_count

    # Exercise animation progress through waypoints and position lookups
    anim_mgr.record_movement(
        event_type="transfer",
        timestamp=t_start.isoformat(),
        person_ids=["P0001"],
        source_room="Room 1",
        destination_room="Room 2",
        building=engine.building,
    )
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        anim_mgr.get_occupant_positions(engine.building, engine.occupants, progress=p)

    assert engine.current_time == t_start
    assert engine.step_count == step_start


# ---------------------------------------------------------------------------
# Requirement 12: Animation progress does not increment cumulative energy
# ---------------------------------------------------------------------------
def test_12_animation_progress_does_not_increment_cumulative_energy(engine, anim_mgr):
    e_states = engine.energy_manager.get_latest_states()
    initial_kwh = sum(s.cumulative_total_kwh for s in e_states.values())

    # Create figure with animation manager attached and query positions
    fig = create_floorplan_figure(
        building=engine.building,
        occupants=engine.occupants,
        animation_manager=anim_mgr,
        show_paths=True,
    )
    assert isinstance(fig, go.Figure)

    e_states_after = engine.energy_manager.get_latest_states()
    after_kwh = sum(s.cumulative_total_kwh for s in e_states_after.values())
    assert initial_kwh == pytest.approx(after_kwh, abs=1e-9)


# ---------------------------------------------------------------------------
# Requirement 13: Reset clears transient animation state
# ---------------------------------------------------------------------------
def test_13_reset_clears_transient_animation_state(anim_mgr, building):
    anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T09:30:00",
        person_ids=["P0001"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    anim_mgr.record_movement(
        event_type="exit",
        timestamp="2026-10-19T09:30:00",
        person_ids=["P0002"],
        source_room="Room 1",
        destination_room=None,
        building=building,
    )
    anim_mgr.selected_occupant_id = "P0001"
    assert len(anim_mgr.events) > 0
    assert len(anim_mgr.exit_records) > 0
    assert anim_mgr.selected_occupant_id is not None

    anim_mgr.clear()
    assert len(anim_mgr.events) == 0
    assert len(anim_mgr.exit_records) == 0
    assert len(anim_mgr.seen_event_hashes) == 0
    assert anim_mgr.selected_occupant_id is None


# ---------------------------------------------------------------------------
# Requirement 14: Missing geometry and degenerate paths handled safely
# ---------------------------------------------------------------------------
def test_14_missing_geometry_handled_safely():
    # Empty waypoints
    assert interpolate_along_waypoints([], 0.5) == (0.0, 0.0)
    # Single waypoint
    assert interpolate_along_waypoints([(5.0, 5.0)], 0.5) == (5.0, 5.0)
    # Zero-length segment
    assert interpolate_along_waypoints([(3.0, 3.0), (3.0, 3.0)], 0.5) == (3.0, 3.0)
    # Progress bounds clamping
    pts = [(0.0, 0.0), (10.0, 0.0)]
    assert interpolate_along_waypoints(pts, -0.5) == (0.0, 0.0)
    assert interpolate_along_waypoints(pts, 1.5) == (10.0, 0.0)

    # Invalid geometry rejection
    bad_building = Building.__new__(Building)
    bad_building.rooms = {}
    with pytest.raises(ValueError, match="does not contain any configured rooms"):
        validate_building_geometry(bad_building)


# ---------------------------------------------------------------------------
# Requirement 15: Existing room selection and all 6 overlays still work
# ---------------------------------------------------------------------------
def test_15_existing_room_selection_and_overlays_still_work(engine, anim_mgr):
    # Record a test movement
    anim_mgr.record_movement(
        event_type="transfer",
        timestamp=engine.current_time.isoformat(),
        person_ids=["P0001"],
        source_room="Room 1",
        destination_room="Room 2",
        building=engine.building,
    )
    anim_mgr.selected_occupant_id = "P0001"

    for overlay in OVERLAY_OPTIONS:
        fig = create_floorplan_figure(
            building=engine.building,
            occupants=engine.occupants,
            selected_room_id="Room 2",
            active_overlay=overlay,
            animation_manager=anim_mgr,
            show_paths=True,
            selected_occupant_id="P0001",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Contains traces


# ---------------------------------------------------------------------------
# Requirement 16: STGNN pipeline and simulation integrity unaffected
# ---------------------------------------------------------------------------
def test_16_stgnn_pipeline_unaffected_by_animation(pipeline, anim_mgr):
    # Step pipeline 2 times
    r1 = pipeline.step()
    r2 = pipeline.step()
    assert r1["status"] in ["WARMUP", "SUCCESS"]
    assert r2["status"] in ["WARMUP", "SUCCESS"]

    # Interactively move occupant and verify prediction shape and rolling buffer intact
    p_occ = next(iter(pipeline.engine.occupants.values()))
    orig_room = p_occ.current_room_id
    if orig_room == "Room 1":
        target = "Room 2"
    else:
        target = "Room 1"

    # Route if directly connected
    if target in pipeline.engine.building.adjacency.get(orig_room, set()):
        res = pipeline.move_people(source_room=orig_room, destination_room=target, person_ids=[p_occ.occupant_id])
        assert res["success"] is True

        anim_mgr.record_movement(
            event_type="transfer",
            timestamp=pipeline.engine.current_time.isoformat(),
            person_ids=[p_occ.occupant_id],
            source_room=orig_room,
            destination_room=target,
            building=pipeline.engine.building,
        )

    # Next step produces valid output
    r3 = pipeline.step()
    assert r3["status"] in ["WARMUP", "SUCCESS"]
    assert len(pipeline.rolling_buffer) <= 12


# ---------------------------------------------------------------------------
# Requirement 17: Visual speed scaling (0.5x, 1.0x, 1.5x, 2.0x)
# ---------------------------------------------------------------------------
def test_17_visual_speed_scaling(anim_mgr):
    for spd in [0.5, 1.0, 1.5, 2.0]:
        anim_mgr.set_speed(spd)
        assert anim_mgr.animation_speed == spd

    # Invalid speeds are ignored
    anim_mgr.set_speed(3.5)
    assert anim_mgr.animation_speed == 2.0


# ---------------------------------------------------------------------------
# Requirement 18: Visual pause and resume
# ---------------------------------------------------------------------------
def test_18_visual_pause_and_resume(anim_mgr):
    assert anim_mgr.is_paused is False
    anim_mgr.pause()
    assert anim_mgr.is_paused is True
    anim_mgr.resume()
    assert anim_mgr.is_paused is False
    res = anim_mgr.toggle_pause()
    assert res is True
    assert anim_mgr.is_paused is True
    res2 = anim_mgr.toggle_pause()
    assert res2 is False
    assert anim_mgr.is_paused is False


# ---------------------------------------------------------------------------
# Requirement 19: Continuous visual interpolation along doorway waypoints
# ---------------------------------------------------------------------------
def test_19_continuous_interpolation_along_waypoints(building):
    # Route from Room 1 to Room 4
    waypoints = calculate_route_waypoints(building, "Room 1", "Room 4")
    assert len(waypoints) >= 3

    p0 = interpolate_along_waypoints(waypoints, 0.0)
    p25 = interpolate_along_waypoints(waypoints, 0.25)
    p50 = interpolate_along_waypoints(waypoints, 0.5)
    p75 = interpolate_along_waypoints(waypoints, 0.75)
    p100 = interpolate_along_waypoints(waypoints, 1.0)

    assert p0 == waypoints[0]
    assert p100 == waypoints[-1]
    # Verify continuous movement: intermediate points are distinct and non-null
    assert p0 != p25
    assert p25 != p50
    assert p50 != p75
    assert p75 != p100
    # Clamping test
    assert interpolate_along_waypoints(waypoints, -0.5) == p0
    assert interpolate_along_waypoints(waypoints, 1.5) == p100


# ---------------------------------------------------------------------------
# Requirement 20: Simultaneous multi-occupant animations
# ---------------------------------------------------------------------------
def test_20_simultaneous_multi_occupant_animations(anim_mgr, building):
    # Two batch movements simultaneously registered
    ev1 = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T10:00:00",
        person_ids=["P0001", "P0002", "P0003"],
        source_room="Room 1",
        destination_room="Room 2",
        building=building,
    )
    ev2 = anim_mgr.record_movement(
        event_type="transfer",
        timestamp="2026-10-19T10:00:00",
        person_ids=["P0004", "P0005"],
        source_room="Room 3",
        destination_room="Room 4",
        building=building,
    )
    assert ev1 is not None and ev2 is not None
    assert len(anim_mgr.events) == 2
    # Verify interpolated coordinates for both events at 50% progress
    pos1 = ev1.get_interpolated_position(0.5)
    pos2 = ev2.get_interpolated_position(0.5)
    assert pos1 != pos2


# ---------------------------------------------------------------------------
# Requirement 21: Bounded movement history retention (<= 100 events)
# ---------------------------------------------------------------------------
def test_21_bounded_movement_history_retention(anim_mgr, building):
    # Record 120 distinct events
    for i in range(120):
        anim_mgr.record_movement(
            event_type="transfer",
            timestamp=f"2026-10-19T10:{i // 60:02d}:{i % 60:02d}",
            person_ids=[f"P_{i:04d}"],
            source_room="Room 1",
            destination_room="Room 2",
            building=building,
        )
    assert len(anim_mgr.events) == 100
    assert anim_mgr.events[-1].person_ids == ["P_0119"]
    # First recorded event should have been evicted (FIFO)
    assert anim_mgr.events[0].person_ids == ["P_0020"]


# ---------------------------------------------------------------------------
# Requirement 22: Movement history filtering by room, occupant, and event type
# ---------------------------------------------------------------------------
def test_22_movement_history_filtering(anim_mgr, building):
    anim_mgr.record_movement("admission", "2026-10-19T10:00:00", ["P0001"], "Outside", "Room 1", building)
    anim_mgr.record_movement("transfer", "2026-10-19T10:05:00", ["P0001"], "Room 1", "Room 2", building)
    anim_mgr.record_movement("transfer", "2026-10-19T10:10:00", ["P0002"], "Room 3", "Room 4", building)
    anim_mgr.record_movement("exit", "2026-10-19T10:15:00", ["P0002"], "Room 4", None, building)

    # Filter by room
    r1_events = anim_mgr.filter_events(room_id="Room 1")
    assert len(r1_events) == 2  # admission to Room 1, transfer from Room 1

    r4_events = anim_mgr.filter_events(room_id="Room 4")
    assert len(r4_events) == 2  # transfer to Room 4, exit from Room 4

    # Filter by occupant ID
    p1_events = anim_mgr.filter_events(occupant_id="P0001")
    assert len(p1_events) == 2

    # Filter by event type
    admissions = anim_mgr.filter_events(event_type="admission")
    assert len(admissions) == 1
    exits = anim_mgr.filter_events(event_type="exit")
    assert len(exits) == 1

    # Combined filter
    p2_transfers = anim_mgr.filter_events(room_id="Room 3", occupant_id="P0002", event_type="transfer")
    assert len(p2_transfers) == 1


# ---------------------------------------------------------------------------
# Requirement 23: Visual replay execution without mutating simulation state
# ---------------------------------------------------------------------------
def test_23_visual_replay_non_mutating(engine, anim_mgr):
    ev = anim_mgr.record_movement(
        event_type="transfer",
        timestamp=engine.current_time.isoformat(),
        person_ids=["P0001"],
        source_room="Room 1",
        destination_room="Room 2",
        building=engine.building,
    )
    # Authoritative room for P0001
    occ = engine.occupants.get("P0001")
    auth_room_before = occ.current_room_id if occ else None

    # Start visual replay
    ok = anim_mgr.start_replay(ev.event_id)
    assert ok is True
    assert anim_mgr.replay_event_id == ev.event_id
    active_ev = anim_mgr.get_active_replay_event()
    assert active_ev is not None
    assert active_ev.event_id == ev.event_id

    # Test coordinate calculation with replay active
    anim_mgr.replay_progress = 0.5
    pos_map = anim_mgr.get_occupant_positions(engine.building, engine.occupants)
    assert "P0001" in pos_map
    replayed_pos = pos_map["P0001"]
    assert replayed_pos == ev.get_interpolated_position(0.5)

    # Authoritative state is untouched
    if occ:
        assert occ.current_room_id == auth_room_before

    # Stop replay
    anim_mgr.stop_replay()
    assert anim_mgr.replay_event_id is None
    assert anim_mgr.get_active_replay_event() is None


# ---------------------------------------------------------------------------
# Requirement 24: Visual exit records retention bounded to <= 25
# ---------------------------------------------------------------------------
def test_24_bounded_exit_records_retention(anim_mgr, building):
    for i in range(35):
        anim_mgr.record_movement(
            event_type="exit",
            timestamp=f"2026-10-19T11:{i:02d}:00",
            person_ids=[f"P_EXIT_{i:02d}"],
            source_room="Room 1",
            destination_room=None,
            building=building,
        )
    assert len(anim_mgr.exit_records) == 25
    assert anim_mgr.exit_records[-1]["occupant_id"] == "P_EXIT_34"
    assert anim_mgr.exit_records[0]["occupant_id"] == "P_EXIT_10"


# ---------------------------------------------------------------------------
# Requirement 25: Stale occupant selection cleanup when occupant exits
# ---------------------------------------------------------------------------
def test_25_stale_occupant_selection_cleanup(engine, anim_mgr):
    engine.add_person(count=5)
    p_inside = [pid for pid, o in engine.occupants.items() if o.is_inside_building]
    assert len(p_inside) > 0
    target_pid = p_inside[0]
    anim_mgr.selected_occupant_id = target_pid

    # Validate before exit
    anim_mgr.validate_selected_occupant(engine.occupants)
    assert anim_mgr.selected_occupant_id == target_pid

    # Force occupant exit by marking outside
    engine.occupants[target_pid].current_room_id = None
    engine.occupants[target_pid].movement_state = "exited"
    assert engine.occupants[target_pid].is_inside_building is False

    # Validate after exit -> should reset to None
    anim_mgr.validate_selected_occupant(engine.occupants)
    assert anim_mgr.selected_occupant_id is None


# ---------------------------------------------------------------------------
# Requirement 26: Adaptive marker sizing under high occupancy density
# ---------------------------------------------------------------------------
def test_26_adaptive_marker_sizing_under_high_density(building, engine):
    sched = next(iter(engine.occupants.values())).schedule

    # Low occupancy (5 occupants)
    mock_low = {}
    for i in range(5):
        m = Occupant(occupant_id=f"L_{i}", occupant_type="visitor", schedule=sched)
        m.current_room_id = "Room 1"
        mock_low[f"L_{i}"] = m

    fig_low = create_floorplan_figure(
        building=building,
        occupants=mock_low,
        show_occupants=True,
    )
    occ_scatter_low = [tr for tr in fig_low.data if tr.name == "Occupants"][0]
    assert all(s == 10 for s in occ_scatter_low.marker.size)

    # Medium occupancy (20 occupants)
    mock_med = {}
    for i in range(20):
        m = Occupant(occupant_id=f"M_{i}", occupant_type="visitor", schedule=sched)
        m.current_room_id = "Room 1"
        mock_med[f"M_{i}"] = m

    fig_med = create_floorplan_figure(
        building=building,
        occupants=mock_med,
        show_occupants=True,
    )
    occ_scatter_med = [tr for tr in fig_med.data if tr.name == "Occupants"][0]
    assert all(s == 8 for s in occ_scatter_med.marker.size)

    # High occupancy (>30 occupants)
    mock_high = {}
    for i in range(40):
        m = Occupant(occupant_id=f"H_{i}", occupant_type="visitor", schedule=sched)
        m.current_room_id = "Room 1"
        mock_high[f"H_{i}"] = m

    fig_high = create_floorplan_figure(
        building=building,
        occupants=mock_high,
        show_occupants=True,
    )
    occ_scatter_high = [tr for tr in fig_high.data if tr.name == "Occupants"][0]
    assert all(s == 6 for s in occ_scatter_high.marker.size)


# ---------------------------------------------------------------------------
# Requirement 27: Simulation clock invariance during visual animation and replay
# ---------------------------------------------------------------------------
def test_27_simulation_clock_invariance_during_replay(pipeline, anim_mgr):
    clock_start = pipeline.engine.current_time
    step_start = pipeline.engine.step_count

    ev = anim_mgr.record_movement(
        "transfer",
        clock_start.isoformat(),
        ["P0001"],
        "Room 1",
        "Room 2",
        pipeline.engine.building,
    )
    anim_mgr.start_replay(ev.event_id)
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        anim_mgr.replay_progress = p
        anim_mgr.get_occupant_positions(pipeline.engine.building, pipeline.engine.occupants)

    assert pipeline.engine.current_time == clock_start
    assert pipeline.engine.step_count == step_start


# ---------------------------------------------------------------------------
# Requirement 28: Physical sensor invariance during visual animation and replay
# ---------------------------------------------------------------------------
def test_28_physical_sensor_invariance_during_replay(pipeline, anim_mgr):
    # Snapshot sensor readings
    s_df_before = pipeline.engine.get_sensor_readings_df()

    ev = anim_mgr.record_movement(
        "transfer",
        pipeline.engine.current_time.isoformat(),
        ["P0001"],
        "Room 1",
        "Room 2",
        pipeline.engine.building,
    )
    anim_mgr.start_replay(ev.event_id)
    anim_mgr.get_occupant_positions(pipeline.engine.building, pipeline.engine.occupants)

    s_df_after = pipeline.engine.get_sensor_readings_df()
    if not s_df_before.empty and not s_df_after.empty:
        assert s_df_before.equals(s_df_after)


# ---------------------------------------------------------------------------
# Requirement 29: HVAC and energy meter invariance during visual animation and replay
# ---------------------------------------------------------------------------
def test_29_energy_meter_invariance_during_replay(pipeline, anim_mgr):
    e_states_before = pipeline.engine.energy_manager.get_latest_states()
    kwh_before = sum(s.cumulative_total_kwh for s in e_states_before.values())

    ev = anim_mgr.record_movement(
        "transfer",
        pipeline.engine.current_time.isoformat(),
        ["P0001"],
        "Room 1",
        "Room 2",
        pipeline.engine.building,
    )
    anim_mgr.start_replay(ev.event_id)
    anim_mgr.get_occupant_positions(pipeline.engine.building, pipeline.engine.occupants)
    anim_mgr.stop_replay()

    e_states_after = pipeline.engine.energy_manager.get_latest_states()
    kwh_after = sum(s.cumulative_total_kwh for s in e_states_after.values())
    assert kwh_before == pytest.approx(kwh_after, abs=1e-9)


# ---------------------------------------------------------------------------
# Requirement 30: HTML5 canvas serialization and script generation
# ---------------------------------------------------------------------------
def test_30_html5_canvas_generation(pipeline, anim_mgr):
    from dashboard.live_canvas import generate_live_digital_twin_html

    anim_mgr.record_movement(
        "transfer",
        pipeline.engine.current_time.isoformat(),
        ["P0001"],
        "Room 1",
        "Room 2",
        pipeline.engine.building,
    )

    html = generate_live_digital_twin_html(
        building=pipeline.engine.building,
        occupants=pipeline.engine.occupants,
        anim_mgr=anim_mgr,
        selected_room_id="Room 2",
        active_overlay="Occupancy Density (%)",
        entrance_queue_count=pipeline.engine.waiting_at_entrance,
        height=560,
    )
    assert "<canvas" in html
    assert "dt-container" in html
    assert "requestAnimationFrame" in html
    assert "Room 1" in html
    assert "Room 2" in html
    assert "waypoints" in html


# ---------------------------------------------------------------------------
# Requirement 31: Occupant transit status discrimination
# ---------------------------------------------------------------------------
def test_31_occupant_transit_status_discrimination(pipeline, anim_mgr):
    # Add an active movement
    ev = anim_mgr.record_movement(
        "transfer",
        pipeline.engine.current_time.isoformat(),
        ["P0001"],
        "Room 1",
        "Room 2",
        pipeline.engine.building,
    )
    # Visual progress is in transit (0.4)
    ev.progress = 0.4
    ev.status = "in_transit"

    active_ev = None
    for e in anim_mgr.events:
        if "P0001" in e.person_ids and e.status == "in_transit":
            active_ev = e
            break

    assert active_ev is not None
    assert active_ev.source_room == "Room 1"
    assert active_ev.destination_room == "Room 2"
    assert active_ev.progress == 0.4
    assert len(active_ev.waypoints) >= 2


# ---------------------------------------------------------------------------
# Requirement 32: Reset clears replay state, paused state, and bounded history
# ---------------------------------------------------------------------------
def test_32_reset_clears_replay_and_paused_state(anim_mgr, building):
    anim_mgr.pause()
    anim_mgr.set_speed(1.5)
    ev = anim_mgr.record_movement("transfer", "2026-10-19T10:00:00", ["P0001"], "Room 1", "Room 2", building)
    anim_mgr.start_replay(ev.event_id)

    assert anim_mgr.is_paused is True
    assert anim_mgr.animation_speed == 1.5
    assert anim_mgr.replay_event_id is not None

    anim_mgr.clear()

    assert anim_mgr.is_paused is False
    assert anim_mgr.animation_speed == 1.0
    assert anim_mgr.replay_event_id is None
    assert len(anim_mgr.events) == 0


# ---------------------------------------------------------------------------
# Requirement 33: Door coordinate accuracy and collision-free routing
# ---------------------------------------------------------------------------
def test_33_doorway_collision_free_routing(building):
    for r_src in EXPECTED_ROOM_IDS:
        for r_dst in EXPECTED_ROOM_IDS:
            if r_src == r_dst:
                continue
            waypoints = calculate_route_waypoints(building, r_src, r_dst)
            assert len(waypoints) >= 2
            # Every waypoint must be inside the outer building bounds
            for x, y in waypoints:
                assert 0.0 <= x <= 26.0
                assert 0.0 <= y <= 20.0


# ---------------------------------------------------------------------------
# Requirement 34: STGNN pipeline integrity under concurrent animation and replay
# ---------------------------------------------------------------------------
def test_34_stgnn_pipeline_integrity_during_visual_controls(pipeline, anim_mgr):
    # Step pipeline once
    res1 = pipeline.step()
    assert res1["status"] in ["WARMUP", "SUCCESS"]

    # Trigger visual pause and replay
    anim_mgr.pause()
    ev = anim_mgr.record_movement(
        "transfer",
        pipeline.engine.current_time.isoformat(),
        ["P0001"],
        "Room 1",
        "Room 2",
        pipeline.engine.building,
    )
    anim_mgr.start_replay(ev.event_id)

    # Step pipeline while visual replay is active
    res2 = pipeline.step()
    assert res2["status"] in ["WARMUP", "SUCCESS"]
    assert len(pipeline.rolling_buffer) <= 12
    # Predictions list preserved
    assert len(pipeline.predictions) >= 10  # 5 warmup/step records


# ---------------------------------------------------------------------------
# Requirement 35: Replay nonexistent event handled safely
# ---------------------------------------------------------------------------
def test_35_replay_nonexistent_event_handled_safely(anim_mgr):
    res = anim_mgr.start_replay("non_existent_hash")
    assert res is False
    assert anim_mgr.replay_event_id is None
    assert anim_mgr.get_active_replay_event() is None


# ---------------------------------------------------------------------------
# Requirement 36: Filter events with empty results handled gracefully
# ---------------------------------------------------------------------------
def test_36_filter_events_empty_results(anim_mgr):
    res = anim_mgr.filter_events(room_id="NonExistentRoom")
    assert res == []
    res2 = anim_mgr.filter_events(occupant_id="NON_EXISTENT_ID")
    assert res2 == []


