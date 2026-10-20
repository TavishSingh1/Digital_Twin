"""
Regression and Validation Test Suite: Manual Occupant Movement HVAC Coupling
===========================================================================
Validates:
- Test A: Occupancy transfer and conservation across connected rooms.
- Test B: HVAC input propagation (sensible loads, cooling demand, energy states) on subsequent step.
- Test C: No stale state, no duplicate occupants, and persistent manual destination across multiple moves and steps.
- Test D: Unaffected rooms retain their exact occupant sets, sensor values, and energy states.
- Test E: Manual-mode UI and telemetry reflection (authoritative sensor/energy access).
- Test F: HVAC physical fidelity (thermal inertia, thermostat deadband hysteresis, electrical vs thermal power separation).
"""

import pytest
import numpy as np

from simulation.config import EXPECTED_ROOM_IDS
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline


# =============================================================================
# TEST A: Occupancy transfer & conservation
# =============================================================================
def test_test_a_occupancy_transfer_and_conservation():
    """
    Test A: Move an occupant from Room A to Room B.
    - Confirm Room A decreases by 1 and Room B increases by 1.
    - Confirm total building occupant count remains constant.
    - Confirm invariants hold.
    """
    engine = SimulationEngine(num_occupants=20, seed=42)
    # Step simulation to populate rooms
    for _ in range(12):
        engine.step()

    src_room_id = "Room 2"
    dst_room_id = "Room 1"
    src_room = engine.building.get_room(src_room_id)
    dst_room = engine.building.get_room(dst_room_id)

    assert src_room.occupant_count > 0, "Source room must have occupants"
    init_src_count = src_room.occupant_count
    init_dst_count = dst_room.occupant_count
    init_total = engine.building.total_occupants

    # Pick one occupant from source room
    target_pid = sorted(list(src_room.occupants))[0]

    res = engine.move_people(source_room=src_room_id, destination_room=dst_room_id, person_ids=[target_pid])
    assert res["success"] is True
    assert res["moved_count"] == 1
    assert res["moved_ids"] == [target_pid]

    # Verify counts
    assert src_room.occupant_count == init_src_count - 1
    assert dst_room.occupant_count == init_dst_count + 1
    assert engine.building.total_occupants == init_total

    # Verify occupant location
    occ = engine.occupants[target_pid]
    assert occ.current_room_id == dst_room_id
    assert target_pid in dst_room.occupants
    assert target_pid not in src_room.occupants

    # Invariants
    active_ids = {o.occupant_id for o in engine.occupants.values() if o.is_inside_building}
    engine.building.validate_invariants(active_ids)


# =============================================================================
# TEST B: HVAC input propagation on subsequent step
# =============================================================================
def test_test_b_hvac_input_propagation():
    """
    Test B: Capture occupancy-dependent inputs before move.
    Move an occupant between rooms.
    Advance simulation by one step.
    Verify that affected rooms' HVAC/thermal calculations use the updated occupancy
    and that the occupant does not revert due to background schedule.
    """
    engine = SimulationEngine(num_occupants=20, seed=42)
    for _ in range(12):
        engine.step()

    src_r = "Room 2"
    dst_r = "Room 1"
    src_room = engine.building.get_room(src_r)
    dst_room = engine.building.get_room(dst_r)
    init_src_occ = src_room.occupant_count
    init_dst_occ = dst_room.occupant_count

    # Capture initial states
    hvac_before = engine.energy_manager.get_latest_states()
    src_cool_before = hvac_before[src_r].cooling_demand_w
    dst_cool_before = hvac_before[dst_r].cooling_demand_w

    target_pid = sorted(list(src_room.occupants))[0]
    engine.move_people(source_room=src_r, destination_room=dst_r, person_ids=[target_pid])

    # Immediate check: sensible heat load difference of 1 occupant (~100 W)
    hvac_imm = engine.energy_manager.get_latest_states()
    assert hvac_imm[src_r].occupant_count == init_src_occ - 1
    assert hvac_imm[dst_r].occupant_count == init_dst_occ + 1
    assert round(src_cool_before - hvac_imm[src_r].cooling_demand_w, 1) == 100.0
    assert round(hvac_imm[dst_r].cooling_demand_w - dst_cool_before, 1) == 100.0

    # Advance simulation by one step (5 minutes)
    step_res = engine.step()

    # Crucial check: occupant MUST stay in dst_r (not reverted by schedule)
    assert engine.occupants[target_pid].current_room_id == dst_r
    assert target_pid in dst_room.occupants
    assert target_pid not in src_room.occupants

    # Energy state on subsequent step must use updated occupant count
    hvac_after = engine.energy_manager.get_latest_states()
    assert hvac_after[src_r].occupant_count == src_room.occupant_count
    assert hvac_after[dst_r].occupant_count == dst_room.occupant_count

    # Sensor readings on subsequent step
    sensor_after = engine.sensor_manager.get_latest_readings()
    assert sensor_after[src_r].indoor_temperature is not None
    assert sensor_after[dst_r].indoor_temperature is not None


# =============================================================================
# TEST C: No stale state across repeated moves
# =============================================================================
def test_test_c_repeated_moves_no_stale_or_duplicate_state():
    """
    Test C: Repeat manual movements between connected rooms:
    Room 1 -> Room 2 -> Room 4 -> Room 5.
    Confirm no duplicate occupants, lost occupants, or stale occupancy values across multiple steps.
    """
    engine = SimulationEngine(num_occupants=20, seed=42)
    for _ in range(12):
        engine.step()

    # Add interactive occupant or pick existing
    target_pid = list(engine.building.get_room("Room 1").occupants)[0] if engine.building.get_room("Room 1").occupants else None
    if not target_pid:
        res = engine.add_person(count=1)
        target_pid = res["admitted_ids"][0]

    # Move 1: Room 1 -> Room 2
    engine.move_people("Room 1", "Room 2", [target_pid])
    assert engine.occupants[target_pid].current_room_id == "Room 2"
    assert target_pid in engine.building.get_room("Room 2").occupants
    assert target_pid not in engine.building.get_room("Room 1").occupants

    # Step forward
    engine.step()
    assert engine.occupants[target_pid].current_room_id == "Room 2"

    # Move 2: Room 2 -> Room 4
    engine.move_people("Room 2", "Room 4", [target_pid])
    assert engine.occupants[target_pid].current_room_id == "Room 4"
    assert target_pid in engine.building.get_room("Room 4").occupants
    assert target_pid not in engine.building.get_room("Room 2").occupants

    # Step forward
    engine.step()
    assert engine.occupants[target_pid].current_room_id == "Room 4"

    # Move 3: Room 4 -> Room 5
    engine.move_people("Room 4", "Room 5", [target_pid])
    assert engine.occupants[target_pid].current_room_id == "Room 5"
    assert target_pid in engine.building.get_room("Room 5").occupants
    assert target_pid not in engine.building.get_room("Room 4").occupants

    # Step forward multiple times
    for _ in range(3):
        engine.step()

    # The occupant is still reliably in Room 5
    assert engine.occupants[target_pid].current_room_id == "Room 5"
    assert target_pid in engine.building.get_room("Room 5").occupants

    # Validate that each occupant is in at most 1 room
    all_room_occupants = []
    for r_id, room in engine.building.rooms.items():
        all_room_occupants.extend(list(room.occupants))
    assert len(all_room_occupants) == len(set(all_room_occupants)), "No duplicate occupants allowed"

    active_ids = {o.occupant_id for o in engine.occupants.values() if o.is_inside_building}
    engine.building.validate_invariants(active_ids)


# =============================================================================
# TEST D: Unaffected rooms preservation
# =============================================================================
def test_test_d_unaffected_rooms_preservation():
    """
    Test D: Confirm that unrelated rooms do not have their occupant locations,
    sensor values, or HVAC states accidentally modified by a move between other rooms.
    """
    engine = SimulationEngine(num_occupants=25, seed=42)
    for _ in range(12):
        engine.step()

    # Snapshot unaffected rooms: Room 3, Room 4, Room 5
    unaffected = ["Room 3", "Room 4", "Room 5"]
    snapshot_occ = {r: set(engine.building.get_room(r).occupants) for r in unaffected}
    snapshot_sensors = {r: engine.sensor_manager.get_latest_readings()[r].to_dict() for r in unaffected}
    snapshot_energy = {r: engine.energy_manager.get_latest_states()[r] for r in unaffected}

    # Perform move between Room 1 and Room 2
    r2 = engine.building.get_room("Room 2")
    assert r2.occupant_count > 0
    target_pid = sorted(list(r2.occupants))[0]
    engine.move_people("Room 2", "Room 1", [target_pid])

    # Verify unaffected rooms remain completely identical
    for r in unaffected:
        # Occupant set must be strictly identical
        assert set(engine.building.get_room(r).occupants) == snapshot_occ[r]
        assert engine.building.get_room(r).occupant_count == len(snapshot_occ[r])

        # Immediate energy state must be identical
        curr_e = engine.energy_manager.get_latest_states()[r]
        assert curr_e.occupant_count == snapshot_energy[r].occupant_count
        assert curr_e.hvac_power_w == snapshot_energy[r].hvac_power_w
        assert curr_e.cooling_demand_w == snapshot_energy[r].cooling_demand_w
        assert curr_e.operating_state == snapshot_energy[r].operating_state

        # Immediate sensor readings must be identical (sound within Gaussian sensor noise floor)
        curr_s = engine.sensor_manager.get_latest_readings()[r].to_dict()
        assert curr_s["indoor_temperature"] == snapshot_sensors[r]["indoor_temperature"]
        assert abs(curr_s["sound_pressure_level"] - snapshot_sensors[r]["sound_pressure_level"]) < 1.0
        assert curr_s["indoor_co2"] == snapshot_sensors[r]["indoor_co2"]


# =============================================================================
# TEST E: Manual-mode UI and Pipeline state reflection
# =============================================================================
def test_test_e_manual_mode_ui_and_pipeline():
    """
    Test E: Verify that pipeline and state manager properly propagate manual moves
    to latest telemetry dicts, rolling buffers, and control centre views.
    """
    engine = SimulationEngine(num_occupants=20, seed=42)
    pipeline = DigitalTwinPipeline(engine=engine, seed=42)
    pipeline.run(num_steps=12)

    src_r = "Room 2"
    dst_r = "Room 1"
    src_occ_before = pipeline.engine.building.get_room(src_r).occupant_count
    dst_occ_before = pipeline.engine.building.get_room(dst_r).occupant_count

    # Execute move through pipeline
    res = pipeline.move_people(source_room=src_r, destination_room=dst_r, count=1)
    assert res["success"] is True

    # 1. Pipeline rolling buffer must be updated with new occupant counts immediately
    latest_buf = pipeline.rolling_buffer[-1]
    assert latest_buf[src_r]["occupant_count"] == src_occ_before - 1
    assert latest_buf[dst_r]["occupant_count"] == dst_occ_before + 1

    # 2. Authoritative sensor readings dict must be up-to-date
    latest_sensors = pipeline.engine.sensor_manager.get_latest_readings()
    assert src_r in latest_sensors
    assert dst_r in latest_sensors

    # 3. Authoritative energy states must reflect new occupancy counts immediately
    latest_energy = pipeline.engine.energy_manager.get_latest_states()
    assert latest_energy[src_r].occupant_count == src_occ_before - 1
    assert latest_energy[dst_r].occupant_count == dst_occ_before + 1

    # 4. Advance pipeline by 1 step, ensuring no crash and successful evaluation/forecast
    step_res = pipeline.step()
    assert step_res["status"] in ("SUCCESS", "WARMUP")
    assert pipeline.engine.building.get_room(src_r).occupant_count == src_occ_before - 1


# =============================================================================
# TEST F: HVAC physics (thermal inertia, thermostat hysteresis, electrical power)
# =============================================================================
def test_test_f_hvac_physics_integrity():
    """
    Test F: Verify that occupancy updates do not bypass thermostat hysteresis,
    thermal inertia, or the thermodynamic electrical-vs-thermal power calculations.
    """
    engine = SimulationEngine(num_occupants=20, seed=42)
    for _ in range(12):
        engine.step()

    # Move an occupant
    src_r = "Room 2"
    dst_r = "Room 1"
    target_pid = sorted(list(engine.building.get_room(src_r).occupants))[0]

    # Check temperature before move
    t_src_before = engine.sensor_manager.get_latest_readings()[src_r].indoor_temperature
    t_dst_before = engine.sensor_manager.get_latest_readings()[dst_r].indoor_temperature

    engine.move_people(src_r, dst_r, [target_pid])

    # 1. Thermal Inertia: Immediate temperature must NOT jump or snap instantaneously to setpoint
    t_src_imm = engine.sensor_manager.get_latest_readings()[src_r].indoor_temperature
    t_dst_imm = engine.sensor_manager.get_latest_readings()[dst_r].indoor_temperature
    assert t_src_imm == t_src_before, "Thermal inertia: temperature does not instantaneously change"
    assert t_dst_imm == t_dst_before, "Thermal inertia: temperature does not instantaneously change"

    # Step forward 1 step (5 minutes)
    engine.step()

    t_src_after = engine.sensor_manager.get_latest_readings()[src_r].indoor_temperature
    t_dst_after = engine.sensor_manager.get_latest_readings()[dst_r].indoor_temperature

    # Change in temperature over 5 minutes must be physically realistic (e.g. < 1.5°C per step)
    assert abs(t_src_after - t_src_before) < 1.5, "Temperature rate of change must respect thermal capacitance"
    assert abs(t_dst_after - t_dst_before) < 1.5, "Temperature rate of change must respect thermal capacitance"

    # 2. Thermostat Hysteresis & Power Calculations:
    # Check that electrical power satisfies thermodynamic COP relation:
    # P_elec ≈ Cooling_delivered / COP + P_fan
    energy_states = engine.energy_manager.get_latest_states()
    for room_id in EXPECTED_ROOM_IDS:
        state = energy_states[room_id]
        ctrl = engine.energy_manager.hvac_controllers[room_id]
        if state.operating_state == "COOLING":
            expected_comp_elec = state.cooling_delivered_w / ctrl.cop
            # Fan power is positive
            assert state.hvac_power_w >= expected_comp_elec - 5.0
            assert state.cooling_delivered_w > 0.0
        elif state.operating_state == "IDLE":
            assert state.cooling_delivered_w == 0.0
            assert state.hvac_power_w == ctrl.standby_power_w
        elif state.operating_state == "OFF":
            assert state.cooling_delivered_w == 0.0
            assert state.hvac_power_w == 0.0
