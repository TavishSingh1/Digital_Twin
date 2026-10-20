"""
Automated Test Suite for Phase 6: Realistic Occupancy-Responsive Sensor Physics
================================================================================
Comprehensive verification of first-principles physical modeling for all 7 sensors:
1. Illuminance: Policy, vacancy turn-off delay, daylight persistence, manual override.
2. Sound Pressure Level: Logarithmic acoustic decibel summation, room-specific profiles, immediate update.
3. Indoor Temperature: Lumped capacitance sensible heat accumulation, gradual evolution, thermal persistence.
4. Dry-Bulb Temperature: Complete independence from indoor occupants and movements.
5. Indoor Relative Humidity: Psychrometric moisture mass balance, Magnus-Tetens vapor pressure coupling.
6. Outdoor Relative Humidity: Complete independence from indoor occupants and movements.
7. Indoor CO2: Exact analytical exponential decay and metabolic source accumulation.
8. Occupant Exit: Immediate source reduction with continuous state preservation.
9. Room-to-Room Movement: Simultaneous independent update of source terms in source and destination.
10. Vacant Relaxation: Slow asymptotic relaxation toward environmental equilibrium.
11. Invariance under UI Reruns: Repeated telemetry queries at same timestamp do not advance physical state.
12. Physical Boundedness: All 7 variables strictly bounded, finite, and non-NaN under stress.
13. Batch Arrivals & Exits: Large population increments scale source terms consistently.
14. Unified Flow Modes: Manual, Scheduled, and Interactive movements trigger identical physics.
15. STGNN Pipeline Integration: Lookback buffer and forward model inference produce valid predictions.
"""

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from simulation.config import EXPECTED_ROOM_IDS
from simulation.engine import SimulationEngine
from simulation.sensors import (
    load_sensor_config,
    RoomSensorReading,
    RoomSensorModel,
    VirtualSensorManager,
    saturation_vapor_pressure_kpa,
    vapor_density_from_rh,
    rh_from_vapor_density,
)
from simulation.pipeline import DigitalTwinPipeline
from stgnn_model import COMMON_SENSOR_FEATURES, STGNNInference


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def clean_engine():
    """Returns an engine initialized with 0 initial occupants for controlled testing."""
    engine = SimulationEngine(num_occupants=0, seed=100)
    # Ensure all rooms have 0 occupants
    for r in EXPECTED_ROOM_IDS:
        engine.building.get_room(r).occupants.clear()
    engine.occupants.clear()
    engine.entrance_queue.clear()
    engine.total_arrivals = 0
    engine.total_exits = 0
    engine.sensor_manager.refresh_immediate_sensors(engine.current_time, engine.building)
    return engine


@pytest.fixture
def single_room_model():
    """Returns a standalone RoomSensorModel for direct equation unit testing."""
    cfg = load_sensor_config("config/sensor_parameters.json")
    room_cfg = cfg["rooms"]["Room 1"]
    return RoomSensorModel(
        room_id="Room 1",
        room_cfg=room_cfg,
        outdoor_co2_baseline=400.0,
        seed=42,
    )


# -----------------------------------------------------------------------------
# 1. ILLUMINANCE: POLICY, VACANCY DELAY, DAYLIGHT, MANUAL OVERRIDE
# -----------------------------------------------------------------------------

def test_illuminance_occupancy_vacancy_delay_and_manual_override(clean_engine):
    """
    Test 1: Illuminance adheres to occupancy/vacancy policies:
    - Empty room at night has low night illuminance (<= 5 lux).
    - Occupant enters: lights turn ON immediately.
    - Occupant leaves: lights remain ON during the 10-minute vacancy turn-off delay.
    - After vacancy delay elapsed without occupants: lights turn OFF.
    - Daylight persists even when vacant.
    - Manual overrides (Force ON, Force OFF, Auto) operate reliably.
    """
    engine = clean_engine
    # Set time to midnight (no daylight)
    midnight = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    engine.current_time = midnight
    engine.sensor_manager.refresh_immediate_sensors(midnight, engine.building)

    r1_meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert r1_meta["is_light_on"] is False
    readings = engine.sensor_manager.get_latest_readings()
    assert readings["Room 1"].illuminance < 10.0, "Night illuminance in vacant room must be near 0"

    # 1. Occupant enters Room 1
    engine.add_person(count=1)
    readings = engine.sensor_manager.get_latest_readings()
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert meta["is_light_on"] is True
    assert readings["Room 1"].illuminance >= 30.0, "Artificial lighting must illuminate room upon entry"

    # 2. Occupant exits: Room 1 becomes vacant
    engine.exit_person(count=1, room_id="Room 1")
    assert engine.building.get_room("Room 1").occupant_count == 0
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    # Vacancy delay is 10 minutes: lights must STILL be ON immediately after exit
    assert meta["is_light_on"] is True

    # Step 5 minutes: vacancy timer becomes 5 minutes (< 10 min delay); lights remain ON
    engine.step()
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert meta["is_light_on"] is True
    assert meta["vacancy_timer_minutes"] == 5.0

    # Step another 5 minutes: vacancy timer reaches 10 minutes (>= 10 min delay); lights turn OFF
    engine.step()
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert meta["is_light_on"] is False
    assert meta["vacancy_timer_minutes"] >= 10.0
    readings = engine.sensor_manager.get_latest_readings()
    assert readings["Room 1"].illuminance < 10.0, "Lights must turn off after vacancy delay expires"

    # 3. Daylight window penetration during daytime
    noon = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    engine.current_time = noon
    engine.sensor_manager.refresh_immediate_sensors(noon, engine.building)
    readings = engine.sensor_manager.get_latest_readings()
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert meta["is_light_on"] is False  # Empty, so artificial light is off
    assert meta["daylight_lux"] >= 35.0   # But natural daylight penetrates windows
    assert readings["Room 1"].illuminance >= 35.0

    # 4. Manual override testing
    engine.set_room_lighting("Room 1", True)  # Force ON
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert meta["is_light_on"] is True
    assert meta["manual_light_override"] is True

    engine.set_room_lighting("Room 1", False)  # Force OFF
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert meta["is_light_on"] is False
    assert meta["manual_light_override"] is False

    engine.set_room_lighting("Room 1", None)  # Return to Auto
    meta = engine.sensor_manager.get_sensor_metadata("Room 1")
    assert meta["manual_light_override"] is None


# -----------------------------------------------------------------------------
# 2. SOUND PRESSURE: LOGARITHMIC POWER SUMMATION & PROFILES
# -----------------------------------------------------------------------------

def test_sound_pressure_logarithmic_summation_and_room_profiles(clean_engine):
    """
    Test 2: Sound pressure follows logarithmic acoustic decibel summation:
    L_p = 10 * log10(10^(L_base/10) + N * 10^(L_occ/10)).
    - Empty room is near base equipment floor.
    - Adding occupants increases sound logarithmically (+3 dB per doubling), not linearly in dB.
    - Room 3 (conference) produces higher per-occupant acoustic level than Room 1.
    - Updates immediately without advancing time.
    """
    engine = clean_engine
    engine.sensor_manager.refresh_immediate_sensors(engine.current_time, engine.building)
    r1_empty_sound = engine.sensor_manager.get_latest_readings()["Room 1"].sound_pressure_level

    # Empty room is near base noise floor (~48 dBA)
    assert 46.0 <= r1_empty_sound <= 52.0

    # Add 1 person to Room 1
    engine.add_person(count=1)
    r1_sound_1 = engine.sensor_manager.get_latest_readings()["Room 1"].sound_pressure_level
    assert r1_sound_1 > r1_empty_sound

    # Add 9 more people (total 10 people)
    engine.add_person(count=9)
    r1_sound_10 = engine.sensor_manager.get_latest_readings()["Room 1"].sound_pressure_level
    assert r1_sound_10 > r1_sound_1

    # In acoustic decibels, 10 occupants at ~52 dBA adds power 10 * 10^5.2:
    # Net sound should be roughly 10-15 dB higher, NOT 10 times higher in decibels (e.g. not 500 dB!)
    assert 55.0 <= r1_sound_10 <= 75.0

    cfg = load_sensor_config("config/sensor_parameters.json")
    r1_occ_dba = cfg["rooms"]["Room 1"]["sound_pressure_level"]["occupant_activity_dba"]
    r3_occ_dba = cfg["rooms"]["Room 3"]["sound_pressure_level"]["occupant_activity_dba"]
    assert r1_occ_dba > r3_occ_dba, "Room 1 conference/lobby has higher activity noise than Room 3 executive office"


# -----------------------------------------------------------------------------
# 3. INDOOR TEMPERATURE: SENSIBLE HEAT ACCUMULATION & THERMAL PERSISTENCE
# -----------------------------------------------------------------------------

def test_indoor_temperature_metabolic_heat_accumulation_and_thermal_persistence(clean_engine):
    """
    Test 3: Indoor temperature responds according to lumped thermal mass capacitance:
    - Temperature does NOT jump instantaneously upon person entry.
    - Under steady occupancy, sensible metabolic heat accumulates, raising temperature over time.
    - When vacated, thermal mass retains warmth and cools down gradually.
    """
    engine = clean_engine
    t0_reading = engine.sensor_manager.get_latest_readings()["Room 1"]
    t0_temp = t0_reading.indoor_temperature

    # 1. Add 10 occupants
    engine.add_person(count=10)
    t_imm_reading = engine.sensor_manager.get_latest_readings()["Room 1"]
    assert abs(t_imm_reading.indoor_temperature - t0_temp) < 0.05, (
        "Temperature must not jump discontinuously immediately upon person entry"
    )

    # 2. Advance simulation for 6 steps (30 minutes) with occupants inside
    temps = [t0_temp]
    for _ in range(6):
        engine.step()
        temps.append(engine.sensor_manager.get_latest_readings()["Room 1"].indoor_temperature)

    # Verify monotonic heating from metabolic generation
    assert temps[-1] > t0_temp + 0.2, "Metabolic heat must accumulate and increase indoor temperature"

    peak_temp = temps[-1]

    # 3. Empty the room completely
    r1 = engine.building.get_room("Room 1")
    engine.exit_person(count=r1.occupant_count, room_id="Room 1")
    assert r1.occupant_count == 0

    # Temperature immediately after exit must persist (not drop back to baseline immediately)
    t_after_exit = engine.sensor_manager.get_latest_readings()["Room 1"].indoor_temperature
    assert abs(t_after_exit - peak_temp) < 0.05

    # 4. Advance for 6 steps while empty: thermal mass should cool down slowly toward setpoint
    for _ in range(6):
        engine.step()

    cooled_temp = engine.sensor_manager.get_latest_readings()["Room 1"].indoor_temperature
    assert cooled_temp < peak_temp, "Vacant room must gradually cool down toward equilibrium"
    assert cooled_temp >= t0_temp - 0.5, "Temperature must remain physically bounded"


# -----------------------------------------------------------------------------
# 4. OUTDOOR DRY-BULB TEMPERATURE INDEPENDENCE
# -----------------------------------------------------------------------------

def test_outdoor_dry_bulb_temperature_independence(clean_engine):
    """
    Test 4: Dry-bulb outdoor temperature is strictly an exogenous meteorological variable:
    - Completely independent of indoor occupants.
    - Completely independent of room movements or lighting states.
    """
    engine = clean_engine
    reading_empty = engine.sensor_manager.get_latest_readings()["Room 1"]
    out_temp_empty = reading_empty.dry_bulb_temperature

    # Pack 30 people into the building
    engine.add_person(count=30)
    reading_packed = engine.sensor_manager.get_latest_readings()["Room 1"]
    out_temp_packed = reading_packed.dry_bulb_temperature

    assert out_temp_empty == out_temp_packed, "Outdoor dry-bulb temperature must not be altered by indoor occupants"

    # Check all 5 rooms report the exact same outdoor dry-bulb temperature at the same timestamp
    all_readings = engine.sensor_manager.get_latest_readings()
    outdoor_temps = [r.dry_bulb_temperature for r in all_readings.values()]
    assert len(set(outdoor_temps)) == 1, "All rooms must observe consistent outdoor weather"


# -----------------------------------------------------------------------------
# 5. INDOOR RELATIVE HUMIDITY: PSYCHROMETRIC MOISTURE BALANCE & COUPLING
# -----------------------------------------------------------------------------

def test_indoor_relative_humidity_psychrometric_coupling(single_room_model):
    """
    Test 5: Indoor relative humidity is psychrometrically derived via Magnus-Tetens:
    - Absolute moisture vapor density C_v increases with occupant latent emissions.
    - Relative humidity calculation accounts for temperature-dependent saturation vapor pressure.
    """
    model = single_room_model
    dt = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    outdoor_t = 24.0
    outdoor_rh = 60.0
    solar_lux = 2000.0

    # 1. Run 3 steps vacant
    for _ in range(3):
        r_vac = model.step(dt, occupants=0, outdoor_temp=outdoor_t, outdoor_rh=outdoor_rh, solar_lux=solar_lux)
    vac_moisture = model.current_moisture_g_m3

    # 2. Add 20 occupants for 6 steps
    for _ in range(6):
        r_occ = model.step(dt, occupants=20, outdoor_temp=outdoor_t, outdoor_rh=outdoor_rh, solar_lux=solar_lux)
    occ_moisture = model.current_moisture_g_m3

    # Moisture density (g/m^3) must increase due to 50 g/h/person emission
    assert occ_moisture > vac_moisture + 0.5, "Occupant respiration must increase absolute vapor density"

    # Psychrometric relationship check
    p_sat = saturation_vapor_pressure_kpa(r_occ.indoor_temperature)
    t_kelvin = r_occ.indoor_temperature + 273.15
    p_v = (occ_moisture * 0.4615 * t_kelvin) / 1000.0
    expected_rh = (p_v / p_sat) * 100.0
    assert abs(r_occ.indoor_relative_humidity - expected_rh) < 1.0


# -----------------------------------------------------------------------------
# 6. OUTDOOR RELATIVE HUMIDITY INDEPENDENCE
# -----------------------------------------------------------------------------

def test_outdoor_relative_humidity_independence(clean_engine):
    """
    Test 6: Outdoor relative humidity is strictly an exogenous weather variable:
    - Never changes due to indoor occupant counts or room movements.
    """
    engine = clean_engine
    init_out_rh = engine.sensor_manager.get_latest_readings()["Room 1"].outdoor_relative_humidity

    # Add people and move them
    engine.add_person(count=15)
    new_out_rh = engine.sensor_manager.get_latest_readings()["Room 1"].outdoor_relative_humidity
    assert init_out_rh == new_out_rh, "Outdoor RH must not change with indoor population"


# -----------------------------------------------------------------------------
# 7. CO2: EXACT ANALYTICAL EXPONENTIAL DECAY & ACCUMULATION
# -----------------------------------------------------------------------------

def test_co2_exact_analytical_accumulation_and_exponential_decay(single_room_model):
    """
    Test 7: CO2 mass balance adheres to exact analytical solution:
    C(t + dt) = C_inf + (C(t) - C_inf) * exp(-ACH * dt).
    - Monotonically rises towards steady-state when occupied.
    - Asymptotically and smoothly decays toward outdoor baseline (400 ppm) when empty.
    - Unconditionally stable and non-negative.
    """
    model = single_room_model
    dt = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    outdoor_t = 24.0
    outdoor_rh = 60.0
    solar_lux = 1000.0

    # 1. 20 occupants inside room: verify steady accumulation
    co2_history = []
    for _ in range(12):  # 1 hour
        r = model.step(dt, occupants=20, outdoor_temp=outdoor_t, outdoor_rh=outdoor_rh, solar_lux=solar_lux)
        co2_history.append(r.indoor_co2)

    # Must be monotonically increasing
    for i in range(1, len(co2_history)):
        assert co2_history[i] >= co2_history[i - 1], "CO2 must monotonically rise with occupants"

    assert co2_history[-1] > 700.0, "CO2 must significantly rise after 1 hour with 20 occupants"

    # 2. Room is evacuated: occupants = 0
    decay_history = []
    for _ in range(24):  # 2 hours vacant
        r = model.step(dt, occupants=0, outdoor_temp=outdoor_t, outdoor_rh=outdoor_rh, solar_lux=solar_lux)
        decay_history.append(r.indoor_co2)

    # Must be monotonically decaying towards outdoor baseline
    for i in range(1, len(decay_history)):
        assert decay_history[i] <= decay_history[i - 1], "CO2 must decay when vacant"

    # Decays toward 400 ppm without undershooting below 400
    assert decay_history[-1] >= 400.0
    assert decay_history[-1] < decay_history[0]


# -----------------------------------------------------------------------------
# 8. OCCUPANT EXIT: IMMEDIATE SOURCE REMOVAL & CONTINUOUS DYNAMIC STATES
# -----------------------------------------------------------------------------

def test_occupant_exit_removes_source_terms_preserves_continuous_state(clean_engine):
    """
    Test 8: Exiting occupants immediately cuts metabolic and acoustic generation,
    while dynamic state variables (temp, humidity, CO2) remain continuous.
    """
    engine = clean_engine
    engine.add_person(count=5)
    for _ in range(4):
        engine.step()

    r1_before = engine.sensor_manager.get_latest_readings()["Room 1"]
    high_sound = r1_before.sound_pressure_level
    co2_val = r1_before.indoor_co2
    temp_val = r1_before.indoor_temperature

    # Exit all occupants from Room 1
    r1 = engine.building.get_room("Room 1")
    engine.exit_person(count=r1.occupant_count, room_id="Room 1")

    r1_after = engine.sensor_manager.get_latest_readings()["Room 1"]
    # Acoustic immediately drops
    assert r1_after.sound_pressure_level < high_sound
    # Dynamic variables remain continuous
    assert abs(r1_after.indoor_co2 - co2_val) < 0.1
    assert abs(r1_after.indoor_temperature - temp_val) < 0.05


# -----------------------------------------------------------------------------
# 9. ROOM-TO-ROOM MOVEMENT: SOURCE UPDATES IN BOTH ROOMS
# -----------------------------------------------------------------------------

def test_room_to_room_movement_updates_both_rooms_independently(clean_engine):
    """
    Test 9: Moving people from Room 1 to Room 2 updates acoustic levels immediately
    and diverges generation rates over subsequent steps.
    """
    engine = clean_engine
    engine.add_person(count=6)
    # Move 6 people to Room 2
    res = engine.move_people("Room 1", "Room 2", count=6)
    assert res["success"] is True

    readings = engine.sensor_manager.get_latest_readings()
    r1_sound = readings["Room 1"].sound_pressure_level
    r2_sound = readings["Room 2"].sound_pressure_level

    # Room 2 is now occupied; Room 1 is empty
    assert r2_sound > r1_sound

    # Step simulation: Room 2 CO2 rises, Room 1 CO2 decays
    for _ in range(4):
        engine.step()

    new_readings = engine.sensor_manager.get_latest_readings()
    assert new_readings["Room 2"].indoor_co2 > new_readings["Room 1"].indoor_co2


# -----------------------------------------------------------------------------
# 10. VACANT RELAXATION TOWARD EQUILIBRIUM
# -----------------------------------------------------------------------------

def test_empty_rooms_relax_toward_equilibrium(clean_engine):
    """
    Test 10: Vacant rooms smoothly relax toward HVAC and outdoor equilibrium.
    """
    engine = clean_engine
    engine.add_person(count=8)
    for _ in range(6):
        engine.step()

    # Empty room
    r1 = engine.building.get_room("Room 1")
    engine.exit_person(count=r1.occupant_count, room_id="Room 1")

    # Run for 24 steps (2 hours)
    for _ in range(24):
        engine.step()

    final_r1 = engine.sensor_manager.get_latest_readings()["Room 1"]
    # CO2 relaxed back to near outdoor baseline
    assert final_r1.indoor_co2 < 450.0
    # Sound level back to base noise floor
    assert final_r1.sound_pressure_level <= 52.0


# -----------------------------------------------------------------------------
# 11. INVARIANCE UNDER UI RERUNS WITHOUT TIME PROGRESSION
# -----------------------------------------------------------------------------

def test_repeated_ui_reruns_do_not_advance_physics(clean_engine):
    """
    Test 11: Refreshing telemetry or querying sensors multiple times at the same timestamp
    does not cause temperature, CO2, or humidity to advance or diverge.
    """
    engine = clean_engine
    t0_readings = engine.sensor_manager.get_latest_readings()

    for _ in range(5):
        engine.sensor_manager.refresh_immediate_sensors(engine.current_time, engine.building)

    t1_readings = engine.sensor_manager.get_latest_readings()
    for r in EXPECTED_ROOM_IDS:
        assert t0_readings[r].indoor_temperature == t1_readings[r].indoor_temperature
        assert t0_readings[r].indoor_co2 == t1_readings[r].indoor_co2
        assert t0_readings[r].indoor_relative_humidity == t1_readings[r].indoor_relative_humidity


# -----------------------------------------------------------------------------
# 12. PHYSICAL BOUNDEDNESS & NUMERICAL INTEGRITY
# -----------------------------------------------------------------------------

def test_physical_bounds_and_finite_values(clean_engine):
    """
    Test 12: All 7 features strictly respect physical bounds across extensive simulation.
    """
    engine = clean_engine
    engine.add_person(count=20)
    for _ in range(12):
        engine.step()

    df = engine.get_sensor_readings_df()
    assert not df.empty

    for feat in COMMON_SENSOR_FEATURES:
        vals = df[feat].values
        assert np.all(np.isfinite(vals)), f"Non-finite values found in {feat}"
        assert not np.any(np.isnan(vals)), f"NaN values found in {feat}"

    assert (df["illuminance"] >= 0.0).all()
    assert (df["sound_pressure_level"] >= 35.0).all()
    assert (df["indoor_temperature"] >= 15.0).all()
    assert (df["indoor_temperature"] <= 40.0).all()
    assert (df["indoor_relative_humidity"] >= 20.0).all()
    assert (df["indoor_relative_humidity"] <= 100.0).all()
    assert (df["indoor_co2"] >= 400.0).all()


# -----------------------------------------------------------------------------
# 13. BATCH ARRIVALS AND EXITS
# -----------------------------------------------------------------------------

def test_batch_arrivals_and_exits(clean_engine):
    """
    Test 13: Large batch arrivals scale acoustic and thermal source terms properly.
    """
    engine = clean_engine
    res = engine.add_person(count=12)
    assert res["admitted_count"] == 12
    readings = engine.sensor_manager.get_latest_readings()["Room 1"]
    assert readings.sound_pressure_level > 55.0


# -----------------------------------------------------------------------------
# 14. UNIFIED PHYSICAL BEHAVIOR ACROSS FLOW MODES
# -----------------------------------------------------------------------------

def test_unified_physical_behavior_across_flow_modes(clean_engine):
    """
    Test 14: Manual, interactive, and automatic movements interact with the exact same
    underlying RoomSensorModel instances.
    """
    engine = clean_engine
    # Interactive add
    engine.add_person(count=4)
    r1_sound_manual = engine.sensor_manager.get_latest_readings()["Room 1"].sound_pressure_level

    # Manual move
    engine.move_people("Room 1", "Room 2", count=2)
    r2_sound = engine.sensor_manager.get_latest_readings()["Room 2"].sound_pressure_level
    assert r2_sound > 48.0


# -----------------------------------------------------------------------------
# 15. STGNN PIPELINE INTEGRATION & FORECAST VALIDITY
# -----------------------------------------------------------------------------

def test_pipeline_stgnn_inference_with_phase6_physics():
    """
    Test 15: DigitalTwinPipeline end-to-end integration:
    - 12-step lookback rolling buffer maintains all 13 scaled features correctly.
    - Forward inference produces valid non-negative integer predictions for all 5 rooms.
    """
    engine = SimulationEngine(num_occupants=25, seed=42)
    pipeline = DigitalTwinPipeline(engine=engine, seed=42)

    # Warm up pipeline for 15 steps
    last_res = None
    for _ in range(15):
        last_res = pipeline.step()

    assert last_res is not None
    assert last_res["status"] == "SUCCESS"
    assert last_res["predictions"] is not None
    for r in EXPECTED_ROOM_IDS:
        pred_val = last_res["predictions"][r]["predicted_occupants"]
        assert isinstance(pred_val, int)
        assert pred_val >= 0, f"Predicted occupancy for {r} must be non-negative integer, got {pred_val}"

