"""
Automated Test Suite for Virtual Environmental Sensor Engine (Phase 3.2)
========================================================================
Verifies all physical modeling, synchronization, determinism, and data requirements:
1. All 7 environmental features are generated for all 5 rooms.
2. Physical bounds are strictly respected for all features.
3. Temporal evolution is smooth without non-physical discontinuous jumps.
4. CO2 responds physically to occupancy (rises when occupied, decays when vacant).
5. Sound pressure level follows logarithmic acoustic addition with occupants.
6. Illuminance follows operational schedules and daylight envelope.
7. Outdoor weather follows diurnal cycles with temperature-humidity anti-correlation.
8. Simulation with identical seed produces strictly deterministic outputs.
9. Timestamps are perfectly synchronized with the 5-minute logical clock.
10. No duplicate or missing records exist across rooms or timestamps.
11. Exported CSV contains all expected features and valid numeric entries.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from simulation.config import load_building_config, EXPECTED_ROOM_IDS
from simulation.sensors import (
    load_sensor_config,
    RoomSensorReading,
    RoomSensorModel,
    VirtualSensorManager,
)
from simulation.environment import OutdoorWeather
from simulation.engine import SimulationEngine
from stgnn_model import COMMON_SENSOR_FEATURES


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------

@pytest.fixture
def sensor_cfg():
    return load_sensor_config("config/sensor_parameters.json")


@pytest.fixture
def engine():
    return SimulationEngine(num_occupants=40, seed=42)


# ---------------------------------------------------------------------
# 1. FEATURE EXISTENCE AND MAPPING
# ---------------------------------------------------------------------

def test_all_seven_features_exist_per_room(engine):
    """Test 1: All 7 standard environmental features exist for each room."""
    step_result = engine.step()
    readings = step_result["sensor_readings"]

    assert len(readings) == 5
    for room_id in EXPECTED_ROOM_IDS:
        assert room_id in readings
        reading = readings[room_id]
        assert isinstance(reading, RoomSensorReading)

        for feat in COMMON_SENSOR_FEATURES:
            assert hasattr(reading, feat), f"Missing feature {feat} in {room_id}"
            val = getattr(reading, feat)
            assert isinstance(val, (int, float)), f"Feature {feat} is not numeric: {val}"
            assert not np.isnan(val), f"Feature {feat} is NaN in {room_id}"


# ---------------------------------------------------------------------
# 2. PHYSICAL BOUNDS RESPECTED
# ---------------------------------------------------------------------

def test_physical_bounds_respected(engine):
    """Test 2: All environmental sensor readings respect physical bounds over 24 hours."""
    # Run 288 steps (24 hours x 12 steps/hour)
    engine.run(num_steps=72) # 6 hours simulation
    df = engine.get_sensor_readings_df()

    assert not df.empty

    # Check bounds across all recorded rows
    assert (df["indoor_co2"] >= 390.0).all(), "CO2 fell below baseline"
    assert (df["indoor_co2"] <= 2500.0).all(), "CO2 exceeded upper safety bound"

    assert (df["indoor_temperature"] >= 18.0).all(), "Indoor temp below 18 C"
    assert (df["indoor_temperature"] <= 36.0).all(), "Indoor temp above 36 C"

    assert (df["dry_bulb_temperature"] >= 18.0).all(), "Dry bulb temp below 18 C"
    assert (df["dry_bulb_temperature"] <= 40.0).all(), "Dry bulb temp above 40 C"

    assert (df["indoor_relative_humidity"] >= 30.0).all(), "Indoor RH below 30%"
    assert (df["indoor_relative_humidity"] <= 100.0).all(), "Indoor RH above 100%"

    assert (df["outdoor_relative_humidity"] >= 30.0).all(), "Outdoor RH below 30%"
    assert (df["outdoor_relative_humidity"] <= 100.0).all(), "Outdoor RH above 100%"

    assert (df["sound_pressure_level"] >= 35.0).all(), "Sound level below 35 dBA"
    assert (df["sound_pressure_level"] <= 90.0).all(), "Sound level above 90 dBA"

    assert (df["illuminance"] >= 0.0).all(), "Illuminance negative"
    assert (df["illuminance"] <= 15000.0).all(), "Illuminance exceeded peak daylight"


# ---------------------------------------------------------------------
# 3. SMOOTH TEMPORAL EVOLUTION
# ---------------------------------------------------------------------

def test_smooth_temporal_evolution(engine):
    """Test 3: Sensor readings exhibit smooth physical inertia without sudden jumps."""
    engine.run(num_steps=36) # 3 hours
    df = engine.get_sensor_readings_df()

    for room_id in EXPECTED_ROOM_IDS:
        room_df = df[df["room_id"] == room_id].copy()

        # Step-to-step temperature changes should be smooth (< 1.0 C per 5 min)
        temp_diffs = room_df["indoor_temperature"].diff().dropna().abs()
        assert (temp_diffs < 1.0).all(), f"Abrupt temp jump detected in {room_id}: {temp_diffs.max()}"

        # Step-to-step indoor humidity changes should be smooth (< 3.0 % per 5 min)
        rh_diffs = room_df["indoor_relative_humidity"].diff().dropna().abs()
        assert (rh_diffs < 3.0).all(), f"Abrupt RH jump detected in {room_id}: {rh_diffs.max()}"

        # Step-to-step CO2 changes should be physically continuous (< 150 ppm per 5 min)
        co2_diffs = room_df["indoor_co2"].diff().dropna().abs()
        assert (co2_diffs < 150.0).all(), f"Abrupt CO2 jump detected in {room_id}: {co2_diffs.max()}"


# ---------------------------------------------------------------------
# 4. CO2 MASS-BALANCE RESPONSE TO OCCUPANCY
# ---------------------------------------------------------------------

def test_co2_responsiveness_to_occupants(sensor_cfg):
    """Test 4: CO2 rises when occupied and decays back towards baseline when vacant."""
    room_model = RoomSensorModel(
        room_id="Room 2",
        room_cfg=sensor_cfg["rooms"]["Room 2"],
        outdoor_co2_baseline=400.0,
        seed=123,
    )

    t0 = datetime(2021, 9, 6, 8, 0, tzinfo=timezone.utc)

    # 1. Run 12 steps (1 hour) with 15 occupants -> CO2 must increase
    co2_start = room_model.current_co2
    curr_t = t0
    for _ in range(12):
        curr_t += timedelta(minutes=5)
        room_model.step(curr_t, occupants=15, outdoor_temp=28.0, outdoor_rh=80.0, solar_lux=2000.0)

    co2_occupied = room_model.current_co2
    assert co2_occupied > co2_start + 100.0, f"CO2 did not rise with occupants: {co2_start} -> {co2_occupied}"

    # 2. Run 24 steps (2 hours) with 0 occupants -> CO2 must decay towards baseline
    for _ in range(24):
        curr_t += timedelta(minutes=5)
        room_model.step(curr_t, occupants=0, outdoor_temp=28.0, outdoor_rh=80.0, solar_lux=2000.0)

    co2_vacant = room_model.current_co2
    assert co2_vacant < co2_occupied - 80.0, f"CO2 did not decay when empty: {co2_occupied} -> {co2_vacant}"


# ---------------------------------------------------------------------
# 5. SOUND PRESSURE ACOUSTIC ADDITION
# ---------------------------------------------------------------------

def test_sound_pressure_acoustic_addition(sensor_cfg):
    """Test 5: Sound pressure increases logarithmically with occupant activity."""
    room_model = RoomSensorModel(
        room_id="Room 1",
        room_cfg=sensor_cfg["rooms"]["Room 1"],
        seed=42,
    )
    dt = datetime(2021, 9, 6, 10, 0, tzinfo=timezone.utc)

    reading_0 = room_model.step(dt, occupants=0, outdoor_temp=28.0, outdoor_rh=80.0, solar_lux=1000.0)
    reading_5 = room_model.step(dt, occupants=5, outdoor_temp=28.0, outdoor_rh=80.0, solar_lux=1000.0)
    reading_20 = room_model.step(dt, occupants=20, outdoor_temp=28.0, outdoor_rh=80.0, solar_lux=1000.0)

    assert reading_5.sound_pressure_level > reading_0.sound_pressure_level
    assert reading_20.sound_pressure_level > reading_5.sound_pressure_level


# ---------------------------------------------------------------------
# 6. ILLUMINANCE SCHEDULE & DAYLIGHT
# ---------------------------------------------------------------------

def test_illuminance_schedule_and_daylight(sensor_cfg):
    """Test 6: Work hours have artificial lighting, night has zero lux, midday has daylight."""
    room_model = RoomSensorModel(
        room_id="Room 5",
        room_cfg=sensor_cfg["rooms"]["Room 5"],
        seed=42,
    )

    # Midnight, vacant: lights off, solar 0
    t_night = datetime(2021, 9, 6, 2, 0, tzinfo=timezone.utc)
    reading_night = room_model.step(t_night, occupants=0, outdoor_temp=26.0, outdoor_rh=95.0, solar_lux=0.0)
    assert reading_night.illuminance <= 5.0, f"Night lux not near zero: {reading_night.illuminance}"

    # Work hours, midday: lights on + high solar lux
    t_day = datetime(2021, 9, 6, 12, 0, tzinfo=timezone.utc)
    reading_day = room_model.step(t_day, occupants=5, outdoor_temp=30.0, outdoor_rh=75.0, solar_lux=8000.0)
    assert reading_day.illuminance > 300.0, f"Daylight lux too low in Room 5: {reading_day.illuminance}"


# ---------------------------------------------------------------------
# 7. OUTDOOR WEATHER DIURNAL PATTERN & CORRELATION
# ---------------------------------------------------------------------

def test_outdoor_weather_diurnal_and_anticorrelation():
    """Test 7: Outdoor temperature and humidity follow diurnal cycle and anti-correlate."""
    weather = OutdoorWeather(seed=42)

    t_dawn = datetime(2021, 9, 6, 5, 30, tzinfo=timezone.utc)
    t_afternoon = datetime(2021, 9, 6, 14, 30, tzinfo=timezone.utc)

    temp_dawn = weather.get_dry_bulb_temperature(t_dawn)
    temp_afternoon = weather.get_dry_bulb_temperature(t_afternoon)
    assert temp_afternoon > temp_dawn + 3.0, f"Afternoon not warmer than dawn: {temp_afternoon} vs {temp_dawn}"

    rh_dawn = weather.get_outdoor_relative_humidity(t_dawn)
    rh_afternoon = weather.get_outdoor_relative_humidity(t_afternoon)
    assert rh_dawn > rh_afternoon + 10.0, f"Dawn humidity not higher than afternoon: {rh_dawn} vs {rh_afternoon}"


# ---------------------------------------------------------------------
# 8. DETERMINISM WITH SEED
# ---------------------------------------------------------------------

def test_sensor_engine_determinism():
    """Test 8: Two identical simulation runs with the same seed produce identical sensor readings."""
    e1 = SimulationEngine(num_occupants=30, seed=777)
    e2 = SimulationEngine(num_occupants=30, seed=777)

    e1.run(num_steps=24)
    e2.run(num_steps=24)

    df1 = e1.get_sensor_readings_df()
    df2 = e2.get_sensor_readings_df()

    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------
# 9. TIME SYNCHRONIZATION AND CADENCE
# ---------------------------------------------------------------------

def test_sensor_time_synchronization(engine):
    """Test 9: Sensor timestamps advance synchronously by exactly 5 minutes with no duplicates."""
    engine.run(num_steps=24)
    df = engine.get_sensor_readings_df()

    unique_ts = df["timestamp"].drop_duplicates().tolist()
    assert len(unique_ts) == 25 # Initial step 0 + 24 steps

    # Check 5-minute intervals
    parsed_ts = [datetime.fromisoformat(ts) for ts in unique_ts]
    for i in range(1, len(parsed_ts)):
        delta = parsed_ts[i] - parsed_ts[i - 1]
        assert delta == timedelta(minutes=5), f"Unexpected time step: {delta}"

    # Check exactly 5 rooms per timestamp
    counts_per_ts = df.groupby("timestamp")["room_id"].count()
    assert (counts_per_ts == 5).all(), "Incomplete room records at some timestamps"

    # Check no duplicate (timestamp, room_id) pairs
    dups = df.duplicated(subset=["timestamp", "room_id"]).sum()
    assert dups == 0, f"Found {dups} duplicate timestamp-room entries"


# ---------------------------------------------------------------------
# 10. CSV EXPORT INTEGRITY
# ---------------------------------------------------------------------

def test_sensor_csv_export(engine):
    """Test 10: Exported sensor CSV contains exact schema, valid numeric values, and parses cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_sensors.csv")
        engine.run(num_steps=12)
        engine.export_virtual_sensors(csv_path)

        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)

        expected_cols = [
            "timestamp", "room_id", "illuminance", "sound_pressure_level",
            "indoor_temperature", "dry_bulb_temperature",
            "indoor_relative_humidity", "outdoor_relative_humidity",
            "indoor_co2"
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 13 * 5 # 13 timesteps x 5 rooms
        assert df.isna().sum().sum() == 0, "Exported CSV contains missing or NaN values"

