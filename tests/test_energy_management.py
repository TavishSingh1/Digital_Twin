"""
Automated Test Suite for Phase 7: Smart HVAC Management and Energy Consumption
=============================================================================
Comprehensive test suite verifying:
1.  Independent room energy states with required telemetry fields.
2.  Occupancy changes affect control decisions promptly (ECO -> NORMAL).
3.  Low/zero occupancy reduces setpoint demand and cooling requirements.
4.  High occupant count increases internal sensible heat gain and cooling demand.
5.  Vacant rooms transition to ECO mode after 10-minute vacancy delay.
6.  Vacancy timers and energy do NOT advance during repeated UI reruns (advance_time=False).
7.  Daylight-harvesting lighting modulates power according to natural daylight lux.
8.  Lighting responds to vacancy and turns off after 10-minute timeout.
9.  FORCE_ON and FORCE_OFF lighting overrides function as expected.
10. HVAC manual overrides (ECO, FORCE_OFF, AUTO) function as expected.
11. HVAC electrical power remains strictly within physical limits [P_standby, P_max].
12. Energy integration formula E = P * dt / 3,600,000 calculates kWh accurately.
13. Repeated processing of the same timestamp is strictly idempotent (zero double-counting).
14. HVAC, lighting, and auxiliary cumulative energies are tracked separately.
15. Room-wise powers and energies sum exactly to building-level totals.
16. Manual room transfers update both source and destination rooms immediately.
17. Simulation reset clears energy history and cumulative kWh meters.
18. Invalid energy configuration parameters raise clean, informative errors.
19. Energy dataframe export generates valid CSV schema.
20. Baseline comparator computes consistent savings metrics.
"""

import math
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from simulation.config import EXPECTED_ROOM_IDS
from simulation.engine import SimulationEngine
from simulation.energy import (
    load_energy_config,
    RoomEnergyState,
    HVACController,
    LightingController,
    BaselinePolicySimulator,
    EnergyManager,
)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def clean_engine():
    """Returns an engine initialized with 0 initial occupants for controlled testing."""
    engine = SimulationEngine(num_occupants=0, seed=100)
    for r in EXPECTED_ROOM_IDS:
        engine.building.get_room(r).occupants.clear()
    engine.occupants.clear()
    engine.entrance_queue.clear()
    engine.total_arrivals = 0
    engine.total_exits = 0
    engine.sensor_manager.refresh_immediate_sensors(engine.current_time, engine.building)
    engine.energy_manager.refresh_immediate(engine.current_time, engine.building, engine.sensor_manager)
    return engine


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test_1_room_independent_energy_state(clean_engine):
    """Test 1: Every room receives an independent energy state with all required fields."""
    states = clean_engine.energy_manager.get_latest_states()
    assert len(states) == 5, f"Expected 5 room states, got {len(states)}"

    required_fields = [
        "room_id", "timestamp", "occupancy", "hvac_mode", "is_eco_mode",
        "hvac_setpoint_c", "cooling_demand_w", "hvac_power_w", "lighting_power_w",
        "auxiliary_power_w", "total_power_w", "cumulative_hvac_kwh",
        "cumulative_lighting_kwh", "cumulative_auxiliary_kwh", "cumulative_total_kwh",
    ]

    for r_id in EXPECTED_ROOM_IDS:
        assert r_id in states, f"Missing room {r_id} in energy states"
        s = states[r_id]
        assert isinstance(s, RoomEnergyState)
        for field in required_fields:
            assert hasattr(s, field), f"RoomEnergyState missing field '{field}'"
            val = getattr(s, field)
            assert val is not None, f"Field '{field}' is None for {r_id}"

    # Verify rooms have distinct ratings (e.g. Room 1 is larger than Room 3)
    cfg1 = clean_engine.energy_manager.controllers["Room 1"].hvac.config
    cfg3 = clean_engine.energy_manager.controllers["Room 3"].hvac.config
    assert cfg1["cooling_capacity_w"] > cfg3["cooling_capacity_w"], "Room 1 cooling rating should exceed Room 3"



def test_2_occupancy_affects_control_promptly(clean_engine):
    """Test 2: Occupancy changes affect control decisions promptly (ECO -> NORMAL)."""
    # Room 1 is initially vacant -> ECO mode
    s_before = clean_engine.energy_manager.get_latest_states()["Room_1"]
    assert s_before.occupancy == 0
    assert s_before.is_eco_mode is True

    # Add 1 person to Room 1
    clean_engine.add_person("Room_1")

    s_after = clean_engine.energy_manager.get_latest_states()["Room_1"]
    assert s_after.occupancy == 1
    assert s_after.is_eco_mode is False
    assert s_after.hvac_mode in ["NORMAL", "HIGH_DEMAND"]


def test_3_cooling_demand_reduction_when_vacant(clean_engine):
    """Test 3: Zero/low occupancy reduces setpoint demand via eco setback."""
    ctrl = clean_engine.energy_manager.controllers["Room_1"]
    
    # Under identical thermal conditions (e.g. indoor 26°C, outdoor 32°C)
    now = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
    
    # Case A: Occupied (setpoint 24.5°C) -> demands active cooling
    power_occ, q_cool_occ, mode_occ, setpoint_occ = ctrl.hvac.compute(
        current_temp=26.0, outdoor_temp=32.0, occupancy=2,
        current_time=now, dt_seconds=300.0, advance_time=False
    )
    assert setpoint_occ == 24.5
    assert q_cool_occ > 0.0

    # Case B: Vacant (setpoint 28.0°C) -> current temp 26°C is BELOW 28°C eco setpoint
    # Set vacancy timer past delay
    ctrl.hvac.vacancy_timer_seconds = 600.0
    power_vac, q_cool_vac, mode_vac, setpoint_vac = ctrl.hvac.compute(
        current_temp=26.0, outdoor_temp=32.0, occupancy=0,
        current_time=now, dt_seconds=300.0, advance_time=False
    )
    assert setpoint_vac == 28.0
    assert q_cool_vac == 0.0
    assert power_vac < power_occ


def test_4_high_occupancy_increases_heat_gain_and_cooling(clean_engine):
    """Test 4: High occupant count increases internal sensible heat gain and cooling demand."""
    ctrl = clean_engine.energy_manager.controllers["Room_4"]
    now = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)

    # 1 person (100 W metabolic heat)
    p_1, q_1, _, _ = ctrl.hvac.compute(
        current_temp=25.5, outdoor_temp=30.0, occupancy=1,
        current_time=now, dt_seconds=300.0, advance_time=False
    )

    # 15 persons (1500 W metabolic heat)
    p_15, q_15, _, _ = ctrl.hvac.compute(
        current_temp=25.5, outdoor_temp=30.0, occupancy=15,
        current_time=now, dt_seconds=300.0, advance_time=False
    )

    assert q_15 > q_1, f"Expected higher cooling load with 15 occupants (got {q_15} W vs {q_1} W)"
    assert p_15 > p_1, f"Expected higher electrical power with 15 occupants (got {p_15} W vs {p_1} W)"


def test_5_vacant_room_switches_to_eco_after_delay(clean_engine):
    """Test 5: Vacant room switches to ECO mode after 10-minute simulated vacancy delay."""
    # Add person to Room 1 and move to Room 2
    clean_engine.add_person(1)
    clean_engine.move_people("Room 1", "Room 2", 1)
    s = clean_engine.energy_manager.get_latest_states()["Room 2"]
    assert s.is_eco_mode is False

    # Remove person
    occ_id = next(iter(clean_engine.building.get_room("Room 2").occupants))
    clean_engine.exit_person(occ_id)
    assert clean_engine.building.get_room("Room 2").occupant_count == 0


    # Step 1 (5 minutes elapsed): Vacancy timer is 300s (5 min) < 600s (10 min)
    clean_engine.step()
    s1 = clean_engine.energy_manager.get_latest_states()["Room 2"]
    assert s1.is_eco_mode is False, "Room should remain in normal mode before 10-min vacancy delay"

    # Step 2 (10 minutes elapsed): Vacancy timer is 600s >= 600s
    clean_engine.step()
    s2 = clean_engine.energy_manager.get_latest_states()["Room 2"]
    assert s2.is_eco_mode is True, "Room should switch to ECO mode after 10-min vacancy delay"
    assert s2.hvac_mode == "ECO"


def test_6_idempotency_under_repeated_queries_advance_time_false(clean_engine):
    """Test 6: Vacancy timers and energy do NOT advance during repeated calls with advance_time=False (UI reruns)."""
    clean_engine.step()
    s_initial = clean_engine.energy_manager.get_latest_states()["Room_1"]
    kwh_initial = s_initial.cumulative_total_kwh
    timer_initial = clean_engine.energy_manager.controllers["Room_1"].hvac.vacancy_timer_seconds

    # Simulate 10 Streamlit UI reruns at the exact same timestamp
    for _ in range(10):
        clean_engine.energy_manager.refresh_immediate(
            clean_engine.current_time,
            clean_engine.building,
            clean_engine.sensor_manager,
        )

    s_final = clean_engine.energy_manager.get_latest_states()["Room_1"]
    timer_final = clean_engine.energy_manager.controllers["Room_1"].hvac.vacancy_timer_seconds

    assert s_final.cumulative_total_kwh == kwh_initial, "Cumulative kWh changed during UI reruns!"
    assert timer_final == timer_initial, "Vacancy timer advanced during UI reruns!"


def test_7_daylight_harvesting_modulates_lighting_power(clean_engine):
    """Test 7: Lighting responds to daylight availability (daylight harvesting)."""
    ctrl = clean_engine.energy_manager.controllers["Room_1"]

    # Occupied room
    # High daylight condition (midday summer, 800 lux daylight)
    p_bright, _, _ = ctrl.lighting.compute(
        occupancy=1, daylight_lux=800.0, dt_seconds=300.0, advance_time=False
    )
    # Low daylight condition (night / overcast, 50 lux daylight)
    p_dark, _, _ = ctrl.lighting.compute(
        occupancy=1, daylight_lux=50.0, dt_seconds=300.0, advance_time=False
    )

    # In bright daylight, lighting should dim down to minimum fraction
    assert p_bright < p_dark, f"Daylight harvesting failed: bright {p_bright} W not < dark {p_dark} W"
    expected_min_power = ctrl.lighting.installed_power_w * (0.10 + 0.90 * ctrl.lighting.config["min_dim_fraction"])
    assert math.isclose(p_bright, expected_min_power, rel_tol=1e-2)



def test_8_lighting_vacancy_timeout(clean_engine):
    """Test 8: Lighting responds to vacancy and turns off after 10-minute timeout."""
    ctrl = clean_engine.energy_manager.controllers["Room_3"]
    # Step 1: Occupied
    p1, on1, _ = ctrl.lighting.compute(occupancy=1, daylight_lux=0.0, dt_seconds=300.0, advance_time=True)
    assert on1 is True
    assert p1 > 0.0

    # Step 2: Vacant for 5 min (300s < 600s) -> should remain ON
    p2, on2, _ = ctrl.lighting.compute(occupancy=0, daylight_lux=0.0, dt_seconds=300.0, advance_time=True)
    assert on2 is True, "Lighting should stay on during 10-min vacancy delay"
    assert p2 > 0.0

    # Step 3: Vacant for another 5 min (600s >= 600s) -> should turn OFF
    p3, on3, _ = ctrl.lighting.compute(occupancy=0, daylight_lux=0.0, dt_seconds=300.0, advance_time=True)
    assert on3 is False, "Lighting should turn off after 10-min vacancy timeout"
    assert p3 == 0.0


def test_9_lighting_manual_overrides(clean_engine):
    """Test 9: FORCE_ON and FORCE_OFF lighting overrides function as expected."""
    ctrl = clean_engine.energy_manager.controllers["Room_1"]

    # Vacant at night (auto would turn OFF)
    ctrl.lighting.vacancy_timer_seconds = 1000.0
    p_auto, on_auto, _ = ctrl.lighting.compute(occupancy=0, daylight_lux=0.0, dt_seconds=300.0, advance_time=False)
    assert on_auto is False
    assert p_auto == 0.0

    # Override FORCE_ON
    ctrl.lighting.override_mode = "FORCE_ON"
    p_force_on, on_force_on, _ = ctrl.lighting.compute(occupancy=0, daylight_lux=0.0, dt_seconds=300.0, advance_time=False)
    assert on_force_on is True
    assert p_force_on > 0.0

    # Override FORCE_OFF (even with 5 occupants)
    ctrl.lighting.override_mode = "FORCE_OFF"
    p_force_off, on_force_off, _ = ctrl.lighting.compute(occupancy=5, daylight_lux=0.0, dt_seconds=300.0, advance_time=False)
    assert on_force_off is False
    assert p_force_off == 0.0


def test_10_hvac_manual_overrides(clean_engine):
    """Test 10: HVAC manual overrides (ECO, FORCE_OFF, AUTO) function as expected."""
    ctrl = clean_engine.energy_manager.controllers["Room_1"]
    now = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)

    # Force OFF
    ctrl.hvac.override_mode = "FORCE_OFF"
    p_off, _, mode_off, _ = ctrl.hvac.compute(
        current_temp=27.0, outdoor_temp=32.0, occupancy=5,
        current_time=now, dt_seconds=300.0, advance_time=False
    )
    assert mode_off == "OFF"
    assert p_off in (0.0, ctrl.hvac.config["standby_power_w"])

    # Force ECO
    ctrl.hvac.override_mode = "ECO"
    _, _, mode_eco, sp_eco = ctrl.hvac.compute(
        current_temp=25.0, outdoor_temp=32.0, occupancy=5,
        current_time=now, dt_seconds=300.0, advance_time=False
    )
    assert mode_eco == "ECO"
    assert sp_eco == ctrl.hvac.config["setpoint_eco_c"]

    # Return to AUTO
    ctrl.hvac.override_mode = "AUTO"
    _, _, mode_auto, sp_auto = ctrl.hvac.compute(
        current_temp=25.0, outdoor_temp=32.0, occupancy=5,
        current_time=now, dt_seconds=300.0, advance_time=False
    )
    assert mode_auto == "NORMAL"
    assert sp_auto == ctrl.hvac.config["setpoint_normal_c"]


def test_11_hvac_power_within_physical_limits(clean_engine):
    """Test 11: HVAC electrical power stays within configured minimum/maximum limits."""
    ctrl = clean_engine.energy_manager.controllers["Room_5"]
    now = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
    cop = ctrl.hvac.cop
    p_standby = ctrl.hvac.config["standby_power_w"]
    p_max = ctrl.hvac.config["cooling_capacity_w"] / cop

    # Extreme high load condition (very hot indoor 38°C, outdoor 45°C, 30 occupants)
    p_extreme, _, _, _ = ctrl.hvac.compute(
        current_temp=38.0, outdoor_temp=45.0, occupancy=30,
        current_time=now, dt_seconds=300.0, advance_time=False
    )
    assert p_extreme <= p_max * 1.001, f"HVAC power {p_extreme} exceeded maximum rated capacity {p_max}"
    assert p_extreme >= p_standby, f"HVAC power {p_extreme} fell below standby power {p_standby}"

    # Idle within deadband
    p_idle, _, _, _ = ctrl.hvac.compute(
        current_temp=24.5, outdoor_temp=24.5, occupancy=0,
        current_time=now, dt_seconds=300.0, advance_time=False
    )
    assert p_idle >= p_standby
    assert p_idle <= p_max


def test_12_energy_integration_kwh_formula(clean_engine):
    """Test 12: Energy integration formula E = P * dt / 3,600,000 calculates kWh accurately."""
    manager = clean_engine.energy_manager
    # Ensure fresh state
    manager.reset()
    
    # Step 1
    t1 = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
    manager.step(t1, clean_engine.building, clean_engine.sensor_manager, dt_seconds=300.0)
    s1 = manager.get_latest_states()["Room_1"]
    
    # Mathematical integration check: P_watts * 300s / 3.6e6 J/kWh = P_watts / 12,000 kWh
    expected_kwh = s1.total_power_w * 300.0 / 3_600_000.0
    assert math.isclose(s1.cumulative_total_kwh, expected_kwh, abs_tol=1e-4), (
        f"Cumulative kWh {s1.cumulative_total_kwh} != expected {expected_kwh}"
    )



def test_13_strictly_idempotent_timestamp_processing(clean_engine):
    """Test 13: Repeated processing of the same timestamp is strictly idempotent and does not double-count energy."""
    manager = clean_engine.energy_manager
    manager.reset()

    t1 = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
    manager.step(t1, clean_engine.building, clean_engine.sensor_manager, dt_seconds=300.0)
    kwh_after_step1 = manager.get_building_kpis()["cumulative_total_kwh"]

    # Call step again with the EXACT same timestamp
    manager.step(t1, clean_engine.building, clean_engine.sensor_manager, dt_seconds=300.0)
    kwh_after_duplicate = manager.get_building_kpis()["cumulative_total_kwh"]

    assert kwh_after_step1 == kwh_after_duplicate, "Energy was double-counted on duplicate timestamp step!"


def test_14_subsystem_cumulative_energy_tracking(clean_engine):
    """Test 14: HVAC, lighting, and auxiliary cumulative energies are tracked separately and sum to total."""
    # Run 3 simulation steps
    for _ in range(3):
        clean_engine.step()

    for r_id in EXPECTED_ROOM_IDS:
        s = clean_engine.energy_manager.get_latest_states()[r_id]
        assert s.cumulative_hvac_kwh >= 0.0
        assert s.cumulative_lighting_kwh >= 0.0
        assert s.cumulative_auxiliary_kwh >= 0.0
        summed = s.cumulative_hvac_kwh + s.cumulative_lighting_kwh + s.cumulative_auxiliary_kwh
        assert math.isclose(s.cumulative_total_kwh, summed, rel_tol=1e-5), (
            f"Subsystem energies do not sum to total for {r_id}: {s.cumulative_total_kwh} vs {summed}"
        )


def test_15_room_wise_powers_sum_to_building_totals(clean_engine):
    """Test 15: Room-wise powers and energies sum exactly to building-level totals."""
    clean_engine.step()
    kpis = clean_engine.energy_manager.get_building_kpis()

    canonical_states = [clean_engine.energy_manager.latest_states[r] for r in EXPECTED_ROOM_IDS]
    total_power_sum_w = sum(s.total_power_w for s in canonical_states)
    total_kwh_sum = sum(s.cumulative_total_kwh for s in canonical_states)

    assert math.isclose(kpis["total_power_kw"], total_power_sum_w / 1000.0, rel_tol=1e-5), (
        f"Building power KPI {kpis['total_power_kw']} != sum of rooms {total_power_sum_w / 1000.0}"
    )
    assert math.isclose(kpis["cumulative_total_kwh"], total_kwh_sum, abs_tol=1e-3), (
        f"Building energy KPI {kpis['cumulative_total_kwh']} != sum of rooms {total_kwh_sum}"
    )


def test_16_manual_room_transfer_updates_immediately(clean_engine):
    """Test 16: Manual room transfers update both source and destination rooms immediately."""
    # Place 3 people in Room 1
    clean_engine.add_person(3)

    s1_before = clean_engine.energy_manager.get_latest_states()["Room 1"]
    s2_before = clean_engine.energy_manager.get_latest_states()["Room 2"]
    assert s1_before.occupancy == 3
    assert s2_before.occupancy == 0
    assert s2_before.is_eco_mode is True

    # Move 3 people from Room 1 to Room 2
    res = clean_engine.move_people("Room 1", "Room 2", 3)
    assert res["moved_count"] == 3

    s1_after = clean_engine.energy_manager.get_latest_states()["Room 1"]
    s2_after = clean_engine.energy_manager.get_latest_states()["Room 2"]

    assert s1_after.occupancy == 0
    assert s2_after.occupancy == 3
    assert s2_after.is_eco_mode is False, "Destination room should wake from ECO immediately on arrival"



def test_17_simulation_reset_clears_energy_history(clean_engine):
    """Test 17: Resetting simulation resets energy history and clears cumulative kWh meters."""
    for _ in range(4):
        clean_engine.step()

    kpis_before = clean_engine.energy_manager.get_building_kpis()
    assert kpis_before["cumulative_total_kwh"] > 0.0

    clean_engine.reset()

    kpis_after = clean_engine.energy_manager.get_building_kpis()
    assert kpis_after["cumulative_total_kwh"] == 0.0
    for r_id in EXPECTED_ROOM_IDS:
        s = clean_engine.energy_manager.get_latest_states()[r_id]
        assert s.cumulative_total_kwh == 0.0


def test_18_invalid_energy_config_raises_error():
    """Test 18: Invalid energy configuration parameters raise clean, informative errors."""
    bad_config = {
        "rooms": {
            "Room_1": {
                "hvac": {
                    "cop": -1.0, # Negative COP is physically invalid!
                    "cooling_capacity_w": 8000.0,
                    "standby_power_w": 25.0,
                    "setpoint_normal_c": 24.5,
                    "setpoint_eco_c": 28.0,
                    "deadband_c": 0.5,
                    "vacancy_delay_minutes": 10.0,
                },
                "lighting": {
                    "installed_power_w": 240.0,
                    "target_illuminance_lux": 300.0,
                    "min_dim_fraction": 0.15,
                    "vacancy_delay_minutes": 10.0,
                },
                "auxiliary": {
                    "base_power_w": 100.0,
                },
            }
        }
    }
    with pytest.raises(ValueError, match="COP must be positive"):
        EnergyManager(config=bad_config)


def test_19_energy_dataframe_and_csv_export(clean_engine, tmp_path):
    """Test 19: Energy dataframe export generates valid CSV schema with all required columns."""
    clean_engine.step()
    clean_engine.step()

    df = clean_engine.get_energy_readings_df()
    assert not df.empty
    assert len(df) == 15 # 5 rooms * 3 steps (step 0, 1, 2)

    csv_path = tmp_path / "test_energy_export.csv"
    clean_engine.export_energy_readings(str(csv_path))

    assert csv_path.exists()
    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == 15
    expected_cols = [
        "timestamp", "room_id", "occupant_count", "hvac_mode", "is_eco_mode",
        "hvac_power_w", "lighting_power_w", "auxiliary_power_w", "total_power_w",
        "cumulative_hvac_kwh", "cumulative_lighting_kwh", "cumulative_auxiliary_kwh",
        "cumulative_total_kwh", "indoor_temperature", "cooling_demand_w",
    ]
    for col in expected_cols:
        assert col in loaded_df.columns, f"Exported CSV missing column '{col}'"



def test_20_baseline_comparator_savings_consistency(clean_engine):
    """Test 20: Baseline policy simulator computes consistent, non-NaN savings metrics."""
    # Advance 5 steps
    for _ in range(5):
        clean_engine.step()

    metrics = clean_engine.energy_manager.get_comparison_metrics()
    assert "baseline_total_kwh" in metrics
    assert "smart_total_kwh" in metrics
    assert "savings_kwh" in metrics
    assert "savings_percent" in metrics
    assert "peak_smart_demand_kw" in metrics
    assert "avg_smart_demand_kw" in metrics

    # All metrics must be finite and non-negative
    for k, v in metrics.items():
        assert not np.isnan(v), f"Metric {k} is NaN"
        assert v >= 0.0, f"Metric {k} is negative: {v}"

    # Savings = baseline - smart
    expected_savings = metrics["baseline_total_kwh"] - metrics["smart_total_kwh"]
    assert math.isclose(metrics["savings_kwh"], max(0.0, expected_savings), abs_tol=1e-2)

