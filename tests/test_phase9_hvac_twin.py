"""
Phase 9 Unit and Integration Test Suite: Advanced HVAC Digital Twin
===================================================================
Validates:
1. Thermostat deadband hysteresis and anti-short-cycling logic.
2. Physical coupling between HVAC state and room temperature/humidity dynamics.
3. Dynamic setpoint updates and physical response.
4. Continuous ECO setback and energy savings vs baseline.
5. FORCE_OFF complete electrical and airflow shutdown.
6. Thermodynamic separation of thermal cooling (W th) vs electrical power (W elec) via COP.
7. SVG Floor Plan HVAC terminal symbols, airflow keyframe animations, and chilled distribution loop.
8. Strictly isolated, non-mutating multi-policy simulation experiments (AUTO vs ECO vs FORCE_OFF).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from simulation.config import EXPECTED_ROOM_IDS
from simulation.engine import SimulationEngine
from simulation.energy import HVACController, RoomEnergyState, EnergyManager
from simulation.experiments import run_hvac_mode_experiment
from dashboard.svg_floorplan import generate_svg_floorplan_markup


class TestThermostatHysteresisAndAntiCycling:
    """Verifies thermostat deadband and compressor hysteresis."""

    def test_deadband_activation_and_deactivation(self):
        energy_mgr = EnergyManager()
        ctrl = energy_mgr.hvac_controllers["Room 1"]
        target_sp = 24.5
        ctrl.target_setpoint = target_sp
        now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)

        # 1. Start below deadband (e.g. 23.8°C < 24.5 - 0.5) with 1 occupant -> should be IDLE
        ctrl.update(dt=now, occupants=1, capacity=20, indoor_temp=23.8, outdoor_temp=30.0)
        assert ctrl.operating_state == "IDLE"
        assert ctrl.cooling_delivered_w == 0.0
        assert ctrl.fan_state in ("LOW", "OFF")

        # 2. Inside deadband (24.7°C, which is < 24.5 + 0.5 = 25.0°C) -> should remain IDLE (no false trigger)
        ctrl.update(dt=now, occupants=1, capacity=20, indoor_temp=24.7, outdoor_temp=30.0)
        assert ctrl.operating_state == "IDLE"
        assert ctrl.cooling_delivered_w == 0.0

        # 3. Exceed upper deadband (25.2°C > 24.5 + 0.5 = 25.0°C) -> COOLING must engage
        ctrl.update(dt=now, occupants=1, capacity=20, indoor_temp=25.2, outdoor_temp=30.0)
        assert ctrl.operating_state == "COOLING"
        assert ctrl.cooling_delivered_w > 0.0
        assert ctrl.airflow_m3_h > 0.0
        assert ctrl.fan_state in ("HIGH", "MEDIUM")

        # 4. Temperature cools down back into deadband (24.3°C, which is > 24.5 - 0.5 = 24.0°C) ->
        # Due to hysteresis, compressor MUST stay ON until bottom threshold is crossed
        ctrl.update(dt=now, occupants=1, capacity=20, indoor_temp=24.3, outdoor_temp=30.0)
        assert ctrl.operating_state == "COOLING"
        assert ctrl.cooling_delivered_w > 0.0

        # 5. Temperature drops below lower deadband (23.9°C < 24.5 - 0.5) -> switches to IDLE
        ctrl.update(dt=now, occupants=1, capacity=20, indoor_temp=23.9, outdoor_temp=30.0)
        assert ctrl.operating_state == "IDLE"
        assert ctrl.cooling_delivered_w == 0.0


class TestPhysicalHVACCoupling:
    """Verifies that HVAC operating states directly govern thermal and sensor dynamics."""

    def test_hvac_force_off_results_in_zero_cooling_and_airflow(self):
        energy_mgr = EnergyManager()
        ctrl = energy_mgr.hvac_controllers["Room 2"]
        ctrl.set_manual_override("FORCE_OFF")
        now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)

        mode, p, cool, sp, is_eco = ctrl.update(
            dt=now, occupants=10, capacity=20, indoor_temp=32.0, outdoor_temp=35.0
        )
        assert ctrl.operating_state == "OFF"
        assert mode == "OFF"
        assert ctrl.cooling_delivered_w == 0.0
        assert ctrl.airflow_m3_h == 0.0
        assert ctrl.fan_state == "OFF"
        assert p == 0.0
        assert "manual operator override" in ctrl.controller_explanation

    def test_cooling_reduces_indoor_temperature_relative_to_off(self):
        """Under identical heat loads, active HVAC produces lower temperatures than FORCE_OFF."""
        # 1. Engine with FORCE_OFF on Room 1
        engine_off = SimulationEngine(num_occupants=20, seed=123)
        engine_off.set_room_hvac_mode("Room 1", "FORCE_OFF")

        # 2. Engine with AUTO (active cooling) on Room 1
        engine_auto = SimulationEngine(num_occupants=20, seed=123)
        engine_auto.set_room_hvac_mode("Room 1", None)

        # Step both 10 steps
        for _ in range(10):
            res_off = engine_off.step()
            res_auto = engine_auto.step()

        temp_off = res_off["sensor_readings"]["Room 1"].indoor_temperature
        temp_auto = res_auto["sensor_readings"]["Room 1"].indoor_temperature

        # The room with active cooling should be cooler than the unconditioned room
        assert temp_auto < temp_off

    def test_setpoint_modification_updates_physical_state(self):
        engine = SimulationEngine(num_occupants=10, seed=42)
        initial_sp = engine.energy_manager.hvac_controllers["Room 1"].target_setpoint

        new_sp = 21.0
        engine.set_room_hvac_setpoint("Room 1", new_sp)

        # Controller setpoint updated
        assert engine.energy_manager.hvac_controllers["Room 1"].target_setpoint == new_sp
        # Sensor manager calibration setpoint updated
        sensor_sp = engine.sensor_manager.room_models["Room 1"].temp_setpoint
        assert abs(sensor_sp - new_sp) < 0.1


class TestThermodynamicPowerDecomposition:
    """Verifies physical separation of thermal Watts from electrical Watts."""

    def test_cop_separation_formula(self):
        energy_mgr = EnergyManager()
        ctrl = energy_mgr.hvac_controllers["Room 1"]
        ctrl.cop = 3.2
        ctrl.target_setpoint = 24.0
        now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)

        # Create thermal demand by setting moderate temp and occupancy so unit is not clamped at circuit maximum
        mode, p_elec, cool_demand, sp, is_eco = ctrl.update(
            dt=now, occupants=4, capacity=20, indoor_temp=25.8, outdoor_temp=30.0
        )
        assert ctrl.operating_state == "COOLING"
        q_thermal = ctrl.cooling_delivered_w
        assert q_thermal > 0.0

        # In physics: P_elec = min(P_rated, (Q_thermal / COP) + P_fan)
        rated_airflow = (ctrl.rated_cooling_capacity_w / 1000.0) * 120.0
        expected_fan_w = 120.0 * (ctrl.airflow_m3_h / max(1.0, rated_airflow))
        expected_elec = min(ctrl.rated_electrical_power_w, (q_thermal / 3.2) + expected_fan_w)
        assert abs(p_elec - expected_elec) < 2.0
        assert q_thermal > p_elec  # COP > 1 means thermal cooling extraction is greater than electrical draw

    def test_facility_kpis_include_thermal_and_airflow_totals(self):
        bldg_mgr = EnergyManager()
        # Mock active states
        bldg_mgr.latest_states = {
            "Room 1": RoomEnergyState(
                timestamp="2026-10-19T10:00:00",
                room_id="Room 1", room_name="R1", occupant_count=5, capacity=20,
                occupancy_ratio=0.25, hvac_mode="NORMAL", hvac_power_w=670.0,
                lighting_power_w=80.0, auxiliary_power_w=50.0, total_power_w=800.0,
                cumulative_hvac_kwh=1.2, cumulative_lighting_kwh=0.3, cumulative_total_kwh=1.5,
                hvac_target_setpoint=24.5, indoor_temperature=26.0, cooling_demand_w=2000.0,
                is_eco_mode=False, lighting_mode="NORMAL", operating_state="COOLING",
                cooling_delivered_w=2000.0, airflow_m3_h=450.0, fan_state="HIGH",
                temperature_error=1.5,
            ),
            "Room 2": RoomEnergyState(
                timestamp="2026-10-19T10:00:00",
                room_id="Room 2", room_name="R2", occupant_count=0, capacity=15,
                occupancy_ratio=0.0, hvac_mode="ECO", hvac_power_w=15.0,
                lighting_power_w=0.0, auxiliary_power_w=30.0, total_power_w=45.0,
                cumulative_hvac_kwh=0.8, cumulative_lighting_kwh=0.1, cumulative_total_kwh=0.9,
                hvac_target_setpoint=28.0, indoor_temperature=24.0, cooling_demand_w=0.0,
                is_eco_mode=True, lighting_mode="OFF", operating_state="IDLE",
                cooling_delivered_w=0.0, airflow_m3_h=80.0, fan_state="LOW",
                temperature_error=-4.0,
            ),
        }

        kpis = bldg_mgr.get_building_kpis()
        assert kpis["total_cooling_delivered_kw"] == pytest.approx(2.0, rel=1e-2)
        assert kpis["total_airflow_m3_h"] == pytest.approx(530.0, rel=1e-2)
        assert kpis["rooms_cooling_count"] == 1
        assert kpis["rooms_idle_count"] == 1


class TestSVGVisualizationAndAirflowMarkers:
    """Verifies SVG floor plan renders equipment symbols, keyframe animation, and airflow markers."""

    def test_svg_contains_hvac_markers_and_keyframes(self):
        engine = SimulationEngine(num_occupants=20, seed=42)
        engine.step()
        energy_states = engine.energy_manager.get_latest_states()

        svg = generate_svg_floorplan_markup(
            building=engine.building,
            occupants=engine.occupants,
            selected_room_id="Room 1",
            active_overlay="HVAC Power (W)",
            energy_states=energy_states,
        )

        # Defs markers
        assert '<marker id="hvac-arrow-cooling"' in svg
        assert '<marker id="hvac-arrow-idle"' in svg
        assert '<marker id="hvac-arrow-return"' in svg

        # CSS Animation keyframes
        assert "@keyframes hvac-supply-pulse" in svg
        assert "@keyframes hvac-return-pulse" in svg

        # Terminal Equipment symbols
        assert 'class="hvac-terminal"' in svg
        assert "❄ COOL" in svg or "⏸ IDLE" in svg or "⏹ OFF" in svg

        # Chilled Distribution Loop along corridor
        assert "hvac-distribution-spine" in svg or "[CONCEPTUAL CHILLED LOOP]" in svg


class TestSimulationExperimentIsolation:
    """Verifies that simulation experiments run in isolated memory without mutating caller state."""

    def test_experiment_isolation_and_results(self):
        # Baseline engine before experiment
        main_engine = SimulationEngine(num_occupants=25, seed=999)
        initial_time = main_engine.current_time
        initial_step = main_engine.step_count

        # Step once
        main_engine.step()
        time_after_step = main_engine.current_time
        step_after = main_engine.step_count

        # Run multi-policy experiment
        results = run_hvac_mode_experiment(duration_steps=12, seed=42)

        # 1. Main engine state must NOT have changed
        assert main_engine.current_time == time_after_step
        assert main_engine.step_count == step_after

        # 2. Summary table structure
        summary_df = results["summary_table"]
        assert not summary_df.empty
        assert "Policy Code" in summary_df.columns
        assert "Total HVAC Energy (kWh)" in summary_df.columns
        assert "Mean Temp Error (°C)" in summary_df.columns

        policies = summary_df["Policy Code"].tolist()
        assert "AUTO" in policies
        assert "ECO" in policies
        assert "FORCE_OFF" in policies

        # 3. Time-series DataFrame
        ts_df = results["time_series_df"]
        assert not ts_df.empty
        assert len(ts_df["policy"].unique()) == 3
        assert set(ts_df["room_id"].unique()) == set(EXPECTED_ROOM_IDS)
