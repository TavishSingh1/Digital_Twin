"""
Smart HVAC & Room-Wise Energy Management Module (Phase 7)
=========================================================
Implements occupancy-responsive HVAC control, daylight-harvesting lighting,
room-wise instantaneous power monitoring, cumulative energy accounting,
and comparative baseline savings evaluation.

Governing Principles:
1. Occupancy-Responsive Modulation: HVAC output and lighting dimming are modulated
   by real-time room occupancy, sensible heat gains, and environmental conditions.
2. Thermodynamic Separation: Cooling capacity (W) and electrical power (W) are
   conceptually separated via the Coefficient of Performance (COP).
3. Strictly Causal Integration: Energy advances exclusively when logical simulation
   time progresses. Repeated UI renderings and sensor queries at the same timestamp
   are strictly idempotent and do not consume simulated energy.
4. Transparent Baseline Benchmarking: Parallel evaluation of a standard non-smart
   control policy on the identical simulation trace to calculate honest energy savings.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Any
import numpy as np
import pandas as pd

from simulation.building import Building
from simulation.config import EXPECTED_ROOM_IDS


def load_energy_config(filepath: str = "config/energy_parameters.json") -> dict[str, Any]:
    """Loads energy parameter configuration from JSON with validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Energy configuration file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Basic schema validation
    if "rooms" not in cfg or "global_defaults" not in cfg:
        raise ValueError("Invalid energy config: missing 'rooms' or 'global_defaults' section.")
    for r in EXPECTED_ROOM_IDS:
        if r not in cfg["rooms"]:
            raise ValueError(f"Invalid energy config: missing configuration for {r}")
    return cfg


# -----------------------------------------------------------------------------
# ROOM ENERGY STATE DATACLASS
# -----------------------------------------------------------------------------

@dataclass
class RoomEnergyState:
    """Synchronized instantaneous power and cumulative energy telemetry for a single room."""
    timestamp: str
    room_id: str
    room_name: str
    occupant_count: int
    capacity: int
    occupancy_ratio: float
    hvac_mode: str  # "OFF", "ECO", "NORMAL", "HIGH_DEMAND"
    hvac_power_w: float
    lighting_power_w: float
    auxiliary_power_w: float
    total_power_w: float
    cumulative_hvac_kwh: float
    cumulative_lighting_kwh: float
    cumulative_total_kwh: float
    hvac_target_setpoint: float
    indoor_temperature: float
    cooling_demand_w: float
    is_eco_mode: bool
    lighting_mode: str  # "AUTO", "FORCE_ON", "FORCE_OFF"
    hvac_manual_override: Optional[str] = None  # None, "AUTO", "ECO", "FORCE_OFF"
    cumulative_auxiliary_kwh: float = 0.0
    # Phase 9: Advanced HVAC Digital Twin Telemetry
    operating_state: str = "IDLE"  # "COOLING", "IDLE", "OFF"
    cooling_delivered_w: float = 0.0  # Thermal cooling output (Watts thermal)
    airflow_m3_h: float = 0.0  # Volumetric supply airflow rate
    fan_state: str = "OFF"  # "OFF", "LOW", "MEDIUM", "HIGH"
    temperature_error: float = 0.0  # indoor_temp - setpoint
    controller_explanation: str = ""

    @property
    def occupancy(self) -> int:
        return self.occupant_count

    @property
    def hvac_setpoint_c(self) -> float:
        return self.hvac_target_setpoint

    @property
    def hvac_cop(self) -> float:
        return 3.2

    @property
    def cop(self) -> float:
        return 3.2


# -----------------------------------------------------------------------------
# SMART HVAC CONTROLLER
# -----------------------------------------------------------------------------

class HVACController:
    """
    Occupancy-responsive virtual HVAC controller for an individual room.
    Modulates cooling output based on indoor/outdoor temperature, target setpoint,
    deadband hysteresis, metabolic sensible heat, and vacancy timers.
    """

    def __init__(
        self,
        room_id: str,
        room_cfg: dict[str, Any],
        global_cfg: dict[str, Any],
    ):
        self.room_id = room_id
        hvac_room = room_cfg.get("hvac", {})
        hvac_glob = global_cfg.get("hvac", {})

        self.rated_cooling_capacity_w = float(hvac_room.get("rated_cooling_capacity_w", hvac_room.get("cooling_capacity_w", 10000.0)))
        self.rated_electrical_power_w = float(hvac_room.get("rated_electrical_power_w", 3125.0))
        self.cop = float(hvac_room.get("cop", hvac_glob.get("cop", 3.2)))
        if self.cop <= 0:
            raise ValueError(f"HVAC COP must be positive, got {self.cop}")

        self.target_setpoint = float(hvac_room.get("target_setpoint_celsius", hvac_glob.get("default_setpoint_celsius", 24.5)))
        self.eco_setpoint = float(hvac_room.get("eco_setpoint_celsius", hvac_glob.get("eco_setpoint_celsius", 28.0)))
        self.deadband = float(hvac_glob.get("deadband_celsius", 0.5))
        self.occupant_sensible_heat_w = float(hvac_glob.get("occupant_sensible_heat_w", 100.0))
        self.vacancy_delay_minutes = float(hvac_glob.get("vacancy_delay_minutes", 10.0))
        self.standby_power_w = float(hvac_room.get("standby_power_w", hvac_glob.get("standby_power_w", 25.0)))
        self.min_stable_power_w = float(hvac_room.get("min_stable_power_w", self.rated_electrical_power_w * 0.15))

        # Dynamic runtime states
        self.manual_override: Optional[str] = None  # None, "AUTO", "ECO", "FORCE_OFF"
        self.vacancy_timer_minutes: float = self.vacancy_delay_minutes  # Initially vacant (ECO)
        self.current_mode: str = "ECO"
        self.current_power_w: float = self.standby_power_w
        self.current_cooling_demand_w: float = 0.0

        # Phase 9: Thermostat Hysteresis & Airflow States
        self.operating_state: str = "IDLE"  # "COOLING", "IDLE", "OFF"
        self.cooling_delivered_w: float = 0.0
        self.airflow_m3_h: float = 0.0
        self.fan_state: str = "OFF"
        self.temperature_error: float = 0.0
        self.controller_explanation: str = "Initialized in standby"
        self.active_setpoint: float = self.target_setpoint

    @property
    def config(self) -> dict[str, Any]:
        return {
            "cooling_capacity_w": self.rated_cooling_capacity_w,
            "rated_cooling_capacity_w": self.rated_cooling_capacity_w,
            "rated_electrical_power_w": self.rated_electrical_power_w,
            "cop": self.cop,
            "standby_power_w": self.standby_power_w,
            "setpoint_normal_c": self.target_setpoint,
            "setpoint_eco_c": self.eco_setpoint,
            "deadband_c": self.deadband,
            "vacancy_delay_minutes": self.vacancy_delay_minutes,
            "min_stable_power_w": self.min_stable_power_w,
        }

    @property
    def vacancy_timer_seconds(self) -> float:
        return self.vacancy_timer_minutes * 60.0

    @vacancy_timer_seconds.setter
    def vacancy_timer_seconds(self, val: float):
        self.vacancy_timer_minutes = val / 60.0

    @property
    def override_mode(self) -> Optional[str]:
        return self.manual_override

    @override_mode.setter
    def override_mode(self, val: Optional[str]):
        self.set_manual_override(val)

    def set_manual_override(self, mode: Optional[str]):
        """Sets or clears manual HVAC override ('AUTO', 'ECO', 'FORCE_OFF', or None)."""
        if mode in (None, "AUTO", "auto"):
            self.manual_override = None
        elif mode in ("ECO", "eco"):
            self.manual_override = "ECO"
        elif mode in ("FORCE_OFF", "force_off", "OFF", "off"):
            self.manual_override = "FORCE_OFF"
        else:
            raise ValueError(f"Invalid HVAC manual override mode: {mode}")

    def set_target_setpoint(self, setpoint_celsius: float):
        """Sets custom occupied target comfort setpoint."""
        if not (16.0 <= setpoint_celsius <= 32.0):
            raise ValueError(f"Target setpoint out of physical range (16-32 C): {setpoint_celsius}")
        self.target_setpoint = float(setpoint_celsius)

    def compute(
        self,
        current_temp: float,
        outdoor_temp: float,
        occupancy: int,
        current_time: Optional[datetime] = None,
        dt_seconds: float = 300.0,
        advance_time: bool = False,
    ) -> tuple[float, float, str, float]:
        """Convenience method returning (power_w, cooling_demand_w, mode, active_setpoint)."""
        from datetime import timezone
        dt = current_time or datetime.now(timezone.utc)
        mode, p_w, q_w, sp, is_eco = self.update(
            dt=dt,
            occupants=occupancy,
            capacity=10,
            indoor_temp=current_temp,
            outdoor_temp=outdoor_temp,
            cadence_minutes=dt_seconds / 60.0,
            advance_time=advance_time,
        )
        return (p_w, q_w, mode, sp)


    def update(
        self,
        dt: datetime,
        occupants: int,
        capacity: int,
        indoor_temp: float,
        outdoor_temp: float,
        cadence_minutes: float = 5.0,
        advance_time: bool = True,
    ) -> tuple[str, float, float, float, bool]:
        """
        Calculates HVAC operating mode, cooling demand, and electrical power with
        thermostat deadband hysteresis, volumetric airflow, and COP thermodynamics.
        Returns:
            (hvac_mode, hvac_electrical_power_w, cooling_demand_w, active_setpoint, is_eco)
        """
        occ_ratio = (occupants / capacity) if capacity > 0 else 0.0

        # 1. Update vacancy timer and operating mode
        if occupants > 0:
            self.vacancy_timer_minutes = 0.0
            is_eco = False
            active_setpoint = self.target_setpoint

            if self.manual_override == "FORCE_OFF":
                mode = "OFF"
            elif self.manual_override == "ECO":
                mode = "ECO"
                is_eco = True
                active_setpoint = self.eco_setpoint
            elif occ_ratio >= 0.75:
                mode = "HIGH_DEMAND"
            else:
                mode = "NORMAL"
        else:
            # Vacant room
            if advance_time:
                self.vacancy_timer_minutes += cadence_minutes
            if self.manual_override == "FORCE_OFF":
                mode = "OFF"
                is_eco = False
                active_setpoint = self.target_setpoint
            elif self.manual_override == "ECO":
                mode = "ECO"
                is_eco = True
                active_setpoint = self.eco_setpoint
            elif self.vacancy_timer_minutes >= self.vacancy_delay_minutes:
                mode = "ECO"
                is_eco = True
                active_setpoint = self.eco_setpoint
            else:
                # Within vacancy delay timeout
                mode = "NORMAL"
                is_eco = False
                active_setpoint = self.target_setpoint

        self.active_setpoint = active_setpoint
        temp_delta = indoor_temp - active_setpoint
        self.temperature_error = round(temp_delta, 2)
        rated_airflow = (self.rated_cooling_capacity_w / 1000.0) * 120.0  # ~120 m^3/h per kW rated cooling

        # 2. Forced OFF handles immediately
        if mode == "OFF":
            self.current_mode = "OFF"
            self.operating_state = "OFF"
            self.fan_state = "OFF"
            self.airflow_m3_h = 0.0
            self.current_power_w = 0.0
            self.cooling_delivered_w = 0.0
            self.current_cooling_demand_w = 0.0
            self.controller_explanation = "HVAC switched OFF via manual operator override. Power: 0 W, Airflow: 0 m³/h."
            return ("OFF", 0.0, 0.0, active_setpoint, False)

        # 3. Cooling demand calculation
        # Internal occupant sensible heat load (W)
        sensible_occ_w = occupants * self.occupant_sensible_heat_w

        # Envelope transmission heat load (approximate UA factor)
        envelope_ua = (self.rated_cooling_capacity_w * 0.30) / max(1.0, 35.0 - active_setpoint)
        transmission_w = max(0.0, envelope_ua * (outdoor_temp - active_setpoint))

        # Temperature pull-down demand
        pulldown_gain = (self.rated_cooling_capacity_w * 0.40)
        pulldown_w = max(0.0, temp_delta * pulldown_gain) if temp_delta > 0 else 0.0

        raw_cooling_demand = sensible_occ_w + transmission_w + pulldown_w

        # 4. Thermostat Deadband Hysteresis Logic
        # Upper turn-on threshold: active_setpoint + deadband
        # Lower turn-off threshold: active_setpoint - deadband
        upper_threshold = active_setpoint + self.deadband
        lower_threshold = active_setpoint - self.deadband

        if self.operating_state == "COOLING":
            # Compressor was running: continue cooling until temperature reaches setpoint / lower deadband
            if indoor_temp <= lower_threshold:
                # Target satisfied: transition compressor to IDLE
                self.operating_state = "IDLE"
            else:
                self.operating_state = "COOLING"
        else:
            # Compressor was IDLE (or previous OFF): only engage when temperature crosses upper threshold
            if indoor_temp > upper_threshold:
                self.operating_state = "COOLING"
            else:
                self.operating_state = "IDLE"

        # 5. Output Modulation based on operating state
        if self.operating_state == "COOLING":
            cooling_demand = raw_cooling_demand
            cooling_output_w = min(self.rated_cooling_capacity_w, max(self.rated_cooling_capacity_w * 0.15, cooling_demand))
            self.cooling_delivered_w = round(cooling_output_w, 1)

            # Airflow modulation: High fan if high demand or large delta, Medium otherwise
            if cooling_output_w >= 0.70 * self.rated_cooling_capacity_w or temp_delta >= 1.5:
                self.fan_state = "HIGH"
                self.airflow_m3_h = round(rated_airflow, 1)
            else:
                self.fan_state = "MEDIUM"
                self.airflow_m3_h = round(rated_airflow * 0.65, 1)

            fan_elec_w = 120.0 * (self.airflow_m3_h / max(1.0, rated_airflow))
            comp_elec_w = cooling_output_w / self.cop
            elec_power = max(self.min_stable_power_w, min(self.rated_electrical_power_w, comp_elec_w + fan_elec_w))

            if is_eco:
                self.controller_explanation = (
                    f"Eco setback cooling: indoor temp ({indoor_temp:.1f}°C) exceeds eco setpoint ({active_setpoint:.1f}°C). "
                    f"Delivering {cooling_output_w:.0f} W thermal extraction at {self.airflow_m3_h:.0f} m³/h ({self.fan_state} fan)."
                )
            else:
                self.controller_explanation = (
                    f"Comfort cooling active: indoor temp ({indoor_temp:.1f}°C) exceeds setpoint ({active_setpoint:.1f}°C) by {temp_delta:+.1f}°C. "
                    f"Delivering {cooling_output_w:.0f} W thermal extraction at {self.airflow_m3_h:.0f} m³/h ({self.fan_state} fan)."
                )
        else:
            # IDLE: compressor off, circulation fan at low speed
            cooling_demand = 0.0
            cooling_output_w = 0.0
            self.cooling_delivered_w = 0.0
            self.fan_state = "LOW"
            self.airflow_m3_h = round(rated_airflow * 0.20, 1)
            elec_power = self.standby_power_w

            if is_eco:
                self.controller_explanation = (
                    f"Eco setback idle: room vacant, temp ({indoor_temp:.1f}°C) within eco setpoint ({active_setpoint:.1f}°C). "
                    f"Circulation fan active ({self.airflow_m3_h:.0f} m³/h)."
                )
            else:
                self.controller_explanation = (
                    f"Thermostat idle: indoor temp ({indoor_temp:.1f}°C) is within deadband of setpoint ({active_setpoint:.1f}°C). "
                    f"Compressor off, low fan circulation ({self.airflow_m3_h:.0f} m³/h)."
                )

        # Clamp limits
        elec_power = max(0.0, min(self.rated_electrical_power_w, elec_power))
        self.current_mode = mode
        self.current_power_w = round(elec_power, 1)
        self.current_cooling_demand_w = round(cooling_demand, 1)

        return (mode, self.current_power_w, self.current_cooling_demand_w, active_setpoint, is_eco)


# -----------------------------------------------------------------------------
# SMART LIGHTING CONTROLLER
# -----------------------------------------------------------------------------

class LightingController:
    """
    Daylight-harvesting and occupancy-responsive lighting controller.
    Modulates LED lighting electrical power based on natural daylight availability,
    occupancy status, and vacancy timeout delays.
    """

    def __init__(
        self,
        room_id: str,
        room_cfg: dict[str, Any],
        global_cfg: dict[str, Any],
    ):
        self.room_id = room_id
        lux_room = room_cfg.get("lighting", {})
        lux_glob = global_cfg.get("lighting", {})

        self.installed_power_w = float(lux_room.get("installed_power_w", 300.0))
        self.target_illuminance_lux = float(lux_room.get("target_illuminance_lux", lux_glob.get("target_illuminance_lux", 300.0)))
        self.min_occupied_fraction = float(lux_room.get("min_occupied_fraction", lux_glob.get("min_occupied_level_fraction", 0.15)))
        self.vacancy_timeout_minutes = float(lux_glob.get("vacancy_timeout_minutes", 10.0))
        self.standby_power_w = float(lux_room.get("standby_power_w", lux_glob.get("standby_power_w", 2.0)))

        # Dynamic runtime states
        self.manual_override: Optional[str] = None  # None, "AUTO", "FORCE_ON", "FORCE_OFF"
        self.vacancy_timer_minutes: float = self.vacancy_timeout_minutes  # Initially vacant (OFF)
        self.current_power_w: float = 0.0
        self.is_light_on: bool = False

    @property
    def config(self) -> dict[str, Any]:
        return {
            "installed_power_w": self.installed_power_w,
            "target_illuminance_lux": self.target_illuminance_lux,
            "min_dim_fraction": self.min_occupied_fraction,
            "min_occupied_fraction": self.min_occupied_fraction,
            "standby_power_w": self.standby_power_w,
            "vacancy_delay_minutes": self.vacancy_timeout_minutes,
            "vacancy_timeout_minutes": self.vacancy_timeout_minutes,
        }

    @property
    def vacancy_timer_seconds(self) -> float:
        return self.vacancy_timer_minutes * 60.0

    @vacancy_timer_seconds.setter
    def vacancy_timer_seconds(self, val: float):
        self.vacancy_timer_minutes = val / 60.0

    @property
    def override_mode(self) -> Optional[str]:
        return self.manual_override

    @override_mode.setter
    def override_mode(self, val: Optional[str]):
        self.set_manual_override(val)

    def set_manual_override(self, mode: Optional[str]):
        """Sets lighting override ('AUTO', 'FORCE_ON', 'FORCE_OFF', or None)."""
        if mode in (None, "AUTO", "auto"):
            self.manual_override = None
        elif mode in ("FORCE_ON", "force_on", "ON", "on"):
            self.manual_override = "FORCE_ON"
        elif mode in ("FORCE_OFF", "force_off", "OFF", "off"):
            self.manual_override = "FORCE_OFF"
        else:
            raise ValueError(f"Invalid lighting override mode: {mode}")

    def compute(
        self,
        occupancy: int,
        daylight_lux: float,
        dt_seconds: float = 300.0,
        advance_time: bool = False,
    ) -> tuple[float, bool, str]:
        """Convenience method returning (power_w, is_on, mode)."""
        from datetime import timezone
        dt = datetime.now(timezone.utc)
        p_w, mode, is_on = self.update(
            dt=dt,
            occupants=occupancy,
            daylight_lux=daylight_lux,
            cadence_minutes=dt_seconds / 60.0,
            advance_time=advance_time,
        )
        return (p_w, is_on, mode)


    def update(
        self,
        dt: datetime,
        occupants: int,
        daylight_lux: float,
        cadence_minutes: float = 5.0,
        advance_time: bool = True,
    ) -> tuple[float, str, bool]:
        """
        Calculates lighting electrical power, active mode, and lamp status.
        Returns:
            (lighting_power_w, active_mode_str, is_light_on_bool)
        """
        # 1. Manual Overrides
        if self.manual_override == "FORCE_ON":
            self.is_light_on = True
            self.current_power_w = self.installed_power_w
            return (self.installed_power_w, "FORCE_ON", True)
        elif self.manual_override == "FORCE_OFF":
            self.is_light_on = False
            self.current_power_w = 0.0
            return (0.0, "FORCE_OFF", False)

        # 2. AUTO Mode: Occupancy & Vacancy Timeout
        if occupants > 0:
            self.vacancy_timer_minutes = 0.0
            self.is_light_on = True


            # Daylight Harvesting: calculate artificial illuminance deficit
            deficit_lux = max(0.0, self.target_illuminance_lux - max(0.0, daylight_lux))

            if deficit_lux <= 0.0:
                # Natural daylight completely satisfies target; dim to minimal fraction
                dim_fraction = self.min_occupied_fraction
            else:
                dim_fraction = min(1.0, max(self.min_occupied_fraction, deficit_lux / self.target_illuminance_lux))

            # LED driver dimming power curve: P = P_installed * (0.10 + 0.90 * dim_fraction)
            elec_power = self.installed_power_w * (0.10 + 0.90 * dim_fraction)
            self.current_power_w = round(elec_power, 1)
            return (self.current_power_w, "AUTO", True)
        else:
            # Vacant
            if advance_time:
                self.vacancy_timer_minutes += cadence_minutes

            if self.vacancy_timer_minutes >= self.vacancy_timeout_minutes:
                # Vacancy timeout expired: turn lights off
                self.is_light_on = False
                self.current_power_w = 0.0
                return (0.0, "AUTO", False)
            else:
                # Within vacancy delay: maintain minimal standby lighting
                self.is_light_on = True
                elec_power = self.installed_power_w * self.min_occupied_fraction
                self.current_power_w = round(elec_power, 1)
                return (self.current_power_w, "AUTO", True)


# -----------------------------------------------------------------------------
# PARALLEL BASELINE POLICY SIMULATOR
# -----------------------------------------------------------------------------

class BaselinePolicySimulator:
    """
    Evaluates a standard non-smart building control policy on the exact same
    occupancy trace for fair, reproducible energy savings benchmarking:
    - Constant unmodulated cooling setpoint (24.0 C) with no eco mode.
    - Constant 100% lighting during business hours (08:00 - 18:00) with no daylight harvesting.
    """

    def __init__(self, room_id: str, room_cfg: dict[str, Any], global_cfg: dict[str, Any]):
        self.room_id = room_id
        hvac_room = room_cfg.get("hvac", {})
        hvac_glob = global_cfg.get("hvac", {})
        base_glob = global_cfg.get("baseline_policy", {})

        self.rated_cooling_w = float(hvac_room.get("rated_cooling_capacity_w", 10000.0))
        self.rated_elec_w = float(hvac_room.get("rated_electrical_power_w", 3125.0))
        self.cop = float(hvac_room.get("cop", hvac_glob.get("cop", 3.2)))
        self.fixed_setpoint = float(base_glob.get("fixed_setpoint_celsius", 24.0))

        lux_room = room_cfg.get("lighting", {})
        self.installed_lux_w = float(lux_room.get("installed_power_w", 300.0))
        self.aux_w = float(room_cfg.get("auxiliary", {}).get("base_power_w", 100.0))

        self.cumulative_hvac_kwh: float = 0.0
        self.cumulative_lighting_kwh: float = 0.0
        self.cumulative_total_kwh: float = 0.0

    def step(
        self,
        dt: datetime,
        occupants: int,
        indoor_temp: float,
        outdoor_temp: float,
        elapsed_seconds: float = 300.0,
    ) -> tuple[float, float, float]:
        """
        Calculates baseline power and increments cumulative baseline energy.
        Returns:
            (hvac_power_w, lighting_power_w, total_power_w)
        """
        # Baseline HVAC: constant setpoint, no eco setback
        temp_delta = max(0.0, indoor_temp - self.fixed_setpoint)
        trans_w = max(0.0, 100.0 * (outdoor_temp - self.fixed_setpoint))
        occ_w = occupants * 100.0
        load_w = occ_w + trans_w + (temp_delta * 4000.0)
        cooling_w = min(self.rated_cooling_w, max(self.rated_cooling_w * 0.25, load_w))
        hvac_power_w = min(self.rated_elec_w, max(self.rated_elec_w * 0.25, cooling_w / self.cop))

        # Baseline Lighting: 100% on during business hours (8 to 18), or full on if occupied
        hour_frac = dt.hour + (dt.minute / 60.0)
        is_bus_hours = (8.0 <= hour_frac < 18.0)
        if is_bus_hours or occupants > 0:
            lighting_power_w = self.installed_lux_w
        else:
            lighting_power_w = 0.0

        total_power_w = hvac_power_w + lighting_power_w + self.aux_w

        # Accumulate
        hrs = elapsed_seconds / 3600.0
        self.cumulative_hvac_kwh += (hvac_power_w * hrs) / 1000.0
        self.cumulative_lighting_kwh += (lighting_power_w * hrs) / 1000.0
        self.cumulative_total_kwh += (total_power_w * hrs) / 1000.0

        return (hvac_power_w, lighting_power_w, total_power_w)


class RoomControllerPair:
    """Pair of HVAC and Lighting controllers for convenient test and diagnostic access."""
    def __init__(self, hvac: HVACController, lighting: LightingController):
        self.hvac = hvac
        self.lighting = lighting


class AliasedRoomDict(dict):
    """Dictionary holding 5 canonical rooms while supporting transparent access by either 'Room 1' or 'Room_1'."""
    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        normalized = key.replace("_", " ") if isinstance(key, str) else key
        if super().__contains__(normalized):
            return super().__getitem__(normalized)
        raise KeyError(key)

    def get(self, key, default=None):
        if super().__contains__(key):
            return super().get(key, default)
        normalized = key.replace("_", " ") if isinstance(key, str) else key
        if super().__contains__(normalized):
            return super().get(normalized, default)
        return default

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        normalized = key.replace("_", " ") if isinstance(key, str) else key
        return super().__contains__(normalized)



# -----------------------------------------------------------------------------
# ENERGY MANAGER (COORDINATOR)
# -----------------------------------------------------------------------------


class EnergyManager:
    """
    Central energy management coordinator for the 5-room facility.
    Manages HVAC and Lighting controllers for all rooms, records room-wise power,
    accumulates energy strictly on logical time steps, and computes comparative savings.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.config = config or load_energy_config()
        self.seed = seed
        self.global_defaults = self.config.get("global_defaults", {})
        self.rooms_cfg = self.config.get("rooms", {})

        # Strict validation of physical ratings
        for r_id, r_cfg in self.rooms_cfg.items():
            cop = r_cfg.get("hvac", {}).get("cop", self.global_defaults.get("hvac", {}).get("cop", 3.2))
            if cop <= 0:
                raise ValueError(f"COP must be positive, got {cop}")

        # Sub-controllers per room
        self.hvac_controllers: dict[str, HVACController] = {}
        self.lighting_controllers: dict[str, LightingController] = {}
        self.baseline_simulators: dict[str, BaselinePolicySimulator] = {}
        self.auxiliary_loads: dict[str, float] = {}

        for room_id in EXPECTED_ROOM_IDS:
            r_cfg = self.rooms_cfg.get(room_id, {})
            self.hvac_controllers[room_id] = HVACController(room_id, r_cfg, self.global_defaults)
            self.lighting_controllers[room_id] = LightingController(room_id, r_cfg, self.global_defaults)
            self.baseline_simulators[room_id] = BaselinePolicySimulator(room_id, r_cfg, self.global_defaults)
            self.auxiliary_loads[room_id] = float(r_cfg.get("auxiliary", {}).get("base_power_w", 100.0))

        # Cumulative energy tracking per room
        self.cumulative_hvac_kwh: dict[str, float] = {r: 0.0 for r in EXPECTED_ROOM_IDS}
        self.cumulative_lighting_kwh: dict[str, float] = {r: 0.0 for r in EXPECTED_ROOM_IDS}
        self.cumulative_auxiliary_kwh: dict[str, float] = {r: 0.0 for r in EXPECTED_ROOM_IDS}
        self.cumulative_total_kwh: dict[str, float] = {r: 0.0 for r in EXPECTED_ROOM_IDS}

        # Telemetry history and idempotency protection
        self.last_accumulated_timestamp: Optional[str] = None
        self.latest_states: dict[str, RoomEnergyState] = {}
        self.history: list[RoomEnergyState] = []

    @property
    def controllers(self) -> AliasedRoomDict:
        """Provides access to room controllers with support for both 'Room 1' and 'Room_1' keys."""
        return AliasedRoomDict({
            r: RoomControllerPair(self.hvac_controllers[r], self.lighting_controllers[r])
            for r in EXPECTED_ROOM_IDS
        })


    def set_hvac_mode_override(self, room_id: str, mode: Optional[str]):
        """Sets manual HVAC override ('AUTO', 'ECO', 'FORCE_OFF', or None) for a room."""
        r_key = room_id.replace("_", " ")
        if r_key in self.hvac_controllers:
            self.hvac_controllers[r_key].set_manual_override(mode)

    def set_hvac_setpoint(self, room_id: str, setpoint_celsius: float):
        """Sets custom HVAC target comfort temperature for a room."""
        r_key = room_id.replace("_", " ")
        if r_key in self.hvac_controllers:
            self.hvac_controllers[r_key].set_target_setpoint(setpoint_celsius)

    def set_lighting_override(self, room_id: str, mode: Optional[str]):
        """Sets manual lighting override ('AUTO', 'FORCE_ON', 'FORCE_OFF') for a room."""
        r_key = room_id.replace("_", " ")
        if r_key in self.lighting_controllers:
            self.lighting_controllers[r_key].set_manual_override(mode)

    def step(
        self,
        current_time: datetime,
        building: Building,
        sensor_manager: Any,
        elapsed_seconds: float = 300.0,
        **kwargs: Any,
    ) -> dict[str, RoomEnergyState]:
        """
        Advances energy management by one simulation time step:
        - Evaluates smart HVAC and lighting controllers.
        - Accumulates cumulative energy (kWh) strictly once per time step.
        - Updates parallel baseline policy simulator.
        """
        if "dt_seconds" in kwargs:
            elapsed_seconds = float(kwargs["dt_seconds"])

        ts_str = current_time.isoformat()
        is_new_step = (ts_str != self.last_accumulated_timestamp)

        step_states = {}
        latest_sensors = sensor_manager.get_latest_readings()

        for room_id in EXPECTED_ROOM_IDS:
            room = building.get_room(room_id)
            occ = room.occupant_count
            cap = room.capacity
            occ_ratio = (occ / cap) if cap > 0 else 0.0

            sensor_r = latest_sensors.get(room_id)
            indoor_temp = getattr(sensor_r, "indoor_temperature", 24.5) if sensor_r else 24.5
            outdoor_temp = getattr(sensor_r, "dry_bulb_temperature", 28.0) if sensor_r else 28.0

            meta = sensor_manager.get_sensor_metadata(room_id)
            daylight_lux = meta.get("daylight_lux", 0.0)

            # 1. Update HVAC controller
            hvac_ctrl = self.hvac_controllers[room_id]
            hvac_mode, hvac_w, cool_demand_w, active_setpoint, is_eco = hvac_ctrl.update(
                dt=current_time,
                occupants=occ,
                capacity=cap,
                indoor_temp=indoor_temp,
                outdoor_temp=outdoor_temp,
                cadence_minutes=elapsed_seconds / 60.0,
                advance_time=is_new_step,
            )

            # 2. Update Lighting controller
            light_ctrl = self.lighting_controllers[room_id]
            light_w, light_mode_str, is_light_on = light_ctrl.update(
                dt=current_time,
                occupants=occ,
                daylight_lux=daylight_lux,
                cadence_minutes=elapsed_seconds / 60.0,
                advance_time=is_new_step,
            )

            # 3. Auxiliary & Total power
            aux_w = self.auxiliary_loads.get(room_id, 100.0)
            total_w = round(hvac_w + light_w + aux_w, 1)

            # 4. Energy Accumulation (idempotent: strictly once per new step)
            if is_new_step:
                hrs = elapsed_seconds / 3600.0
                self.cumulative_hvac_kwh[room_id] += (hvac_w * hrs) / 1000.0
                self.cumulative_lighting_kwh[room_id] += (light_w * hrs) / 1000.0
                self.cumulative_auxiliary_kwh[room_id] += (aux_w * hrs) / 1000.0
                self.cumulative_total_kwh[room_id] += (total_w * hrs) / 1000.0

                # Also advance baseline simulator
                self.baseline_simulators[room_id].step(
                    dt=current_time,
                    occupants=occ,
                    indoor_temp=indoor_temp,
                    outdoor_temp=outdoor_temp,
                    elapsed_seconds=elapsed_seconds,
                )

            state = RoomEnergyState(
                timestamp=ts_str,
                room_id=room_id,
                room_name=room.name,
                occupant_count=occ,
                capacity=cap,
                occupancy_ratio=round(occ_ratio, 3),
                hvac_mode=hvac_mode,
                hvac_power_w=hvac_w,
                lighting_power_w=light_w,
                auxiliary_power_w=aux_w,
                total_power_w=total_w,
                cumulative_hvac_kwh=round(self.cumulative_hvac_kwh[room_id], 4),
                cumulative_lighting_kwh=round(self.cumulative_lighting_kwh[room_id], 4),
                cumulative_total_kwh=round(self.cumulative_total_kwh[room_id], 4),
                hvac_target_setpoint=round(active_setpoint, 2),
                indoor_temperature=round(indoor_temp, 2),
                cooling_demand_w=round(cool_demand_w, 1),
                is_eco_mode=is_eco,
                lighting_mode=light_mode_str,
                hvac_manual_override=hvac_ctrl.manual_override,
                cumulative_auxiliary_kwh=round(self.cumulative_auxiliary_kwh[room_id], 4),
                operating_state=hvac_ctrl.operating_state,
                cooling_delivered_w=round(hvac_ctrl.cooling_delivered_w, 1),
                airflow_m3_h=round(hvac_ctrl.airflow_m3_h, 1),
                fan_state=hvac_ctrl.fan_state,
                temperature_error=round(indoor_temp - active_setpoint, 2),
                controller_explanation=hvac_ctrl.controller_explanation,
            )

            step_states[room_id] = state
            if is_new_step:
                self.history.append(state)

        if is_new_step:
            self.last_accumulated_timestamp = ts_str

        self.latest_states = step_states
        return step_states

    def refresh_immediate(
        self,
        current_time: datetime,
        building: Building,
        sensor_manager: Any,
    ) -> dict[str, RoomEnergyState]:
        """
        Immediately updates instantaneous control demand and electrical power
        following occupant entry, exit, movement, or manual override without
        advancing logical clock or accumulating additional cumulative kWh.
        """
        ts_str = current_time.isoformat()
        latest_sensors = sensor_manager.get_latest_readings()

        for room_id in EXPECTED_ROOM_IDS:
            room = building.get_room(room_id)
            occ = room.occupant_count
            cap = room.capacity
            occ_ratio = (occ / cap) if cap > 0 else 0.0

            sensor_r = latest_sensors.get(room_id)
            indoor_temp = getattr(sensor_r, "indoor_temperature", 24.5) if sensor_r else 24.5
            outdoor_temp = getattr(sensor_r, "dry_bulb_temperature", 28.0) if sensor_r else 28.0

            meta = sensor_manager.get_sensor_metadata(room_id)
            daylight_lux = meta.get("daylight_lux", 0.0)

            # Recalculate without advancing vacancy timers or time
            hvac_ctrl = self.hvac_controllers[room_id]
            hvac_mode, hvac_w, cool_demand_w, active_setpoint, is_eco = hvac_ctrl.update(
                dt=current_time,
                occupants=occ,
                capacity=cap,
                indoor_temp=indoor_temp,
                outdoor_temp=outdoor_temp,
                advance_time=False,
            )

            light_ctrl = self.lighting_controllers[room_id]
            light_w, light_mode_str, is_light_on = light_ctrl.update(
                dt=current_time,
                occupants=occ,
                daylight_lux=daylight_lux,
                advance_time=False,
            )

            aux_w = self.auxiliary_loads.get(room_id, 100.0)
            total_w = round(hvac_w + light_w + aux_w, 1)

            state = RoomEnergyState(
                timestamp=ts_str,
                room_id=room_id,
                room_name=room.name,
                occupant_count=occ,
                capacity=cap,
                occupancy_ratio=round(occ_ratio, 3),
                hvac_mode=hvac_mode,
                hvac_power_w=hvac_w,
                lighting_power_w=light_w,
                auxiliary_power_w=aux_w,
                total_power_w=total_w,
                cumulative_hvac_kwh=round(self.cumulative_hvac_kwh[room_id], 4),
                cumulative_lighting_kwh=round(self.cumulative_lighting_kwh[room_id], 4),
                cumulative_total_kwh=round(self.cumulative_total_kwh[room_id], 4),
                hvac_target_setpoint=round(active_setpoint, 2),
                indoor_temperature=round(indoor_temp, 2),
                cooling_demand_w=round(cool_demand_w, 1),
                is_eco_mode=is_eco,
                lighting_mode=light_mode_str,
                hvac_manual_override=hvac_ctrl.manual_override,
                cumulative_auxiliary_kwh=round(self.cumulative_auxiliary_kwh[room_id], 4),
                operating_state=hvac_ctrl.operating_state,
                cooling_delivered_w=round(hvac_ctrl.cooling_delivered_w, 1),
                airflow_m3_h=round(hvac_ctrl.airflow_m3_h, 1),
                fan_state=hvac_ctrl.fan_state,
                temperature_error=round(indoor_temp - active_setpoint, 2),
                controller_explanation=hvac_ctrl.controller_explanation,
            )
            self.latest_states[room_id] = state

            # Update matching entry in history if current timestamp matches
            if self.history:
                for idx in range(len(self.history) - 1, max(-1, len(self.history) - 1 - len(EXPECTED_ROOM_IDS)), -1):
                    if self.history[idx].room_id == room_id and self.history[idx].timestamp == ts_str:
                        self.history[idx] = state
                        break

        return self.get_latest_states()

    def get_latest_states(self) -> AliasedRoomDict:
        """Returns the most recent energy states for all 5 rooms, accessible by space or underscore name."""
        return AliasedRoomDict(self.latest_states)


    def get_building_kpis(self) -> dict[str, Any]:
        """Calculates instantaneous building-wide power and cumulative energy KPIs."""
        if not self.latest_states:
            return {
                "total_power_kw": 0.0,
                "hvac_power_kw": 0.0,
                "lighting_power_kw": 0.0,
                "auxiliary_power_kw": 0.0,
                "cumulative_total_kwh": 0.0,
                "rooms_in_eco_count": 0,
                "total_occupants": 0,
            }

        # Use canonical room IDs to prevent double counting
        canonical_states = [self.latest_states[r] for r in EXPECTED_ROOM_IDS if r in self.latest_states]

        total_p = sum(s.total_power_w for s in canonical_states)
        hvac_p = sum(s.hvac_power_w for s in canonical_states)
        light_p = sum(s.lighting_power_w for s in canonical_states)
        aux_p = sum(s.auxiliary_power_w for s in canonical_states)
        cum_total = sum(self.cumulative_total_kwh[r] for r in EXPECTED_ROOM_IDS)
        cum_hvac = sum(self.cumulative_hvac_kwh[r] for r in EXPECTED_ROOM_IDS)
        eco_count = sum(1 for s in canonical_states if s.is_eco_mode or s.hvac_mode == "ECO")
        total_occ = sum(s.occupant_count for s in canonical_states)

        # Phase 9 facility-level HVAC KPIs
        cooling_count = sum(1 for s in canonical_states if getattr(s, "operating_state", "") == "COOLING")
        idle_count = sum(1 for s in canonical_states if getattr(s, "operating_state", "") == "IDLE")
        off_count = sum(1 for s in canonical_states if getattr(s, "operating_state", "") == "OFF")
        tot_cooling_w = sum(getattr(s, "cooling_delivered_w", 0.0) for s in canonical_states)
        tot_airflow = sum(getattr(s, "airflow_m3_h", 0.0) for s in canonical_states)
        temp_errors = [abs(getattr(s, "temperature_error", s.indoor_temperature - s.hvac_target_setpoint)) for s in canonical_states]
        avg_temp_error = (sum(temp_errors) / len(temp_errors)) if temp_errors else 0.0

        return {
            "total_power_kw": round(total_p / 1000.0, 3),
            "hvac_power_kw": round(hvac_p / 1000.0, 3),
            "lighting_power_kw": round(light_p / 1000.0, 3),
            "auxiliary_power_kw": round(aux_p / 1000.0, 3),
            "cumulative_total_kwh": round(cum_total, 4),
            "cumulative_hvac_kwh": round(cum_hvac, 4),
            "rooms_in_eco_count": eco_count,
            "total_occupants": total_occ,
            "rooms_cooling_count": cooling_count,
            "rooms_idle_count": idle_count,
            "rooms_off_count": off_count,
            "total_cooling_delivered_kw": round(tot_cooling_w / 1000.0, 3),
            "total_airflow_m3_h": round(tot_airflow, 1),
            "avg_temp_error_celsius": round(avg_temp_error, 2),
        }

    def get_comparison_metrics(self) -> dict[str, Any]:
        """
        Computes comparative energy savings between the smart occupancy-responsive
        policy and the unmodulated baseline policy on the exact same simulation trace.
        """
        smart_total_kwh = sum(self.cumulative_total_kwh[r] for r in EXPECTED_ROOM_IDS)
        baseline_total_kwh = sum(b.cumulative_total_kwh for b in self.baseline_simulators.values())

        savings_kwh = max(0.0, baseline_total_kwh - smart_total_kwh)
        savings_pct = (100.0 * savings_kwh / baseline_total_kwh) if baseline_total_kwh > 0.001 else 0.0

        smart_df = self.to_dataframe()
        if not smart_df.empty:
            peak_smart_kw = smart_df.groupby("timestamp")["total_power_w"].sum().max() / 1000.0
            avg_smart_kw = smart_df.groupby("timestamp")["total_power_w"].sum().mean() / 1000.0
        else:
            peak_smart_kw = 0.0
            avg_smart_kw = 0.0

        return {
            "smart_total_kwh": round(smart_total_kwh, 3),
            "baseline_total_kwh": round(baseline_total_kwh, 3),
            "savings_kwh": round(savings_kwh, 3),
            "savings_percent": round(savings_pct, 2),
            "peak_smart_demand_kw": round(peak_smart_kw, 3),
            "avg_smart_demand_kw": round(avg_smart_kw, 3),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Converts logged energy history into a pandas DataFrame."""
        if not self.history:
            cols = [
                "timestamp", "room_id", "room_name", "occupant_count", "capacity",
                "occupancy_ratio", "hvac_mode", "hvac_power_w", "lighting_power_w",
                "auxiliary_power_w", "total_power_w", "cumulative_hvac_kwh",
                "cumulative_lighting_kwh", "cumulative_auxiliary_kwh", "cumulative_total_kwh",
                "hvac_target_setpoint", "indoor_temperature", "cooling_demand_w", "is_eco_mode",
                "lighting_mode", "operating_state", "cooling_delivered_w", "airflow_m3_h",
                "fan_state", "temperature_error", "controller_explanation",
            ]
            return pd.DataFrame(columns=cols)
        data = [asdict(s) for s in self.history]
        return pd.DataFrame(data)

    def export_csv(self, filepath: str = "artifacts/room_energy_consumption.csv"):
        """Exports room-wise energy history to CSV."""
        df = self.to_dataframe()
        os_dir = os.path.dirname(filepath)
        if os_dir:
            os.makedirs(os_dir, exist_ok=True)
        df.to_csv(filepath, index=False)

    def reset(self, new_seed: Optional[int] = None):
        """Resets all energy controllers, baseline simulators, and cumulative meters."""
        if new_seed is not None:
            self.seed = new_seed

        for room_id in EXPECTED_ROOM_IDS:
            r_cfg = self.rooms_cfg.get(room_id, {})
            self.hvac_controllers[room_id] = HVACController(room_id, r_cfg, self.global_defaults)
            self.lighting_controllers[room_id] = LightingController(room_id, r_cfg, self.global_defaults)
            self.baseline_simulators[room_id] = BaselinePolicySimulator(room_id, r_cfg, self.global_defaults)
            self.cumulative_hvac_kwh[room_id] = 0.0
            self.cumulative_lighting_kwh[room_id] = 0.0
            self.cumulative_auxiliary_kwh[room_id] = 0.0
            self.cumulative_total_kwh[room_id] = 0.0

        self.last_accumulated_timestamp = None
        self.latest_states.clear()
        self.history.clear()

