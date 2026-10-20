"""
Virtual Environmental Sensor Engine (Phase 6 Physical Reform)
=============================================================
Models physically motivated, deterministic environmental sensors for all 5 rooms:
1. illuminance [lux] — combined daylight window penetration and policy-driven artificial lighting
2. sound_pressure_level [dBA] — logarithmic acoustic energy summation of ambient noise and human activity
3. indoor_temperature [Celsius] — lumped thermal capacitance balancing envelope, HVAC, and metabolic heat
4. dry_bulb_temperature [Celsius] — ambient outdoor weather condition (strictly independent of indoor occupancy)
5. indoor_relative_humidity [%] — derived from moisture mass balance and temperature saturation vapor pressure
6. outdoor_relative_humidity [%] — atmospheric ambient humidity (strictly independent of indoor occupancy)
7. indoor_co2 [ppm] — room-level mass-balance differential equation with exact analytical exponential decay

Dual-Timing Update Architecture:
- Immediate Updates: Sound pressure level, artificial lighting, and instantaneous occupant source terms
  update immediately after entry, exit, or room-to-room movement without advancing the simulation clock.
- Time-Dependent Updates: Indoor temperature, moisture/relative humidity, and CO2 evolve dynamically
  over continuous 5-minute logical clock steps according to differential mass and energy balances.
"""

import os
import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Any
import pandas as pd

from simulation.environment import OutdoorWeather
from simulation.building import Building
from simulation.config import EXPECTED_ROOM_IDS


def load_sensor_config(filepath: str = "config/sensor_parameters.json") -> dict[str, Any]:
    """Loads physical sensor parameter configuration from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sensor parameters config not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# PSYCHROMETRIC THERMODYNAMIC UTILITIES
# ---------------------------------------------------------------------

def saturation_vapor_pressure_kpa(temp_celsius: float) -> float:
    """
    Magnus-Tetens formulation for saturation vapor pressure over liquid water (kPa).
    Valid for standard building environmental temperatures (-20 C to +50 C).
    """
    return 0.61078 * math.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))


def vapor_density_from_rh(rh_percent: float, temp_celsius: float) -> float:
    """
    Calculates absolute moisture density (vapor concentration in g/m^3)
    from relative humidity (%) and air temperature (Celsius).
    Uses ideal gas law for water vapor: rho_v = (p_v * 1000) / (R_v * T_k).
    R_v = 0.4615 kJ/(kg*K).
    """
    p_sat = saturation_vapor_pressure_kpa(temp_celsius)
    p_v = (max(0.0, min(100.0, rh_percent)) / 100.0) * p_sat  # kPa
    t_kelvin = temp_celsius + 273.15
    return (p_v * 1000.0) / (0.4615 * t_kelvin)


def rh_from_vapor_density(vapor_density_g_m3: float, temp_celsius: float) -> float:
    """
    Derives relative humidity (%) from absolute moisture density (g/m^3) and air temperature (Celsius).
    Accounts for the exponential growth of saturation vapor pressure with temperature.
    """
    t_kelvin = temp_celsius + 273.15
    p_v = (max(0.1, vapor_density_g_m3) * 0.4615 * t_kelvin) / 1000.0  # kPa
    p_sat = saturation_vapor_pressure_kpa(temp_celsius)
    if p_sat <= 0.0:
        return 50.0
    return (p_v / p_sat) * 100.0


# ---------------------------------------------------------------------
# STANDARDIZED TELEMETRY READING DATACLASS
# ---------------------------------------------------------------------

@dataclass
class RoomSensorReading:
    """Represents a synchronized 7-feature environmental reading for a room."""
    timestamp: str
    room_id: str
    illuminance: float
    sound_pressure_level: float
    indoor_temperature: float
    dry_bulb_temperature: float
    indoor_relative_humidity: float
    outdoor_relative_humidity: float
    indoor_co2: float
    is_light_on: bool = False
    daylight_lux: float = 0.0
    artificial_lux: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Converts sensor reading to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------
# ROOM SENSOR MODEL
# ---------------------------------------------------------------------

class RoomSensorModel:
    """
    Simulates physically coupled environmental sensor dynamics for an individual room.
    Maintains persistent state between 5-minute simulation steps.
    """

    def __init__(
        self,
        room_id: str,
        room_cfg: dict[str, Any],
        outdoor_co2_baseline: float = 400.0,
        seed: int = 42,
    ):
        self.room_id = room_id
        self.cfg = room_cfg
        self.outdoor_co2_baseline = outdoor_co2_baseline
        self.seed = seed
        self.rng = random.Random(seed)

        self.volume_m3 = float(room_cfg.get("volume_m3", 300.0))

        # CO2 parameters
        co2_cfg = room_cfg.get("co2", {})
        self.co2_ach = float(co2_cfg.get("air_exchange_rate_ach", 0.8))
        self.co2_gen_rate = float(co2_cfg.get("co2_gen_rate_ppm_m3_h_person", 18000.0))
        self.co2_noise_std = float(co2_cfg.get("sensor_noise_std", 1.5))
        self.co2_min = float(co2_cfg.get("min_bound", 400.0))
        self.co2_max = float(co2_cfg.get("max_bound", 2500.0))

        # Temperature parameters
        temp_cfg = room_cfg.get("temperature", {})
        self.temp_setpoint = float(temp_cfg.get("setpoint_celsius", 26.5))
        self.temp_env_k = float(temp_cfg.get("envelope_coupling", 0.018))
        self.temp_hvac_k = float(temp_cfg.get("hvac_coupling", 0.08))
        self.temp_occ_k = float(temp_cfg.get("occupant_heat_gain_deg", 0.015))
        self.temp_occ_w = float(temp_cfg.get("occupant_heat_gain_w", 100.0))
        self.thermal_capacitance_kj_k = float(temp_cfg.get("thermal_capacitance_kj_k", 3600.0))
        self.temp_noise_std = float(temp_cfg.get("sensor_noise_std", 0.03))
        self.temp_min = float(temp_cfg.get("min_bound", 20.0))
        self.temp_max = float(temp_cfg.get("max_bound", 36.0))

        # Humidity parameters
        rh_cfg = room_cfg.get("relative_humidity", {})
        self.rh_setpoint = float(rh_cfg.get("setpoint_percent", 70.0))
        self.rh_vent_k = float(rh_cfg.get("vent_coupling", 0.02))
        self.rh_hvac_k = float(rh_cfg.get("hvac_coupling", 0.06))
        self.rh_occ_k = float(rh_cfg.get("occupant_moisture_gain_pct", 0.09))
        self.rh_occ_g_h = float(rh_cfg.get("occupant_moisture_g_h", 50.0))
        self.rh_noise_std = float(rh_cfg.get("sensor_noise_std", 0.15))
        self.rh_min = float(rh_cfg.get("min_bound", 40.0))
        self.rh_max = float(rh_cfg.get("max_bound", 95.0))

        # Sound parameters
        snd_cfg = room_cfg.get("sound_pressure_level", {})
        self.sound_base = float(snd_cfg.get("base_level_dba", 50.0))
        self.sound_occ = float(snd_cfg.get("occupant_activity_dba", 54.0))
        self.sound_noise_std = float(snd_cfg.get("sensor_noise_std", 0.25))
        self.sound_min = float(snd_cfg.get("min_bound", 40.0))
        self.sound_max = float(snd_cfg.get("max_bound", 85.0))

        # Illuminance & Lighting Policy parameters
        lux_cfg = room_cfg.get("illuminance", {})
        self.lighting_policy = str(lux_cfg.get("lighting_policy", "occupancy_vacancy"))
        self.vacancy_delay_minutes = float(lux_cfg.get("vacancy_delay_minutes", 10.0))
        self.lux_work = float(lux_cfg.get("artificial_work_lux", 150.0))
        self.lux_night = float(lux_cfg.get("artificial_night_lux", 0.0))
        self.lux_daylight_factor = float(lux_cfg.get("window_daylight_factor", 0.01))
        self.lux_noise_std = float(lux_cfg.get("sensor_noise_std", 1.0))
        self.lux_min = float(lux_cfg.get("min_bound", 0.0))
        self.lux_max = float(lux_cfg.get("max_bound", 12000.0))

        # Persistent physical states
        self.occupants: int = 0
        self.vacancy_timer_minutes: float = self.vacancy_delay_minutes  # Initially vacant (lights off)
        self.manual_light_override: Optional[bool] = None
        self.is_light_on: bool = False
        self.daylight_lux: float = 0.0
        self.artificial_lux: float = 0.0
        self.current_lux: float = self.lux_night

        self.current_sound: float = self.sound_base
        self.current_temp: float = self.temp_setpoint
        self.current_moisture_g_m3: float = vapor_density_from_rh(self.rh_setpoint, self.temp_setpoint)
        self.current_rh: float = self.rh_setpoint
        self.current_co2: float = self.outdoor_co2_baseline + 15.0

    def set_manual_lighting(self, state: Optional[bool]):
        """
        Overrides lighting policy:
        - True: Forced ON
        - False: Forced OFF
        - None: Automatic (follows lighting policy)
        """
        self.manual_light_override = state

    def _eval_illuminance(
        self,
        dt: datetime,
        occupants: int,
        solar_lux: float,
    ) -> tuple[float, float, float, bool]:
        """
        Evaluates physical illuminance field:
        Returns (total_lux, daylight_lux, artificial_lux, is_light_on).
        """
        # 1. Natural daylight window contribution
        daylight = max(0.0, solar_lux * self.lux_daylight_factor)

        # 2. Artificial lighting status based on policy and manual override
        if self.manual_light_override is not None:
            light_on = bool(self.manual_light_override)
        elif self.lighting_policy == "scheduled":
            hour_fraction = dt.hour + (dt.minute / 60.0)
            light_on = (8.0 <= hour_fraction < 18.0)
        else:
            # Default "occupancy_vacancy" policy:
            if occupants > 0:
                light_on = True
            else:
                # Room is empty; stays ON during vacancy turn-off delay
                light_on = (self.vacancy_timer_minutes < self.vacancy_delay_minutes)

        artificial = self.lux_work if light_on else self.lux_night

        noise = self.rng.gauss(0.0, self.lux_noise_std)
        total_lux = max(self.lux_min, min(self.lux_max, artificial + daylight + noise))

        return total_lux, daylight, artificial, light_on

    def _eval_sound_pressure(self, occupants: int) -> float:
        """
        Evaluates logarithmic acoustic decibel power summation:
        L_p = 10 * log10(10^(L_base/10) + N * 10^(L_activity/10)) + noise.
        """
        linear_power = 10.0 ** (self.sound_base / 10.0)
        if occupants > 0:
            linear_power += occupants * (10.0 ** (self.sound_occ / 10.0))
        calc_sound = 10.0 * math.log10(max(1.0, linear_power))
        noise = self.rng.gauss(0.0, self.sound_noise_std)
        return max(self.sound_min, min(self.sound_max, calc_sound + noise))

    def update_immediate(
        self,
        dt: datetime,
        occupants: int,
        outdoor_temp: float,
        outdoor_rh: float,
        solar_lux: float,
    ) -> RoomSensorReading:
        """
        Immediately updates instantaneous algebraic sensor variables (sound pressure level,
        lighting) following occupant movement, entry, or exit without advancing simulation time.
        Continuous dynamic variables (temperature, relative humidity, CO2) remain unchanged.
        """
        self.occupants = occupants

        # If occupied now, reset vacancy timer
        if occupants > 0:
            self.vacancy_timer_minutes = 0.0

        # Recalculate immediate sound pressure level
        self.current_sound = self._eval_sound_pressure(occupants)

        # Recalculate immediate illuminance field
        total_lux, daylight, artificial, light_on = self._eval_illuminance(dt, occupants, solar_lux)
        self.current_lux = total_lux
        self.daylight_lux = daylight
        self.artificial_lux = artificial
        self.is_light_on = light_on

        return RoomSensorReading(
            timestamp=dt.isoformat(),
            room_id=self.room_id,
            illuminance=round(self.current_lux, 2),
            sound_pressure_level=round(self.current_sound, 2),
            indoor_temperature=round(self.current_temp, 2),
            dry_bulb_temperature=round(outdoor_temp, 2),
            indoor_relative_humidity=round(self.current_rh, 2),
            outdoor_relative_humidity=round(outdoor_rh, 2),
            indoor_co2=round(self.current_co2, 2),
            is_light_on=self.is_light_on,
            daylight_lux=round(self.daylight_lux, 2),
            artificial_lux=round(self.artificial_lux, 2),
        )

    def step(
        self,
        dt: datetime,
        occupants: int,
        outdoor_temp: float,
        outdoor_rh: float,
        solar_lux: float,
        cadence_minutes: float = 5.0,
        hvac_state: Optional[Any] = None,
    ) -> RoomSensorReading:
        """
        Advances the room environmental physical equations by one logical time step (default: 5 minutes)
        and returns the synchronized sensor reading.
        Coupled with authoritative HVAC thermal cooling and setpoints when hvac_state is provided.
        """
        self.occupants = occupants
        dt_hr = cadence_minutes / 60.0

        # Update vacancy timer
        if occupants == 0:
            self.vacancy_timer_minutes += cadence_minutes
        else:
            self.vacancy_timer_minutes = 0.0

        # -------------------------------------------------------------
        # 1. CO2 Mass-Balance Dynamics (Exact Analytical Solution)
        # -------------------------------------------------------------
        # C_inf = C_out + (G * N) / (ACH * V)
        # C(t + dt) = C_inf + (C(t) - C_inf) * exp(-ACH * dt)
        if self.co2_ach > 0.001:
            steady_state_co2 = self.outdoor_co2_baseline + (self.co2_gen_rate * occupants) / (self.co2_ach * self.volume_m3)
            decay_factor = math.exp(-self.co2_ach * dt_hr)
            next_co2 = steady_state_co2 + (self.current_co2 - steady_state_co2) * decay_factor
        else:
            # Low ventilation fallback
            gen_ppm = (self.co2_gen_rate * occupants * dt_hr) / self.volume_m3
            next_co2 = self.current_co2 + gen_ppm

        co2_noise = self.rng.gauss(0.0, self.co2_noise_std)
        self.current_co2 = max(self.co2_min, min(self.co2_max, next_co2 + co2_noise))

        # -------------------------------------------------------------
        # 2. Indoor Air Temperature (Thermal Mass & Sensible Balance)
        # -------------------------------------------------------------
        # Temperature evolves through thermal mass capacitance:
        # Sensible loads: envelope transfer, HVAC regulation, and occupant metabolic dissipation.
        step_factor = cadence_minutes / 5.0
        d_temp_env = self.temp_env_k * (outdoor_temp - self.current_temp)

        if hvac_state is not None:
            op_state = str(getattr(hvac_state, "operating_state", "COOLING"))
            hvac_mode = str(getattr(hvac_state, "hvac_mode", "NORMAL"))

            if op_state == "OFF" or hvac_mode == "OFF":
                d_temp_hvac = 0.0  # HVAC disabled: zero cooling extraction
            else:
                d_temp_hvac = self.temp_hvac_k * (self.temp_setpoint - self.current_temp)
        else:
            d_temp_hvac = self.temp_hvac_k * (self.temp_setpoint - self.current_temp)

        d_temp_occ = self.temp_occ_k * occupants
        d_temp_vent = 0.005 * self.co2_ach * (outdoor_temp - self.current_temp)
        d_temp = (d_temp_env + d_temp_hvac + d_temp_occ + d_temp_vent) * step_factor

        temp_noise = self.rng.gauss(0.0, self.temp_noise_std)
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + d_temp + temp_noise))

        # -------------------------------------------------------------
        # 3. Indoor Relative Humidity (Psychrometric Moisture Balance)
        # -------------------------------------------------------------
        # Absolute moisture mass concentration (g/m^3) is tracked dynamically,
        # then relative humidity is derived from saturation vapor pressure at current_temp.
        c_v_out = vapor_density_from_rh(outdoor_rh, outdoor_temp)
        c_v_set = vapor_density_from_rh(self.rh_setpoint, self.temp_setpoint)

        # Occupant moisture emission: ~50 g/h/person
        moisture_gain_g_m3 = (self.rh_occ_g_h * occupants * dt_hr) / self.volume_m3
        d_cv_vent = self.rh_vent_k * (c_v_out - self.current_moisture_g_m3) * step_factor

        if hvac_state is not None and (getattr(hvac_state, "operating_state", "") == "OFF" or getattr(hvac_state, "hvac_mode", "") == "OFF"):
            d_cv_hvac = 0.0
        else:
            d_cv_hvac = self.rh_hvac_k * (c_v_set - self.current_moisture_g_m3) * step_factor

        self.current_moisture_g_m3 = max(
            1.0,
            self.current_moisture_g_m3 + moisture_gain_g_m3 + d_cv_vent + d_cv_hvac
        )

        # Derive relative humidity at new current_temp
        rh_derived = rh_from_vapor_density(self.current_moisture_g_m3, self.current_temp)
        rh_noise = self.rng.gauss(0.0, self.rh_noise_std)
        self.current_rh = max(self.rh_min, min(self.rh_max, rh_derived + rh_noise))

        # -------------------------------------------------------------
        # 4. Sound Pressure Level (Logarithmic Acoustic Addition)
        # -------------------------------------------------------------
        self.current_sound = self._eval_sound_pressure(occupants)

        # -------------------------------------------------------------
        # 5. Illuminance (Lighting Policy & Daylight Envelope)
        # -------------------------------------------------------------
        total_lux, daylight, artificial, light_on = self._eval_illuminance(dt, occupants, solar_lux)
        self.current_lux = total_lux
        self.daylight_lux = daylight
        self.artificial_lux = artificial
        self.is_light_on = light_on

        return RoomSensorReading(
            timestamp=dt.isoformat(),
            room_id=self.room_id,
            illuminance=round(self.current_lux, 2),
            sound_pressure_level=round(self.current_sound, 2),
            indoor_temperature=round(self.current_temp, 2),
            dry_bulb_temperature=round(outdoor_temp, 2),
            indoor_relative_humidity=round(self.current_rh, 2),
            outdoor_relative_humidity=round(outdoor_rh, 2),
            indoor_co2=round(self.current_co2, 2),
            is_light_on=self.is_light_on,
            daylight_lux=round(self.daylight_lux, 2),
            artificial_lux=round(self.artificial_lux, 2),
        )

    def reset(self, new_seed: Optional[int] = None):
        """Resets the room sensor state and isolated RNG."""
        if new_seed is not None:
            self.seed = new_seed
        self.rng = random.Random(self.seed)
        self.occupants = 0
        self.vacancy_timer_minutes = self.vacancy_delay_minutes
        self.manual_light_override = None
        self.is_light_on = False
        self.daylight_lux = 0.0
        self.artificial_lux = 0.0
        self.current_lux = self.lux_night
        self.current_sound = self.sound_base
        self.current_temp = self.temp_setpoint
        self.current_moisture_g_m3 = vapor_density_from_rh(self.rh_setpoint, self.temp_setpoint)
        self.current_rh = self.rh_setpoint
        self.current_co2 = self.outdoor_co2_baseline + 15.0


# ---------------------------------------------------------------------
# VIRTUAL SENSOR MANAGER
# ---------------------------------------------------------------------

class VirtualSensorManager:
    """
    Coordinates multi-room virtual environmental sensor simulation,
    synchronized with the building simulation logical clock.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.seed = seed
        self.config = config or load_sensor_config()
        self.cadence_minutes = float(self.config.get("cadence_minutes", 5))

        # Weather engine
        self.weather = OutdoorWeather(self.config, seed=self.seed)

        # Room models
        rooms_cfg = self.config.get("rooms", {})
        baseline_co2 = self.weather.outdoor_co2_baseline

        self.room_models: dict[str, RoomSensorModel] = {}
        for room_id in EXPECTED_ROOM_IDS:
            room_cfg = rooms_cfg.get(room_id, {})
            # Derive isolated deterministic seeds per room model
            room_seed = (self.seed * 37 + hash(room_id)) % (2**31 - 1)
            self.room_models[room_id] = RoomSensorModel(
                room_id=room_id,
                room_cfg=room_cfg,
                outdoor_co2_baseline=baseline_co2,
                seed=room_seed,
            )

        # Storage for historical step readings
        self.history: list[RoomSensorReading] = []
        self.latest_readings: dict[str, RoomSensorReading] = {}

    def set_manual_lighting(self, room_id: str, state: Optional[bool]):
        """Sets manual lighting state for a specific room."""
        if room_id in self.room_models:
            self.room_models[room_id].set_manual_lighting(state)

    def get_sensor_metadata(self, room_id: str) -> dict[str, Any]:
        """Returns physical state and policy diagnostic metadata for a room."""
        if room_id not in self.room_models:
            return {}
        m = self.room_models[room_id]
        return {
            "room_id": room_id,
            "occupants": m.occupants,
            "lighting_policy": m.lighting_policy,
            "is_light_on": m.is_light_on,
            "manual_override": m.manual_light_override,
            "vacancy_timer_minutes": round(m.vacancy_timer_minutes, 1),
            "vacancy_delay_minutes": m.vacancy_delay_minutes,
            "daylight_lux": round(m.daylight_lux, 2),
            "artificial_lux": round(m.artificial_lux, 2),
            "total_lux": round(m.current_lux, 2),
            "sound_pressure_level_dba": round(m.current_sound, 2),
            "base_sound_dba": m.sound_base,
            "activity_sound_dba": m.sound_occ,
            "indoor_temp_c": round(m.current_temp, 2),
            "temp_setpoint_c": m.temp_setpoint,
            "moisture_density_g_m3": round(m.current_moisture_g_m3, 2),
            "indoor_rh_percent": round(m.current_rh, 2),
            "rh_setpoint_percent": m.rh_setpoint,
            "indoor_co2_ppm": round(m.current_co2, 2),
            "volume_m3": m.volume_m3,
            "air_exchange_rate_ach": m.co2_ach,
        }

    def step(
        self,
        current_time: datetime,
        building: Building,
        hvac_states: Optional[dict[str, Any]] = None,
    ) -> dict[str, RoomSensorReading]:
        """
        Advances all 5 rooms by one 5-minute timestep at current_time.
        Reads ground-truth occupancy from building rooms, computes new physical
        state, and logs the synchronized readings. Coupled with HVAC thermal state when provided.
        """
        outdoor_temp = self.weather.get_dry_bulb_temperature(current_time)
        outdoor_rh = self.weather.get_outdoor_relative_humidity(current_time)
        solar_lux = self.weather.get_solar_daylight(current_time)

        step_readings = {}
        for room_id in EXPECTED_ROOM_IDS:
            room = building.get_room(room_id)
            occupants = room.occupant_count

            model = self.room_models[room_id]
            hvac_st = hvac_states.get(room_id) if hvac_states else None
            reading = model.step(
                dt=current_time,
                occupants=occupants,
                outdoor_temp=outdoor_temp,
                outdoor_rh=outdoor_rh,
                solar_lux=solar_lux,
                cadence_minutes=self.cadence_minutes,
                hvac_state=hvac_st,
            )
            step_readings[room_id] = reading
            self.history.append(reading)

        self.latest_readings = step_readings
        return step_readings

    def refresh_immediate_sensors(self, current_time: datetime, building: Building) -> dict[str, RoomSensorReading]:
        """
        Immediately updates algebraic sensor values (sound pressure, lighting)
        when occupants enter, leave, or move between rooms between 5-minute clock steps,
        preserving continuous dynamic variables (temp, CO2, humidity).
        """
        if self.latest_readings:
            sample_r = self.latest_readings.get(EXPECTED_ROOM_IDS[0])
            if sample_r and sample_r.timestamp == current_time.isoformat():
                outdoor_temp = sample_r.dry_bulb_temperature
                outdoor_rh = sample_r.outdoor_relative_humidity
                solar_lux = self.weather.get_solar_daylight(current_time)
            else:
                outdoor_temp = self.weather.get_dry_bulb_temperature(current_time)
                outdoor_rh = self.weather.get_outdoor_relative_humidity(current_time)
                solar_lux = self.weather.get_solar_daylight(current_time)
        else:
            outdoor_temp = self.weather.get_dry_bulb_temperature(current_time)
            outdoor_rh = self.weather.get_outdoor_relative_humidity(current_time)
            solar_lux = self.weather.get_solar_daylight(current_time)

        for room_id in EXPECTED_ROOM_IDS:
            room = building.get_room(room_id)
            model = self.room_models[room_id]
            reading = model.update_immediate(
                dt=current_time,
                occupants=room.occupant_count,
                outdoor_temp=outdoor_temp,
                outdoor_rh=outdoor_rh,
                solar_lux=solar_lux,
            )
            self.latest_readings[room_id] = reading
            # Update matching entry in history if present
            if self.history:
                for idx in range(len(self.history) - 1, max(-1, len(self.history) - 1 - len(EXPECTED_ROOM_IDS)), -1):
                    if self.history[idx].room_id == room_id and self.history[idx].timestamp == reading.timestamp:
                        self.history[idx] = reading
                        break
        return dict(self.latest_readings)

    def get_latest_readings(self) -> dict[str, RoomSensorReading]:
        """Returns the most recent reading for each of the 5 rooms."""
        return dict(self.latest_readings)

    def to_dataframe(self, include_metadata: bool = False) -> pd.DataFrame:
        """Converts accumulated sensor history into a pandas DataFrame."""
        standard_cols = [
            "timestamp", "room_id", "illuminance", "sound_pressure_level",
            "indoor_temperature", "dry_bulb_temperature",
            "indoor_relative_humidity", "outdoor_relative_humidity",
            "indoor_co2"
        ]
        if not self.history:
            return pd.DataFrame(columns=standard_cols)
        data = [asdict(r) for r in self.history]
        df = pd.DataFrame(data)
        if not include_metadata:
            df = df[[c for c in standard_cols if c in df.columns]]
        return df

    def export_csv(self, filepath: str = "artifacts/virtual_sensor_readings.csv"):
        """Exports full recorded sensor history to CSV."""
        df = self.to_dataframe()
        os_dir = os.path.dirname(filepath)
        if os_dir:
            os.makedirs(os_dir, exist_ok=True)
        df.to_csv(filepath, index=False)

    def reset(self, new_seed: Optional[int] = None):
        """Resets the sensor manager and all room sub-models."""
        if new_seed is not None:
            self.seed = new_seed
        self.weather.reset(self.seed)
        for room_id, model in self.room_models.items():
            room_seed = (self.seed * 37 + hash(room_id)) % (2**31 - 1)
            model.reset(room_seed)
        self.history.clear()
        self.latest_readings.clear()

    def clear_history(self):
        """Clears logged sensor history."""
        self.history.clear()

    def set_manual_lighting(self, room_id: str, state: Optional[bool]):
        """Sets manual lighting override for a room (True=ON, False=OFF, None=Auto)."""
        if room_id in self.room_models:
            self.room_models[room_id].set_manual_lighting(state)

    def get_sensor_metadata(self, room_id: str) -> dict[str, Any]:
        """Returns physical state parameters, policies, and diagnostic metadata for a room."""
        if room_id not in self.room_models:
            return {}
        model = self.room_models[room_id]
        return {
            "room_id": room_id,
            "lighting_policy": model.lighting_policy,
            "manual_light_override": model.manual_light_override,
            "is_light_on": model.is_light_on,
            "daylight_lux": model.daylight_lux,
            "artificial_lux": model.artificial_lux,
            "vacancy_timer_minutes": model.vacancy_timer_minutes,
            "vacancy_delay_minutes": model.vacancy_delay_minutes,
            "volume_m3": model.volume_m3,
            "co2_ach": model.co2_ach,
            "thermal_capacitance_kj_k": model.thermal_capacitance_kj_k,
            "moisture_density_g_m3": model.current_moisture_g_m3,
            "sound_base_dba": model.sound_base,
            "sound_occ_dba": model.sound_occ,
        }
