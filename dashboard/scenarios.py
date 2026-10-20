"""
Simulation Scenarios & Demonstration Presets
============================================
Defines documented, reproducible operational scenarios for smart-building demonstrations:
1. Normal Workday (Default 40 occupants, standard schedules, 24.5°C setpoints)
2. Low Occupancy (15 occupants, light partial office usage)
3. Peak Occupancy Surge (80 occupants, heavy thermal gain and high CO₂)
4. Empty Building (0 occupants, tests 10-minute vacancy timeouts and eco setbacks)
5. Energy-Saving Policy (Eco mode setpoints 26.5°C across all rooms, daylight harvesting)
6. Summer Heatwave Stress Test (Elevated ambient dry-bulb temperature 35.0°C)

Provides scenario metadata, preview generation, and deterministic state application.
"""

from typing import Dict, Any, Optional
import copy

from simulation.config import EXPECTED_ROOM_IDS
from simulation.sensors import load_sensor_config
from simulation.energy import load_energy_config
from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline


SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "normal": {
        "name": "Normal Workday",
        "description": "Standard business day operation with nominal population across offices and common spaces.",
        "occupants": 40,
        "seed": 42,
        "target_setpoint": 24.5,
        "outdoor_mean_temp": 28.03,
        "hvac_mode": None,  # AUTO
        "expected_power_kw": "10.0 - 15.0 kW",
        "expected_co2_ppm": "450 - 750 ppm",
        "impact_summary": "Balanced thermal comfort and nominal energy consumption with occupancy-driven HVAC modulation.",
    },
    "low_occupancy": {
        "name": "Low Occupancy (Off-Peak)",
        "description": "Light building activity (e.g. early morning or evening shift). Many rooms remain vacant.",
        "occupants": 15,
        "seed": 101,
        "target_setpoint": 24.5,
        "outdoor_mean_temp": 28.03,
        "hvac_mode": None,
        "expected_power_kw": "4.0 - 7.5 kW",
        "expected_co2_ppm": "400 - 550 ppm",
        "impact_summary": "Unoccupied rooms trigger 10-minute vacancy setbacks, dropping lighting to 0W and HVAC to standby.",
    },
    "peak_surge": {
        "name": "Peak Occupancy Surge",
        "description": "Heavy building loading approaching full capacity. Large meetings and congested entrance corridor.",
        "occupants": 80,
        "seed": 202,
        "target_setpoint": 24.0,
        "outdoor_mean_temp": 29.5,
        "hvac_mode": None,
        "expected_power_kw": "18.0 - 26.0 kW",
        "expected_co2_ppm": "800 - 1400 ppm",
        "impact_summary": "High internal sensible heat gains (+100 W/person) trigger HIGH_DEMAND cooling and rapid CO₂ accumulation.",
    },
    "empty_building": {
        "name": "Empty Building (Weekend / Night)",
        "description": "Facility completely vacant. Verifies automated vacancy timeouts, eco mode transitions, and standby base power.",
        "occupants": 0,
        "seed": 303,
        "target_setpoint": 28.0,
        "outdoor_mean_temp": 28.03,
        "hvac_mode": "ECO",
        "expected_power_kw": "1.0 - 2.5 kW",
        "expected_co2_ppm": "400 ppm (ambient baseline)",
        "impact_summary": "Zero occupant metabolic load. After 10 min, all HVAC systems drop to ECO (28°C) and lights turn off completely.",
    },
    "energy_saving": {
        "name": "Aggressive Energy Saving",
        "description": "Sustainability-focused policy enforcing 26.5°C comfort setpoints and strict daylight-harvesting.",
        "occupants": 40,
        "seed": 404,
        "target_setpoint": 26.5,
        "outdoor_mean_temp": 28.03,
        "hvac_mode": "ECO",
        "expected_power_kw": "6.0 - 9.0 kW",
        "expected_co2_ppm": "500 - 800 ppm",
        "impact_summary": "Reduces building electrical demand by ~35-45% compared to the unmodulated baseline policy.",
    },
    "heatwave": {
        "name": "Summer Heatwave Stress Test",
        "description": "Extreme exterior meteorological conditions (35.0°C outdoor ambient dry-bulb temperature).",
        "occupants": 40,
        "seed": 505,
        "target_setpoint": 24.0,
        "outdoor_mean_temp": 35.0,
        "hvac_mode": None,
        "expected_power_kw": "20.0 - 28.0 kW",
        "expected_co2_ppm": "500 - 800 ppm",
        "impact_summary": "High envelope thermal conduction increases cooling demand across all rooms, testing HVAC chiller capacity.",
    },
}


def get_scenario_metadata(scenario_key: str) -> dict[str, Any]:
    """Returns metadata dictionary for a specific scenario preset."""
    return SCENARIO_PRESETS.get(scenario_key, SCENARIO_PRESETS["normal"])


def apply_scenario(scenario_key: str, custom_seed: Optional[int] = None) -> DigitalTwinPipeline:
    """
    Deterministically instantiates a new simulation pipeline configured with the preset parameters.
    Returns the newly instantiated pipeline.
    """
    preset = get_scenario_metadata(scenario_key)
    seed = custom_seed if custom_seed is not None else preset["seed"]
    occupants = preset["occupants"]

    # Build sensor config with modified outdoor weather if heatwave
    sensor_cfg = copy.deepcopy(load_sensor_config())
    if "outdoor_mean_temp" in preset:
        outdoor_cfg = sensor_cfg.setdefault("outdoor_environment", {})
        temp_cfg = outdoor_cfg.setdefault("dry_bulb_temperature", {})
        temp_cfg["mean_celsius"] = float(preset["outdoor_mean_temp"])

    # Build energy config
    energy_cfg = copy.deepcopy(load_energy_config())

    engine = SimulationEngine(
        num_occupants=occupants,
        seed=seed,
        sensor_config=sensor_cfg,
        energy_config=energy_cfg,
    )

    # If scenario specifies HVAC override or custom setpoints, apply to rooms
    if preset.get("hvac_mode") == "ECO":
        for r_id in EXPECTED_ROOM_IDS:
            engine.set_room_hvac_mode(r_id, "ECO")
    elif preset.get("target_setpoint") is not None:
        sp = float(preset["target_setpoint"])
        for r_id in EXPECTED_ROOM_IDS:
            engine.set_room_hvac_setpoint(r_id, sp)

    pipeline = DigitalTwinPipeline(engine=engine, seed=seed)
    return pipeline

