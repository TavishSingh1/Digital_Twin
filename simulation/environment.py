"""
Outdoor Environmental Weather Model
===================================
Models ambient weather dynamics (dry-bulb temperature, outdoor relative humidity,
and solar daylight irradiance) based on diurnal cycles and meteorological physics.

Physical Assumptions & Dataset Alignment:
- In the 5-room building dataset, 'dry_bulb_temperature' corresponds to outdoor
  weather station dry-bulb temperature (mean ~28.03 C, typical range 23-35 C).
- 'outdoor_relative_humidity' represents exterior atmospheric relative humidity
  (mean ~87%, anti-correlated with ambient temperature).
- Diurnal variations follow the standard Parton-Logan meteorological formulation:
  minimum temperature at dawn (~05:30), maximum in early afternoon (~14:30).
"""

from datetime import datetime
import math
import random
from typing import Optional, Any


class OutdoorWeather:
    """
    Deterministic diurnal outdoor weather model providing synchronized ambient
    temperature, relative humidity, and daylight conditions for the building envelope.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        seed: int = 42,
    ):
        cfg = config or {}
        weather_cfg = cfg.get("outdoor_environment", {})

        temp_cfg = weather_cfg.get("dry_bulb_temperature", {})
        self.temp_mean = float(temp_cfg.get("mean_celsius", 28.03))
        self.temp_amp = float(temp_cfg.get("diurnal_amplitude_celsius", 2.8))
        self.temp_peak_hour = float(temp_cfg.get("peak_hour", 14.5))
        self.temp_noise_std = float(temp_cfg.get("stochastic_noise_std", 0.08))
        self.temp_min_bound = float(temp_cfg.get("min_bound", 20.0))
        self.temp_max_bound = float(temp_cfg.get("max_bound", 38.0))

        rh_cfg = weather_cfg.get("outdoor_relative_humidity", {})
        self.rh_mean = float(rh_cfg.get("mean_percent", 87.0))
        self.rh_amp = float(rh_cfg.get("diurnal_amplitude_percent", 11.5))
        self.rh_noise_std = float(rh_cfg.get("stochastic_noise_std", 0.25))
        self.rh_min_bound = float(rh_cfg.get("min_bound", 35.0))
        self.rh_max_bound = float(rh_cfg.get("max_bound", 100.0))

        solar_cfg = weather_cfg.get("solar_daylight", {})
        self.sunrise_hour = float(solar_cfg.get("sunrise_hour", 6.0))
        self.sunset_hour = float(solar_cfg.get("sunset_hour", 18.5))
        self.peak_solar_lux = float(solar_cfg.get("peak_solar_lux", 15000.0))

        self.outdoor_co2_baseline = float(weather_cfg.get("outdoor_co2_baseline_ppm", 400.0))

        self.seed = seed
        self.rng = random.Random(self.seed)

    def get_dry_bulb_temperature(self, dt: datetime) -> float:
        """
        Calculates outdoor dry-bulb temperature (Celsius) at the given datetime.
        Uses asymmetric diurnal model (coolest at dawn ~05:30, warmest at ~14:30).
        """
        t_hour = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)

        t_min_hour = 5.5
        t_max_hour = 14.5
        t_min = self.temp_mean - self.temp_amp
        t_max = self.temp_mean + self.temp_amp

        if t_min_hour <= t_hour <= t_max_hour:
            # Daytime warming curve (9 hours)
            progress = (t_hour - t_min_hour) / (t_max_hour - t_min_hour)
            base_temp = t_min + (t_max - t_min) * math.sin(progress * (math.pi / 2.0))
        else:
            # Evening and night cooling curve (15 hours)
            t_cool = (t_hour - t_max_hour) % 24.0
            base_temp = t_min + (t_max - t_min) * math.cos((t_cool / 15.0) * (math.pi / 2.0))

        noise = self.rng.gauss(0.0, self.temp_noise_std)
        temp = base_temp + noise
        return max(self.temp_min_bound, min(self.temp_max_bound, round(temp, 2)))

    def get_outdoor_relative_humidity(self, dt: datetime) -> float:
        """
        Calculates outdoor atmospheric relative humidity (percent) at the given datetime.
        Physically anti-correlated with ambient dry-bulb temperature.
        """
        t_hour = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)

        t_min_hour = 5.5
        t_max_hour = 14.5

        if t_min_hour <= t_hour <= t_max_hour:
            # Daytime humidity drop as air warms
            progress = (t_hour - t_min_hour) / (t_max_hour - t_min_hour)
            factor = math.sin(progress * (math.pi / 2.0))
            base_rh = (self.rh_mean + self.rh_amp) - (2.0 * self.rh_amp * factor)
        else:
            # Nighttime humidity rise as air cools
            t_cool = (t_hour - t_max_hour) % 24.0
            factor = math.cos((t_cool / 15.0) * (math.pi / 2.0))
            base_rh = (self.rh_mean - self.rh_amp) + (2.0 * self.rh_amp * factor)

        noise = self.rng.gauss(0.0, self.rh_noise_std)
        rh = base_rh + noise
        return max(self.rh_min_bound, min(self.rh_max_bound, round(rh, 2)))

    def get_solar_daylight(self, dt: datetime) -> float:
        """
        Calculates exterior solar daylight illuminance (lux) at the given datetime.
        Zero at night, sinusoidal envelope between sunrise and sunset.
        """
        t_hour = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)
        if self.sunrise_hour <= t_hour <= self.sunset_hour:
            day_length = self.sunset_hour - self.sunrise_hour
            day_progress = (t_hour - self.sunrise_hour) / day_length
            solar_lux = self.peak_solar_lux * math.sin(day_progress * math.pi)
            return max(0.0, solar_lux)
        return 0.0

    def reset(self, new_seed: Optional[int] = None):
        """Resets the weather RNG for deterministic re-runs."""
        if new_seed is not None:
            self.seed = new_seed
        self.rng = random.Random(self.seed)

