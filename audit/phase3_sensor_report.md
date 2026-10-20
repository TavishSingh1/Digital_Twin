# Phase 3.2 — Virtual Environmental Sensor Engine Verification & Audit Report

**Status**: VERIFIED & COMPLETE  
**Date**: 2026-10-15  
**Simulation Cadence**: 5-minute logical discrete steps  
**Target Architecture**: Joint Improved STGNN + LSTM (5 Synchronized Rooms)  
**Simulated Features**: 7 Environmental Features per Room (`COMMON_SENSOR_FEATURES`)  

---

## 1. Executive Summary

Phase 3.2 implements a physically motivated, modular, deterministic virtual environmental sensor engine for all five rooms in the digital twin building simulation. The engine runs synchronously with the 5-minute logical simulation clock, generating realistic environmental telemetry grounded in mass-balance, thermal capacitance, and acoustic principles.

- **Feature Compatibility**: 100% adherence to the verified feature order and physical units (`illuminance` [lux], `sound_pressure_level` [dBA], `indoor_temperature` [°C], `dry_bulb_temperature` [°C], `indoor_relative_humidity` [%], `outdoor_relative_humidity` [%], `indoor_co2` [ppm]).
- **Physical Dynamics**: Explicit state equations for CO₂ accumulation/ventilation, indoor thermal mass and HVAC setpoint recovery, acoustic decibel power summation, moisture dissipation, and diurnal solar radiation.
- **Time Synchronization**: Seamless integration into `SimulationEngine`, maintaining identical timestamp intervals (5-minute logical delta) across all five rooms without duplicates or missing values.
- **Verification**: 24/24 automated tests passed in `tests/test_sensors.py` and `tests/test_simulation.py`.
- **Export**: Full synchronized readings exported to [`artifacts/virtual_sensor_readings.csv`](file:///artifacts/virtual_sensor_readings.csv).

---

## 2. Mathematical Formulation & Physical Equations

### 2.1 Carbon Dioxide (CO₂) Dynamics
Indoor CO₂ follows a physical mass balance between occupant metabolic generation, room air volume, mechanical ventilation exchange, and outdoor ambient baseline:

$$\frac{dC}{dt} = \frac{G_{\text{co2}} \cdot N_{\text{occ}}}{V_{\text{room}}} - \text{ACH} \cdot (C - C_{\text{outdoor}})$$

- $G_{\text{co2}} = 18{,}000\text{ ppm}\cdot\text{m}^3/(\text{h}\cdot\text{person})$ ($0.005\text{ L/s/person}$ standard metabolic exhalation rate).
- $\text{ACH} \in [0.75, 1.10]\text{ h}^{-1}$ (calibrated room air change rate per hour).
- $V_{\text{room}}$: Room volume in $\text{m}^3$ calculated from floor dimensions $\times$ 3.0 m ceiling height.
- $C_{\text{outdoor}} = 400.0\text{ ppm}$ (atmospheric baseline concentration).

### 2.2 Indoor Air Temperature Dynamics
Indoor air temperature is governed by a lumped thermal capacitance model balancing building envelope conduction, HVAC setpoint feedback, and occupant sensible metabolic heat gain (~100 W/person):

$$\Delta T_{\text{indoor}} = \left[ k_{\text{env}} (T_{\text{outdoor}} - T_{\text{indoor}}) + k_{\text{hvac}} (T_{\text{setpoint}} - T_{\text{indoor}}) + k_{\text{occ}} \cdot N_{\text{occ}} \right] \cdot \Delta t$$

- $T_{\text{setpoint}}$: Calibrated to match historical indoor setpoints ($25.5^\circ\text{C}$ to $28.2^\circ\text{C}$).
- $k_{\text{env}} \approx 0.015 - 0.022\text{ step}^{-1}$ (gradual envelope heat transfer).
- $k_{\text{hvac}} \approx 0.07 - 0.08\text{ step}^{-1}$ (active HVAC cooling/heating response).
- $k_{\text{occ}} \approx 0.012 - 0.015^\circ\text{C/person}$ sensible heat dissipation.

### 2.3 Sound Pressure Level (dBA)
Acoustic noise follows logarithmic sound energy summation of baseline room equipment noise floor and conversational/movement occupant activity:

$$L_p = 10 \cdot \log_{10} \left( 10^{L_{\text{base}}/10} + N_{\text{occ}} \cdot 10^{L_{\text{activity}}/10} \right) + \epsilon_{\text{noise}}$$

- $L_{\text{base}}$: Room background acoustic floor ($48.0 - 62.0\text{ dBA}$, reflecting room-specific equipment).
- $L_{\text{activity}} = 53.0 - 56.0\text{ dBA}$ per occupant.

### 2.4 Relative Humidity (%)
Indoor relative humidity balances outdoor air infiltration, HVAC dehumidification setpoint recovery, and occupant latent moisture emission (~60 g/h/person):

$$\Delta RH_{\text{indoor}} = \left[ k_{\text{vent}} (RH_{\text{outdoor}} - RH_{\text{indoor}}) + k_{\text{hvac}} (RH_{\text{setpoint}} - RH_{\text{indoor}}) + k_{\text{moist}} \cdot N_{\text{occ}} \right] \cdot \Delta t$$

### 2.5 Illuminance (lux)
Illuminance combines scheduled artificial lighting and daylight penetration via windows:

$$E = E_{\text{artificial}}(t, N_{\text{occ}}) + W_{\text{window}} \cdot E_{\text{solar}}(t) + \epsilon_{\text{noise}}$$

- $E_{\text{artificial}}$: Active during work hours ($08:00 - 18:00$) or whenever $N_{\text{occ}} > 0$. Drops to 0 at night.
- $E_{\text{solar}}(t)$: Sinusoidal daylight curve between 06:00 and 18:30 (peaks at solar noon).
- $W_{\text{window}}$: Window daylight transmission factor (Room 5 cafeteria has the largest window aperture).

### 2.6 Outdoor Weather Model (`dry_bulb_temperature` & `outdoor_relative_humidity`)
Follows the meteorological Parton-Logan diurnal cycle:
- Minimum temperature at dawn (~05:30), maximum in early afternoon (~14:30), mean $28.03^\circ\text{C}$.
- Outdoor relative humidity is anti-correlated with ambient temperature (trough at 14:30 ~75%, peak at dawn ~97%, mean $87.0\%$).

---

## 3. Statistical Comparison: Simulated vs. Historical Distributions

The table below contrasts the 24-hour simulation statistics against the full historical datasets across all five rooms.

### 3.1 Room 1
| Feature | Sim Mean | Hist Mean | Sim 50% | Hist 50% | Sim Std | Hist Std | Sim [Min, Max] | Hist [Min, Max] |
|---|---|---|---|---|---|---|---|---|
| `illuminance` | 30.92 | 13.29 | 5.02 | 0.36 | 33.71 | 19.32 | [0.0, 81.2] | [0.0, 92.0] |
| `sound_pressure_level` | 52.45 | 49.05 | 48.32 | 48.42 | 6.17 | 3.37 | [47.0, 64.8] | [45.5, 74.9] |
| `indoor_temperature` | 27.49 | 26.84 | 27.41 | 26.92 | 0.58 | 0.98 | [26.6, 28.8] | [24.6, 29.2] |
| `dry_bulb_temperature` | 28.80 | 28.03 | 29.20 | 27.60 | 1.73 | 2.02 | [25.2, 31.0] | [22.6, 35.5] |
| `indoor_relative_humidity` | 80.94 | 75.16 | 80.81 | 76.68 | 2.96 | 6.94 | [75.5, 87.6] | [49.0, 87.8] |
| `outdoor_relative_humidity` | 87.79 | 86.92 | 88.46 | 88.40 | 7.71 | 11.61 | [75.1, 99.1] | [39.8, 100.0] |
| `indoor_co2` | 506.84 | 454.02 | 445.66 | 433.60 | 127.56 | 55.24 | [400.0, 815.1] | [400.0, 761.3] |

### 3.2 Room 2
| Feature | Sim Mean | Hist Mean | Sim 50% | Hist 50% | Sim Std | Hist Std | Sim [Min, Max] | Hist [Min, Max] |
|---|---|---|---|---|---|---|---|---|
| `illuminance` | 146.31 | 52.31 | 38.59 | 0.00 | 160.52 | 81.06 | [0.0, 386.5] | [0.0, 2015.5] |
| `sound_pressure_level` | 59.30 | 55.04 | 53.38 | 54.61 | 7.32 | 3.02 | [52.3, 69.7] | [48.5, 78.6] |
| `indoor_temperature` | 27.41 | 25.57 | 26.45 | 25.51 | 1.70 | 1.83 | [25.6, 30.3] | [20.4, 34.0] |
| `dry_bulb_temperature` | 28.80 | 28.03 | 29.20 | 27.60 | 1.73 | 2.02 | [25.2, 31.0] | [22.6, 34.9] |
| `indoor_relative_humidity` | 83.50 | 71.20 | 78.95 | 74.36 | 8.66 | 11.46 | [71.2, 95.0] | [45.4, 93.0] |
| `outdoor_relative_humidity` | 87.79 | 86.92 | 88.46 | 88.41 | 7.71 | 11.66 | [75.1, 99.1] | [40.3, 100.0] |
| `indoor_co2` | 946.33 | 467.42 | 565.01 | 440.12 | 600.38 | 81.07 | [400.0, 1959.1] | [400.0, 1489.4] |

### 3.3 Room 3
| Feature | Sim Mean | Hist Mean | Sim 50% | Hist 50% | Sim Std | Hist Std | Sim [Min, Max] | Hist [Min, Max] |
|---|---|---|---|---|---|---|---|---|
| `illuminance` | 86.07 | 51.83 | 7.93 | 27.42 | 98.24 | 54.04 | [0.0, 232.1] | [0.0, 206.1] |
| `sound_pressure_level` | 55.37 | 56.07 | 54.10 | 55.22 | 2.92 | 2.77 | [53.2, 66.0] | [51.6, 70.8] |
| `indoor_temperature` | 27.74 | 27.29 | 27.70 | 27.31 | 0.53 | 0.72 | [27.1, 29.2] | [25.6, 29.2] |
| `dry_bulb_temperature` | 28.80 | 28.03 | 29.20 | 27.60 | 1.73 | 2.02 | [25.2, 31.0] | [22.6, 35.5] |
| `indoor_relative_humidity` | 66.92 | 61.40 | 66.46 | 61.09 | 2.53 | 3.38 | [62.0, 74.2] | [53.1, 73.4] |
| `outdoor_relative_humidity` | 87.79 | 86.92 | 88.46 | 88.40 | 7.71 | 11.61 | [75.1, 99.1] | [39.8, 100.0] |
| `indoor_co2` | 480.73 | 483.02 | 408.84 | 471.87 | 150.47 | 62.21 | [400.0, 947.0] | [400.0, 836.7] |

### 3.4 Room 4
| Feature | Sim Mean | Hist Mean | Sim 50% | Hist 50% | Sim Std | Hist Std | Sim [Min, Max] | Hist [Min, Max] |
|---|---|---|---|---|---|---|---|---|
| `illuminance` | 169.55 | 125.95 | 37.12 | 118.83 | 187.20 | 101.84 | [0.0, 452.1] | [0.0, 1310.0] |
| `sound_pressure_level` | 63.44 | 62.17 | 62.24 | 62.17 | 1.74 | 0.78 | [61.4, 67.4] | [49.1, 75.4] |
| `indoor_temperature` | 28.20 | 27.41 | 27.80 | 27.47 | 0.91 | 0.48 | [27.0, 29.9] | [25.1, 28.6] |
| `dry_bulb_temperature` | 28.80 | 27.95 | 29.20 | 27.60 | 1.73 | 2.08 | [25.2, 31.0] | [22.6, 35.5] |
| `indoor_relative_humidity` | 81.12 | 73.08 | 79.51 | 73.87 | 5.03 | 7.76 | [73.3, 91.0] | [59.7, 88.8] |
| `outdoor_relative_humidity` | 87.79 | 87.28 | 88.46 | 89.10 | 7.71 | 11.69 | [75.1, 99.1] | [39.8, 100.0] |
| `indoor_co2` | 701.66 | 482.68 | 500.28 | 466.80 | 332.55 | 64.09 | [400.0, 1357.3] | [400.0, 765.3] |

### 3.5 Room 5
| Feature | Sim Mean | Hist Mean | Sim 50% | Hist 50% | Sim Std | Hist Std | Sim [Min, Max] | Hist [Min, Max] |
|---|---|---|---|---|---|---|---|---|
| `illuminance` | 387.16 | 323.91 | 65.57 | 124.27 | 445.75 | 503.74 | [0.0, 1128.6] | [0.0, 10698.8] |
| `sound_pressure_level` | 50.97 | 51.73 | 49.54 | 51.00 | 4.97 | 2.62 | [48.8, 69.3] | [44.7, 79.1] |
| `indoor_temperature` | 28.52 | 28.18 | 28.37 | 28.13 | 0.76 | 0.70 | [27.6, 31.0] | [26.7, 30.2] |
| `dry_bulb_temperature` | 28.80 | 27.95 | 29.20 | 27.60 | 1.73 | 2.08 | [25.2, 31.0] | [22.6, 35.5] |
| `indoor_relative_humidity` | 68.51 | 62.56 | 68.21 | 61.91 | 2.42 | 3.09 | [63.1, 76.9] | [55.2, 75.1] |
| `outdoor_relative_humidity` | 87.79 | 87.28 | 88.46 | 89.10 | 7.71 | 11.69 | [75.1, 99.1] | [39.8, 100.0] |
| `indoor_co2` | 484.51 | 453.98 | 402.21 | 446.37 | 208.40 | 34.96 | [400.0, 1297.8] | [400.0, 650.8] |

---

## 4. Verification Checklist & Test Results

All requirements specified for Phase 3.2 have been implemented and verified via automated testing:

| ID | Test Requirement | Test Case | Status | Notes |
|---|---|---|---|---|
| 1 | All 7 features generated per room | `test_all_seven_features_exist_per_room` | PASS | Illuminance, sound, indoor temp, dry bulb temp, indoor RH, outdoor RH, indoor CO2 |
| 2 | Physical bounds strictly respected | `test_physical_bounds_respected` | PASS | All 7 variables remain within safe, plausible physical intervals |
| 3 | Smooth temporal evolution | `test_smooth_temporal_evolution` | PASS | Continuous transitions without unnatural teleportation |
| 4 | CO2 responsiveness | `test_co2_responsiveness_to_occupants` | PASS | CO2 rises when occupied, decays exponentially to 400 ppm when vacant |
| 5 | Acoustic power addition | `test_sound_pressure_acoustic_addition` | PASS | Sound increases logarithmically with occupant density |
| 6 | Lighting schedules and daylight | `test_illuminance_schedule_and_daylight` | PASS | 0 lux at night; work hours illumination + solar daylight contribution |
| 7 | Diurnal weather anti-correlation | `test_outdoor_weather_diurnal_and_anticorrelation` | PASS | Peak temp at 14:30; RH reaches peak at dawn and trough in afternoon |
| 8 | Deterministic repeatability | `test_sensor_engine_determinism` | PASS | Identical seed yields exact bitwise-identical telemetry |
| 9 | Logical clock synchronization | `test_sensor_time_synchronization` | PASS | Exactly 5 records per 5-minute timestep; no duplicates or gaps |
| 10 | CSV export integrity | `test_sensor_csv_export` | PASS | Valid CSV schema, clean pandas parsing, no NaNs |
| 11 | Phase 3.1 regression safety | `test_simulation.py` (14 tests) | PASS | 100% pass rate; occupant movement and capacity invariants untouched |

---

## 5. Simulation Assumptions & Engineering Decisions

1. **Outdoor Dry-Bulb Temperature Interpretation**: In accordance with dataset analysis and building meteorological standards, `dry_bulb_temperature` represents outdoor weather station temperature. In historical data, Rooms 1, 2, and 3 share identical dry-bulb readings. We modeled a shared external weather envelope driving envelope heat exchange across all rooms.
2. **Acoustic Noise Floor Calibration**: Historical data revealed that Room 4 has an unusually high background sound floor (~62.2 dBA, compared to ~48-53 dBA in other rooms), likely due to server rack equipment or localized HVAC fan coil ducting. This was faithfully incorporated into `config/sensor_parameters.json`.
3. **Daylight Exposure Calibration**: Room 5 (Cafeteria/Lounge) in the historical dataset experiences extreme daylight exposure (up to 10,000+ lux). We configured a higher window daylight factor for Room 5 while maintaining interior office rooms at standard daylighting levels.
4. **Non-Intrusive Coupling**: Ground-truth actual occupancy counts drive the environmental physics, but ground-truth data remains isolated in `artifacts/ground_truth_occupancy.csv`. Sensor telemetry is logged separately in `artifacts/virtual_sensor_readings.csv`.

---

## 6. Next Steps

Phase 3.2 is complete and verified. Awaiting user approval before proceeding to **Phase 3.3** (End-to-End Simulation Pipeline: linking virtual sensors into the STGNN inference loop, 12-step lookback rolling window, and 15-minute predictive forecasting).