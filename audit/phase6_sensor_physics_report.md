# Phase 6 — Realistic Occupancy-Responsive Sensor Physics Audit & Verification Report

**Status**: COMPLETED & VERIFIED  
**Date**: 2026-10-16  
**Scope**: Comprehensive physical reform of all seven virtual sensor models (Illuminance, Sound Pressure Level, Indoor Temperature, Dry-Bulb Temperature, Indoor Relative Humidity, Outdoor Relative Humidity, CO₂ Concentration), dual-timing update architecture (Immediate vs. Time-Dependent), configurable parameters, dashboard telemetry controls, and verification against 15 strict physical test suites.

---

## 1. Pre-Implementation Code Inspection & Defect Audit

An exhaustive inspection of the pre-Phase 6 sensor implementation ([`simulation/sensors.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/simulation/sensors.py), [`simulation/environment.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/simulation/environment.py), and [`config/sensor_parameters.json`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/config/sensor_parameters.json)) identified the following physical modeling defects:

### 1.1 Illuminance (`illuminance`, lux)
- **Defect**: Hardcoded time-of-day rule (`8.0 <= hour < 18.0`) turned lights on during work hours regardless of occupancy, and instantaneously shut lights off outside work hours the moment the last occupant exited.
- **Physical Reality**: Real office buildings utilize occupancy/vacancy lighting policies with configurable turn-off delays (e.g., 10 minutes) to prevent nuisance cycling.
- **Defect**: No distinction between artificial light and natural daylight. If a room has exterior windows, ambient daylight is present during daytime even if the room is completely empty ($N=0$).
- **Defect**: No support for manual lighting control (manual override) or room-specific lighting policies.

### 1.2 Sound Pressure Level (`sound_pressure_level`, dBA)
- **Defect**: Fixed per-occupant activity level ($54\text{ dBA}$) across all rooms.
- **Physical Reality**: Different functional spaces exhibit distinct acoustic emission profiles (e.g., quiet executive suite vs. conversational conference room vs. noisy cafeteria).
- **Positive Feature**: Employs logarithmic energy summation ($10 \log_{10}(10^{L_{\text{base}}/10} + N \cdot 10^{L_{\text{occ}}/10})$), which must be preserved and expanded with per-room activity tuning.

### 1.3 Indoor Temperature (`indoor_temperature`, °C)
- **Defect**: Occupant heat gain was modeled via an empirical direct increment coefficient (`d_temp_occ = temp_occ_k * occupants`) rather than metabolic sensible heat emission ($\dot{Q}_{\text{occ}} = N \cdot q_{\text{met}}$, $\approx 100\text{ W/person}$) balanced against room thermal capacitance ($C_{\text{th}}$).
- **Physical Reality**: Thermal mass of air and building materials creates substantial physical inertia; temperatures evolve gradually over time steps rather than reacting instantly.

### 1.4 Indoor Relative Humidity (`indoor_relative_humidity`, %)
- **Critical Physical Defect**: Relative humidity was treated as a directly additive quantity (`d_rh_occ = rh_occ_k * occupants`).
- **Thermodynamic Law**: Humans do not emit "relative humidity percent"; they emit water vapor mass ($g_{\text{moisture}} \approx 40\text{--}70\text{ g/h/person}$). Relative humidity is a derived thermodynamic state variable ($RH = p_v / p_{\text{sat}}(T) \times 100\%$).
- **Non-Intuitive Directional Couplings**: As temperature increases, saturation vapor pressure $p_{\text{sat}}(T)$ grows exponentially. If occupant heat load increases room temperature significantly, $RH$ can actually **decrease** even while moisture mass is increasing. The legacy model was completely unable to capture this true psychrometric phenomenon.

### 1.5 Outdoor Environmental Variables (`dry_bulb_temperature`, `outdoor_relative_humidity`)
- **Requirement**: Must remain strictly independent of indoor occupancy events (entry, exit, movement). In earlier iterations, repeated stochastic noise sampling between ticks could cause minor flickering; this is now strictly guarded.

### 1.6 Carbon Dioxide (`indoor_co2`, ppm)
- **Defect**: Forward Euler discretization can exhibit stiffness or sensitivity at higher ventilation air changes per hour (ACH).
- **Physical Reality**: Analytical exponential solution ($C(t + \Delta t) = C_\infty + (C(t) - C_\infty) e^{-\text{ACH} \cdot \Delta t}$) ensures exact stability and smooth asymptotic convergence towards outdoor background levels ($400\text{ ppm}$) when vacant.

---

## 2. Sensor Dependency & Physics Response Matrix

| Sensor Variable | Engineering Unit | Governing Physical Equation | Immediate vs. Dynamic | Response to Occupant Entry ($N \uparrow$) | Response to Occupant Exit ($N \downarrow$) | Key Parameters & Assumptions |
| :--- | :---: | :--- | :---: | :--- | :--- | :--- |
| **Illuminance** | lux | $E = E_{\text{artificial}} + I_{\text{solar}} f_{\text{window}} + \epsilon$ | **Immediate** | Turns on artificial lighting if policy enabled; daylight unaffected | Stays on until vacancy delay expires (e.g. 10m), then shuts off; daylight persists | $E_{\text{work}}$, $E_{\text{night}}$, $f_{\text{window}}$, $\tau_{\text{delay}}$, manual override state |
| **Sound Pressure Level** | dBA | $L_p = 10 \log_{10}\left(10^{\frac{L_{\text{base}}}{10}} + \sum 10^{\frac{L_{\text{occ}}}{10}}\right) + \epsilon$ | **Immediate** | Logarithmic acoustic power increase according to room activity | Immediate drop towards background noise floor $L_{\text{base}}$ | $L_{\text{base}}$, room-specific $L_{\text{activity}}$, noise std |
| **Indoor Temperature** | °C | $C_{\text{th}} \frac{dT}{dt} = N q_{\text{met}} + UA(T_{\text{out}} - T) + K_{\text{hvac}}(T_{\text{set}} - T)$ | **Dynamic** (Time Step) | Heat generation rate rises; temperature climbs gradually with thermal inertia | Heat generation drops; room retains heat and cools slowly via HVAC/envelope | $q_{\text{met}} \approx 100\text{ W}$, $C_{\text{th}}$, $k_{\text{env}}$, $k_{\text{hvac}}$, $T_{\text{set}}$ |
| **Dry-Bulb Temperature** | °C | Meteorological diurnal cycle | **Independent** | Strictly invariant under indoor occupancy | Strictly invariant under indoor occupancy | $T_{\text{mean}}$, diurnal amplitude, solar peak hour |
| **Indoor Relative Humidity** | % | $RH = \frac{p_v(C_v, T)}{p_{\text{sat}}(T)} \times 100\%$ | **Dynamic** (Time Step) | Moisture mass increases; $RH$ can rise or fall depending on concurrent $\Delta T$ | Moisture generation ceases; moisture relaxes toward ventilation equilibrium | $g_{\text{met}} \approx 50\text{ g/h}$, psychrometric vapor equation, ACH |
| **Outdoor Relative Humidity** | % | Atmospheric diurnal anti-correlation | **Independent** | Strictly invariant under indoor occupancy | Strictly invariant under indoor occupancy | $RH_{\text{mean}}$, diurnal amplitude, solar trough hour |
| **CO₂ Concentration** | ppm | $C(t+\Delta t) = C_\infty + (C(t) - C_\infty)e^{-\text{ACH}\Delta t}$ | **Dynamic** (Time Step) | Generation rate increases; concentration rises towards $C_\infty$ | Generation ceases; concentration decays exponentially to $400\text{ ppm}$ | $G_{\text{met}} \approx 18000\text{ ppm}\cdot\text{m}^3/\text{h}$, ACH, room volume $V$ |

---

## 3. Directional Non-Monotonicity & Psychrometric Thermodynamics

A simplistic assumption that "all sensor values increase when people enter and decrease when people leave" is physically invalid. The Phase 6 implementation explicitly models:

1. **Illuminance Decoupling**: Adding additional occupants to an already occupied room does **not** increase illuminance. Illuminance is determined by luminaire field flux and exterior solar daylight, not headcount.
2. **Relative Humidity Inversion**: When people enter a room, they emit sensible heat ($\approx 100\text{ W}$) and moisture ($\approx 50\text{ g/h}$). If the sensible heat causes temperature to rise, the saturation vapor pressure $p_{\text{sat}}(T)$ grows exponentially according to the Magnus-Tetens formulation:
   $$p_{\text{sat}}(T) = 0.61078 \exp\left(\frac{17.27 T}{T + 237.3}\right)\text{ kPa}$$
   If temperature rises from $25.0^\circ\text{C}$ to $26.5^\circ\text{C}$, $p_{\text{sat}}$ increases by $9.6\%$. If moisture mass concentration does not rise by more than $9.6\%$, the relative humidity will **decrease** even though water vapor was added to the air.
3. **Thermal and Chemical Inertia**: Temperature and CO₂ do not jump instantaneously when occupants enter or leave. An empty room retains its warm temperature and elevated CO₂ until physical air changes and thermal dissipation take effect over multiple 5-minute simulation steps.

---

## 4. Technical Implementation Architecture

### 4.1 Psychrometric Utilities (`simulation/sensors.py`)
- `saturation_vapor_pressure_kpa(temp_celsius)`: Computes $p_{\text{sat}}(T)$ via Magnus-Tetens formula.
- `vapor_density_from_rh(rh_percent, temp_celsius)`: Inverts psychrometric state to obtain absolute water vapor density $C_v$ in $\text{g/m}^3$ using the ideal gas law:
  $$\rho_v = \frac{p_v \cdot 1000}{R_v \cdot T_K}, \quad R_v = 0.4615\text{ kJ/(kg}\cdot\text{K)}$$
- `rh_from_vapor_density(vapor_density_g_m3, temp_celsius)`: Calculates relative humidity percentage from current vapor density and air temperature.

### 4.2 Acoustic Decibel Power Summation
The acoustic sound pressure level is computed via logarithmic power addition:
$$L_p = 10 \log_{10}\left(10^{L_{\text{base}}/10} + N \cdot 10^{L_{\text{activity}}/10}\right) + \epsilon$$
Doubling the occupant count adds approximately $+3\text{ dBA}$ to the acoustic contribution rather than multiplying the decibel value, accurately reflecting psychoacoustic physics.

### 4.3 CO₂ Analytical Solution
Rather than forward Euler integration, the discrete mass balance equation is solved analytically:
$$C_\infty = C_{\text{outdoor}} + \frac{G \cdot N}{\text{ACH} \cdot V}$$
$$C(t + \Delta t) = C_\infty + \left(C(t) - C_\infty\right) e^{-\text{ACH} \cdot \Delta t}$$
This guarantees unconditional numerical stability, zero oscillation, non-negativity, and asymptotic decay to $400\text{ ppm}$ when vacant.

### 4.4 Policy-Driven Lighting with Vacancy Delay & Manual Override
- Vacancy turn-off delay ($\tau_{\text{delay}} = 10\text{ minutes}$, or 2 simulation steps) keeps lights energized when an occupant temporarily exits.
- Natural daylight penetration ($I_{\text{solar}} \cdot f_{\text{window}}$) persists even when a room is vacant.
- Manual overrides (`Force ON`, `Force OFF`, `Auto`) take precedence when set by the user or operator.

### 4.5 Exited Occupant State Integrity
When an occupant exits the building via `exit_person()`, their schedule is marked `is_exited = True`. This prevents the occupant from immediately attempting to re-enter through the main entrance on subsequent simulation steps, ensuring that vacant rooms remain truly empty.

---

## 5. Verification Test Results (15/15 Phase 6 Tests, 82/82 Total Suite)

A comprehensive automated test suite ([`tests/test_phase6_sensor_physics.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/tests/test_phase6_sensor_physics.py)) was constructed covering all 15 required physical behaviors:

| # | Test Name | Target Physical Behavior | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_illuminance_occupancy_vacancy_delay_and_manual_override` | Policy activation, 10-min vacancy delay, daylight persistence, manual override | **PASSED** |
| 2 | `test_sound_pressure_logarithmic_summation_and_room_profiles` | Logarithmic decibel power summation, room-specific activity profiles | **PASSED** |
| 3 | `test_indoor_temperature_metabolic_heat_accumulation_and_thermal_persistence` | Lumped capacitance sensible heat accumulation, thermal persistence | **PASSED** |
| 4 | `test_outdoor_dry_bulb_temperature_independence` | Outdoor temperature invariance under indoor occupancy | **PASSED** |
| 5 | `test_indoor_relative_humidity_psychrometric_coupling` | Psychrometric vapor mass balance, Magnus-Tetens temperature coupling | **PASSED** |
| 6 | `test_outdoor_relative_humidity_independence` | Outdoor relative humidity invariance under indoor occupancy | **PASSED** |
| 7 | `test_co2_exact_analytical_accumulation_and_exponential_decay` | Analytical exponential solution, smooth asymptotic decay to 400 ppm | **PASSED** |
| 8 | `test_occupant_exit_removes_source_terms_preserves_continuous_state` | Immediate source reduction with continuous state preservation | **PASSED** |
| 9 | `test_room_to_room_movement_updates_both_rooms_independently` | Independent simultaneous update of source terms in source and destination | **PASSED** |
| 10 | `test_empty_rooms_relax_toward_equilibrium` | Slow asymptotic relaxation toward HVAC and atmospheric equilibrium | **PASSED** |
| 11 | `test_repeated_ui_reruns_do_not_advance_physics` | Invariance of physical dynamics under UI telemetry queries at same timestamp | **PASSED** |
| 12 | `test_physical_bounds_and_finite_values` | All 7 sensor variables strictly finite, bounded, and non-NaN | **PASSED** |
| 13 | `test_batch_arrivals_and_exits` | Large population increments scale source terms consistently | **PASSED** |
| 14 | `test_unified_physical_behavior_across_flow_modes` | Manual, scheduled, and interactive modes share unified physical model | **PASSED** |
| 15 | `test_pipeline_stgnn_inference_with_phase6_physics` | 12-step lookback rolling buffer & 15-minute STGNN forecasts valid | **PASSED** |

### Complete Test Suite Summary
```text
tests/test_dashboard.py ............... 7 passed
tests/test_interactive_population.py .. 16 passed
tests/test_manual_people_flow.py ...... 11 passed
tests/test_phase6_sensor_physics.py ... 15 passed
tests/test_pipeline.py ................ 9 passed
tests/test_sensors.py ................. 10 passed
tests/test_simulation.py .............. 14 passed
============================= 82 passed in 43.11s =============================
```

---

## 6. Streamlit Dashboard Telemetry Enhancements (`app.py`)

Tab 3 (**Live Room Sensors & Environmental Telemetry**) was updated with first-principles physical insights:
1. **Signal Classification Badges**: Explains the 3 classes of physical signals:
   - ⚡ **Immediate Response**: Illuminance (lux) and Sound Pressure Level (dBA).
   - ⏳ **Dynamic Integration (5-min ticks)**: Indoor Temperature (°C), Indoor RH (%), and Indoor CO₂ (ppm).
   - ⛅ **Outdoor Meteorological Drivers**: Dry-Bulb Temperature (°C) and Outdoor RH (%).
2. **Cross-Room Physical Snapshot Table**: Formatted table displaying live lighting states (`💡 ON` / `🌑 OFF`), sound, temperature, humidity, and CO₂ for all 5 rooms.
3. **Manual Lighting Controls**: Interactive radio button per room allowing the user to switch between `Auto (Policy: Vacancy Sensor)`, `Force ON (Manual)`, and `Force OFF (Manual)` with immediate visual and illuminance feedback.
4. **Physical Parameters & Diagnostic Telemetry Expander**: Displays internal physical variables per room, including:
   - Lamp state, natural daylight lux, artificial lux, vacancy countdown timer.
   - Base acoustic noise floor, occupant activity dBA level.
   - Room volume ($m^3$), ventilation rate (ACH), thermal capacitance ($kJ/K$), and absolute moisture vapor density ($g/m^3$).

---

## 7. Assumptions & Known Limitations

1. **Uniform Room Air Mixing (Lumped Parameter)**: Each room is treated as a well-mixed single zone with homogeneous air temperature and gas concentration. Micro-climates or thermal stratifications within a single room are not modeled.
2. **Door Openings & Inter-Zone Airflow**: Inter-room air exchange occurs primarily via HVAC ventilation and envelope infiltration rather than detailed computational fluid dynamic (CFD) door airflow modeling.
3. **Model Weights & Scalers Preserved**: In strict compliance with safety guidelines, model weights in `joint_lstm_gnn_all_5_rooms.keras` and scaler artifacts in `artifacts/scalers.joblib` were not altered or retrained.
