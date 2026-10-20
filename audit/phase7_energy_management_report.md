# Phase 7 — Smart HVAC Management and Room-Wise Energy Consumption Audit Report

**Status**: COMPLETED & VERIFIED  
**Date**: 2026-10-16  
**Scope**: Integrated Smart HVAC and Energy Management module for the five-room virtual building occupancy digital twin. Includes physical cooling demand modeling, COP-separated electrical power calculations, daylight-harvesting lighting control, strict causal energy accumulation, baseline policy comparative evaluation, interactive Streamlit dashboard tab ("⚡ Smart HVAC & Energy"), and 20 dedicated verification tests.

---

## 1. Executive Summary

Phase 7 introduces physical HVAC and lighting energy management into the existing building occupancy digital twin without modifying or retraining the pre-trained STGNN model (`joint_lstm_gnn_all_5_rooms.keras`) or altering the 13-feature input contract `(batch, 12, 5, 13)`.

### Key Capabilities Delivered:
1. **Configurable Physical Energy Parameters** ([`config/energy_parameters.json`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/config/energy_parameters.json)): Room-specific cooling capacities ($8\text{--}14\text{ kW}$), electrical ratings ($2.5\text{--}4.375\text{ kW}$ at $\text{COP}=3.2$), daylight-harvesting lighting ratings ($240\text{--}420\text{ W}$), auxiliary base loads ($100\text{--}150\text{ W}$), comfort setpoints ($24.5^\circ\text{C}$ normal, $28.0^\circ\text{C}$ eco setback), and deadband thresholds ($0.5^\circ\text{C}$).
2. **Modular Energy Engine** ([`simulation/energy.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/simulation/energy.py)):
   - `HVACController`: Modulates cooling capacity and electrical power based on indoor/outdoor temperatures, occupant sensible heat emissions, deadband hysteresis, and 10-minute vacancy delays.
   - `LightingController`: Combines occupancy vacancy turn-off timers ($10\text{ min}$) with daylight harvesting that dims artificial LED power based on ambient daylight illuminance.
   - `BaselinePolicySimulator`: Concurrently runs a standard non-smart control policy (fixed $24.0^\circ\text{C}$ setpoint, unmodulated scheduled lighting) on the identical occupancy and weather trace to provide honest energy savings figures.
   - `EnergyManager`: Coordinates controllers for all five rooms, records instantaneous power, and integrates cumulative energy ($kWh$) strictly once per 5-minute logical simulation time step.
3. **Strict Idempotency Guarantee**: Immediate updates triggered by UI reruns, parameter adjustments, or interactive occupant arrivals/departures recalculate instantaneous power ($W$) with `advance_time=False`, guaranteeing zero double-counting of cumulative energy ($kWh$).
4. **Interactive Dashboard Tab ("⚡ Smart HVAC & Energy")** ([`app.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/app.py)): Building-wide power/energy KPIs, room-wise energy table, stacked power breakdown chart, cumulative energy trend chart, occupancy-versus-power scatter chart, smart vs. baseline comparative savings cards, and per-room manual HVAC/lighting override controls.
5. **Data Export** ([`artifacts/room_energy_consumption.csv`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/artifacts/room_energy_consumption.csv)): Added room-wise energy telemetry download and data preview in Tab 6.
6. **Full Verification Suite** ([`tests/test_energy_management.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/tests/test_energy_management.py)): 20 dedicated automated tests covering all physical, control, and mathematical invariants. Combined with existing tests, the test suite now comprises 102 passing tests (100% pass rate).

---

## 2. Mathematical Formulations & Governing Equations

### 2.1 HVAC Thermal Load & Electrical Power

#### A. Internal Occupant Sensible Heat Load ($\dot{Q}_{\text{occ}}$)
Each occupant emits metabolic sensible heat:
$$\dot{Q}_{\text{occ}} = N \cdot q_{\text{met}}, \quad q_{\text{met}} = 100\text{ W/person}$$

#### B. Building Envelope Transmission Heat Load ($\dot{Q}_{\text{trans}}$)
Heat transfer through the building envelope driven by outdoor air temperature:
$$\dot{Q}_{\text{trans}} = \max\left(0, UA \cdot (T_{\text{outdoor}} - T_{\text{set}})\right), \quad UA = \frac{0.30 \cdot \dot{Q}_{\text{rated}}}{\max(1.0, 35.0 - T_{\text{set}})}$$

#### C. Temperature Pull-Down Load ($\dot{Q}_{\text{pull}}$)
Proportional sensible cooling demand to drive indoor temperature toward setpoint:
$$\dot{Q}_{\text{pull}} = \max\left(0, K_{\text{pull}} \cdot (T_{\text{indoor}} - T_{\text{set}})\right), \quad K_{\text{pull}} = 0.40 \cdot \dot{Q}_{\text{rated}}$$

#### D. Total Cooling Demand & Equipment Capacity
$$\dot{Q}_{\text{demand}} = \dot{Q}_{\text{occ}} + \dot{Q}_{\text{trans}} + \dot{Q}_{\text{pull}}$$
$$\dot{Q}_{\text{cooling}} = \min\left(\dot{Q}_{\text{rated}}, \dot{Q}_{\text{demand}}\right)$$

#### E. Electrical Power Separation via COP
Cooling output thermal power is distinct from electrical compressor power:
$$P_{\text{elec}} = \begin{cases} 
0.0 & \text{if mode is OFF} \\
P_{\text{standby}} & \text{if within deadband } |T_{\text{in}} - T_{\text{set}}| \le 0.5^\circ\text{C} \text{ and } N=0 \\
\max\left(P_{\text{min\_stable}}, \min\left(P_{\text{rated\_elec}}, \frac{\dot{Q}_{\text{cooling}}}{\text{COP}}\right)\right) & \text{if active cooling is engaged}
\end{cases}$$
where $\text{COP} = 3.2$, $P_{\text{rated\_elec}} = \frac{\dot{Q}_{\text{rated}}}{\text{COP}}$, $P_{\text{min\_stable}} = 0.15\text{--}0.20 \cdot P_{\text{rated\_elec}}$, and $P_{\text{standby}} = 25\text{--}35\text{ W}$.

#### F. Operating Modes & Setpoints
- **NORMAL**: Occupied ($N > 0$) or vacant for $< 10\text{ min}$. Setpoint $T_{\text{normal}} = 24.5^\circ\text{C}$.
- **ECO**: Vacant ($N = 0$) for $\ge 10\text{ min}$. Setpoint set back to $T_{\text{eco}} = 28.0^\circ\text{C}$.
- **HIGH_DEMAND**: Occupancy ratio $\ge 75\%$. Compressor operates at elevated modulation.
- **OFF**: Manual operator override. Power is $0.0\text{ W}$.

---

### 2.2 Daylight-Harvesting Lighting Control

Lighting electrical power incorporates both natural daylight availability and occupancy vacancy:

#### A. Artificial Illuminance Deficit
$$\Delta E = \max\left(0, E_{\text{target}} - E_{\text{daylight}}\right)$$
where $E_{\text{target}} = 300\text{--}350\text{ lux}$ and $E_{\text{daylight}}$ is physical daylight calculated from solar altitude and room window geometric factors.

#### B. LED Dimming Fraction
$$\delta_{\text{dim}} = \begin{cases}
f_{\text{min\_dim}} & \text{if } \Delta E = 0\text{ lux (full daylight)} \\
\min\left(1.0, \max\left(f_{\text{min\_dim}}, \frac{\Delta E}{E_{\text{target}}}\right)\right) & \text{if } 0 < \Delta E < E_{\text{target}} \\
1.0 & \text{if } E_{\text{daylight}} = 0\text{ lux (nighttime)}
\end{cases}$$
where $f_{\text{min\_dim}} = 0.15\text{--}0.20$.

#### C. Electrical Power Driver Curve
$$P_{\text{lighting}} = \begin{cases}
0.0 & \text{if vacant for } \ge 10\text{ min or FORCE\_OFF} \\
P_{\text{installed}} & \text{if FORCE\_ON} \\
P_{\text{installed}} \cdot (0.10 + 0.90 \cdot \delta_{\text{dim}}) & \text{if occupied (AUTO)} \\
P_{\text{installed}} \cdot f_{\text{min\_dim}} & \text{if vacant for } < 10\text{ min (timeout period)}
\end{cases}$$

---

### 2.3 Subsystem Summation & Energy Integration

For each room $r$ at timestamp $t$:
$$P_{\text{total}, r}(t) = P_{\text{hvac}, r}(t) + P_{\text{lighting}, r}(t) + P_{\text{auxiliary}, r}(t)$$

Cumulative energy is integrated strictly across logical simulation increments $\Delta t = 300\text{ s}$:
$$E_{\text{hvac}, r}(t + \Delta t) = E_{\text{hvac}, r}(t) + \frac{P_{\text{hvac}, r}(t) \cdot \Delta t}{3{,}600{,}000}\text{ kWh}$$
$$E_{\text{lighting}, r}(t + \Delta t) = E_{\text{lighting}, r}(t) + \frac{P_{\text{lighting}, r}(t) \cdot \Delta t}{3{,}600{,}000}\text{ kWh}$$
$$E_{\text{auxiliary}, r}(t + \Delta t) = E_{\text{auxiliary}, r}(t) + \frac{P_{\text{auxiliary}, r}(t) \cdot \Delta t}{3{,}600{,}000}\text{ kWh}$$
$$E_{\text{total}, r}(t + \Delta t) = E_{\text{hvac}, r} + E_{\text{lighting}, r} + E_{\text{auxiliary}, r}$$

Building-level totals sum room quantities:
$$P_{\text{building}}(t) = \sum_{r=1}^5 P_{\text{total}, r}(t), \quad E_{\text{building}}(t) = \sum_{r=1}^5 E_{\text{total}, r}(t)$$

---

### 2.4 Baseline Policy Comparison & Savings Metrics

To provide an honest, reproducible energy comparison, a standard non-smart policy runs in parallel on the identical simulation trace:
- **Baseline HVAC**: Fixed setpoint $T_{\text{fixed}} = 24.0^\circ\text{C}$ 24/7 without eco setbacks or vacancy delays.
- **Baseline Lighting**: Full $100\%$ power throughout business hours ($08:00\text{--}18:00$), regardless of daylight levels or vacancy.
- **Comparative Metrics**:
  $$E_{\text{savings}} = \max\left(0, E_{\text{baseline}} - E_{\text{smart}}\right)$$
  $$\text{Savings \%} = \frac{E_{\text{savings}}}{E_{\text{baseline}}} \times 100\%$$
  $$P_{\text{peak}} = \max_t P_{\text{smart}}(t), \quad P_{\text{avg}} = \frac{1}{T} \sum_t P_{\text{smart}}(t)$$

---

## 3. Room Energy Parameters Summary

From [`config/energy_parameters.json`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/config/energy_parameters.json):

| Room ID | Room Name | Area ($m^2$) | Capacity | Cooling Capacity ($W$) | Rated Elec Power ($W$) | COP | Lighting Power ($W$) | Target Illuminance | Base Aux Power ($W$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room 1** | Main Conference Room / Lobby | 140.0 | 40 | 14,000 | 4,375.0 | 3.2 | 420.0 | 300 lx | 120.0 |
| **Room 2** | Open Office A | 120.0 | 25 | 12,000 | 3,750.0 | 3.2 | 360.0 | 350 lx | 150.0 |
| **Room 3** | Executive Office Suite | 80.0 | 15 | 8,000 | 2,500.0 | 3.2 | 240.0 | 350 lx | 100.0 |
| **Room 4** | Open Office B / Collaboration Lab | 120.0 | 30 | 12,000 | 3,750.0 | 3.2 | 360.0 | 350 lx | 140.0 |
| **Room 5** | Cafeteria & Breakroom | 100.0 | 35 | 10,000 | 3,125.0 | 3.2 | 300.0 | 300 lx | 150.0 |

---

## 4. Verification Test Suite Execution Results

All 20 required physical and software invariants were verified in [`tests/test_energy_management.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/tests/test_energy_management.py):

| Test # | Test Function Name | Verification Objective | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_1_room_independent_energy_state` | Independent energy states for all 5 rooms with required fields & ratings | **PASSED** |
| 2 | `test_2_occupancy_affects_control_promptly` | Occupant entry triggers prompt ECO $\to$ NORMAL mode transition | **PASSED** |
| 3 | `test_3_cooling_demand_reduction_when_vacant` | Zero occupancy engages eco setback ($28^\circ\text{C}$), reducing cooling demand | **PASSED** |
| 4 | `test_4_high_occupancy_increases_heat_gain_and_cooling` | High occupant count increases metabolic heat ($100\text{ W/p}$) and cooling power | **PASSED** |
| 5 | `test_5_vacant_room_switches_to_eco_after_delay` | Vacant room remains normal for 5 min, then switches to ECO after 10 min | **PASSED** |
| 6 | `test_6_idempotency_under_repeated_queries_advance_time_false` | Repeated UI reruns (`advance_time=False`) do not advance timers or kWh | **PASSED** |
| 7 | `test_7_daylight_harvesting_modulates_lighting_power` | Bright daylight dims lighting output down to minimum fraction ($15\%$) | **PASSED** |
| 8 | `test_8_lighting_vacancy_timeout` | Lighting remains on during 10-min vacancy delay, then shuts off | **PASSED** |
| 9 | `test_9_lighting_manual_overrides` | `FORCE_ON` and `FORCE_OFF` lighting overrides function reliably | **PASSED** |
| 10 | `test_10_hvac_manual_overrides` | `FORCE_OFF`, `ECO`, and `AUTO` manual overrides function reliably | **PASSED** |
| 11 | `test_11_hvac_power_within_physical_limits` | HVAC electrical power strictly bounded in $[P_{\text{standby}}, P_{\text{rated\_max}}]$ | **PASSED** |
| 12 | `test_12_energy_integration_kwh_formula` | Numerical integration $E = P \cdot \Delta t / 3{,}600{,}000$ computes accurately | **PASSED** |
| 13 | `test_13_strictly_idempotent_timestamp_processing` | Duplicate calls at the same timestamp do not double-count energy | **PASSED** |
| 14 | `test_14_subsystem_cumulative_energy_tracking` | HVAC, lighting, and auxiliary kWh tracked separately and sum to total | **PASSED** |
| 15 | `test_15_room_wise_powers_sum_to_building_totals` | Room instantaneous powers and cumulative energies sum to building KPIs | **PASSED** |
| 16 | `test_16_manual_room_transfer_updates_immediately` | Manual room transfers immediately update both source and destination rooms | **PASSED** |
| 17 | `test_17_simulation_reset_clears_energy_history` | Resetting engine resets energy history and clears cumulative kWh meters | **PASSED** |
| 18 | `test_18_invalid_energy_config_raises_error` | Invalid configurations (e.g., negative COP) raise descriptive `ValueError` | **PASSED** |
| 19 | `test_19_energy_dataframe_and_csv_export` | Energy dataframe and CSV export output all expected columns | **PASSED** |
| 20 | `test_20_baseline_comparator_savings_consistency` | Baseline comparator computes non-negative, finite, consistent savings | **PASSED** |

### Complete Test Suite Summary
```text
tests/test_dashboard.py .......                                          [  6%]
tests/test_energy_management.py ....................                     [ 26%]
tests/test_interactive_population.py ................                    [ 42%]
tests/test_manual_people_flow.py ...........                             [ 52%]
tests/test_phase6_sensor_physics.py ...............                      [ 67%]
tests/test_pipeline.py .........                                         [ 76%]
tests/test_sensors.py ..........                                         [ 86%]
tests/test_simulation.py ..............                                  [100%]

============================ 102 passed in 28.45s =============================
```

---

## 5. Engineering Limitations & Billing Disclaimer

> [!NOTE]
> 1. **Virtual Equipment Estimation**: All energy, power, and thermal calculations reflect first-principles engineering models (heat transfer, thermodynamic COP, LED driver curves). They do not represent revenue-grade electric utility meters.
> 2. **Model Boundary Preservation**: The STGNN occupancy prediction model (`joint_lstm_gnn_all_5_rooms.keras`) was preserved in its exact original state without retraining, maintaining its validated 15-minute horizon and 13-feature input shape.

