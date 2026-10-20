# Phase 5 — Interactive Person Entry, Unlimited Population, and Live Sensor Monitoring Report

**Status**: VERIFIED & COMPLETE  
**Date**: 2026-10-15  
**Scope**: Interactive population control, dynamic capacity queueing, immediate vs continuous sensor coupling, live dashboard telemetry with deltas, and automated regression testing.  
**Total Automated Tests**: 56 passing (16 new interactive tests + 40 existing regression tests).  

---

## 1. Executive Summary

Phase 5 successfully expands the 5-room building digital twin with interactive, real-time population management and live environmental sensor telemetry without altering model weights, scalers, or raw historical datasets.

Key accomplishments include:
1. **Interactive Person Entry & Unlimited Population**: Users can add any arbitrary number of individuals (single or batch) through Room 1 (Main Entrance). No artificial ceilings (50/100/120) are imposed.
2. **Deterministic Unique Identification**: Every manually added individual receives a strictly unique, sequential ID (`P0001`, `P0002`, ...). The session counter is monotonic and preserved across resets.
3. **Physical Capacity Enforcement & FIFO Queueing**: When Room 1 reaches capacity (40 occupants), excess arrivals are held in an exterior Entrance Queue outside Room 1. People are never discarded. As occupants move out of Room 1 during simulation steps, queued occupants enter automatically in first-in, first-out (FIFO) order.
4. **Immediate vs. Continuous Sensor Dynamics**: 
   - *Immediate Response*: Sound pressure level ($L_p$) reacts instantaneously via logarithmic acoustic energy summation ($10 \log_{10}(10^{L_{\text{base}}/10} + N \cdot 10^{L_{\text{activity}}/10})$) and artificial lighting turns on immediately when occupied.
   - *Continuous Response*: Carbon dioxide ($CO_2$), indoor temperature, and relative humidity evolve continuously across 5-minute logical steps via physical differential equations.
5. **Interactive 2D Floor Plan Upgrades**: Renders active entrance queue indicators outside Room 1 (`x=7.0, y=-1.5`) and applies deterministic position dispersion/jittering when room populations exceed 30 occupants, preserving Plotly rendering speed.
6. **Live Sensor Monitoring Panel**: Added cross-room comparison table and single-room telemetry cards displaying 7 environmental features with units and step-to-step deltas ($\Delta$).
7. **End-to-End Pipeline Compatibility**: The STGNN inference pipeline continues to receive valid tensors of shape `(1, 12, 5, 13)` and accurately forecasts 15 minutes ahead.
8. **Automated Verification**: All 16 newly implemented test cases pass cleanly, bringing total automated test coverage to 56/56 passing tests.

---

## 2. Interactive Population & Queueing Architecture

```mermaid
flowchart TD
    User["User Interface (Streamlit app.py)"] -->|add_person(count, dest)| Pipeline["DigitalTwinPipeline"]
    Pipeline --> Engine["SimulationEngine"]
    
    subgraph AdmissionControl ["Room 1 Entrance Gateway (Capacity: 40)"]
        Engine --> CheckCap{"Room 1 has capacity?"}
        CheckCap -->|Yes| Admit["Admit into Room 1 (Pxxxx)"]
        CheckCap -->|No| Queue["Place in Entrance Queue (waiting_at_entrance)"]
    end
    
    Admit --> ImmSensors["refresh_immediate_sensors() (Acoustic SPL & Lighting)"]
    Admit --> GroundTruth["Ground Truth Update"]
    
    subgraph SimulationTick ["Simulation Clock Step (+5 min)"]
        Step["SimulationEngine.step()"] --> FreeSpace["Occupants transition out of Room 1"]
        FreeSpace --> Dequeue["Admit FIFO from Entrance Queue up to available capacity"]
        Dequeue --> PhysEng["Environmental Physics Engine (CO2, Temp, RH)"]
        PhysEng --> STGNN["STGNN 15-Min Prediction Pipeline (1, 12, 5, 13)"]
    end
```

### 2.1 Monotonic Sequential Person IDs
- Each interactive addition generates an occupant instance with ID format `P{counter:04d}` (e.g. `P0001`, `P0002`).
- The `SimulationEngine` maintains `self.person_counter`.
- Calling `reset(keep_session_counter=True)` retains monotonic continuity across user resets, preventing identifier collisions during interactive sessions.

### 2.2 Capacity Queueing Mechanics
- Room 1 is the physical entry gateway for the facility (configured capacity: 40).
- If a user adds 55 people to an empty building, 40 are admitted immediately into Room 1 and 15 are placed in `engine.entrance_queue` with state `"waiting_at_entrance"`.
- During each 5-minute logical clock tick:
  1. Queued occupants are checked against Room 1 available capacity ($\text{available} = \text{capacity} - \text{occupants}$).
  2. Up to `available` occupants are popped from the head of the queue (FIFO) and admitted into Room 1.
  3. Admitted occupants begin spatial wandering or following their assigned routes to destination rooms.

---

## 3. Sensor Coupling Dynamics

The virtual sensor engine differentiates between algebraic instantaneous variables and dynamic state variables:

| Environmental Feature | Physical Mechanism | Response Time | Formula / Governing Equation |
| :--- | :--- | :--- | :--- |
| **Sound Pressure Level** (`sound_pressure_level`) | Acoustic power summation | **Immediate** | $L_p = 10 \log_{10}\left(10^{L_{\text{base}}/10} + N \cdot 10^{L_{\text{activity}}/10}\right) + \epsilon$ |
| **Illuminance** (`illuminance`) | Lighting automation | **Immediate** | Artificial illumination ($150\text{ lux}$) triggers if $N > 0$ or within working hours |
| **Carbon Dioxide** (`indoor_co2`) | Metabolic mass balance | **Continuous** (5-min steps) | $\frac{dC}{dt} = \frac{G \cdot N}{V} - \lambda(C - C_{\text{out}})$ |
| **Indoor Temperature** (`indoor_temperature`) | Lumped thermal capacitance | **Continuous** (5-min steps) | $\frac{dT}{dt} = k_{\text{hvac}}(T_{\text{set}} - T) + k_{\text{env}}(T_{\text{out}} - T) + k_{\text{occ}} N$ |
| **Indoor Relative Humidity** (`indoor_relative_humidity`) | Moisture balance & HVAC | **Continuous** (5-min steps) | $\frac{dRH}{dt} = k_{\text{hvac}}(RH_{\text{set}} - RH) + k_{\text{vent}}(RH_{\text{out}} - RH) + k_{\text{occ}} N$ |
| **Dry-Bulb Temperature** (`dry_bulb_temperature`) | Ambient outdoor weather | **Independent** of occupants | Asymmetric diurnal solar heating curve ($23^\circ\text{C} - 35^\circ\text{C}$) |
| **Outdoor Relative Humidity** (`outdoor_relative_humidity`) | Ambient outdoor weather | **Independent** of occupants | Atmospheric humidity anti-correlated with ambient temperature |

---

## 4. Dashboard Implementation

The Streamlit dashboard (`app.py`) includes the following enhancements:

1. **Sidebar Population Controls**:
   - Number input with no arbitrary ceiling (minimum 1, default 1).
   - Optional destination selector (`Room 1` to `Room 5` or `Free Movement`).
   - "Add Person(s) at Entrance" action button with informative toast/status alerts.
   - "Exit Person(s)" action button with building-wide or room-specific filtering.
   - Live KPI counters: Inside Building, Entrance Queue, Total Arrivals, Total Exits.
2. **Floor Plan Upgrades (`dashboard/floorplan.py`)**:
   - Exterior entrance queue indicator rendered outside Room 1 at $(7.0, -1.5)$ when occupants are queued.
   - Deterministic dispersion algorithm for high-density rooms ($N > 30$) to prevent marker stacking and keep Plotly WebGL rendering responsive.
3. **Live Room Sensors Section (Tab 2)**:
   - **Cross-Room Snapshot Table**: Displays all 5 rooms, current occupants, capacities, and all 7 sensor readings in a single responsive table.
   - **Single-Room Live Telemetry & Deltas**: 7 metric cards showing current values and step-to-step deltas ($\Delta = \text{val}_t - \text{val}_{t-1}$) with engineering units.
   - **Trend Visualizer**: Multi-trace Plotly time series for selected features.

---

## 5. Automated Test Suite Results

A dedicated automated test suite (`tests/test_interactive_population.py`) was created, containing 16 unit and integration test cases:

| Test ID | Test Function | Description | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_single_person_entry` | Validates single person admission into Room 1 (`P0001`) | **PASS** |
| 2 | `test_batch_person_entry_25` | Validates simultaneous batch entry of 25 occupants | **PASS** |
| 3 | `test_repeated_person_additions` | Validates sequential additions across separate actions | **PASS** |
| 4 | `test_sequential_unique_ids` | Validates sequential `P0001`+ numbering without collision | **PASS** |
| 5 | `test_occupants_persist_across_steps` | Validates occupants persist inside building over steps | **PASS** |
| 6 | `test_entrance_queueing_at_capacity` | Validates excess arrivals queue when Room 1 capacity is full | **PASS** |
| 7 | `test_fifo_admission_from_entrance_queue` | Validates queued occupants enter FIFO as capacity frees | **PASS** |
| 8 | `test_no_silent_discarding_under_high_load` | Validates 100 people added at once are all accounted for | **PASS** |
| 9 | `test_metrics_consistency_no_double_counting` | Validates `arrivals - exits == inside` and queue consistency | **PASS** |
| 10 | `test_exit_person_functionality` | Validates manual occupant removal from rooms | **PASS** |
| 11 | `test_immediate_acoustic_sensor_response` | Validates sound level increases immediately upon entry | **PASS** |
| 12 | `test_continuous_co2_accumulation_over_steps` | Validates CO2 accumulates gradually across steps | **PASS** |
| 13 | `test_outdoor_environmental_features_independent` | Validates outdoor temperature/RH unaffected by occupants | **PASS** |
| 14 | `test_pipeline_compatibility_and_forecasting` | Validates STGNN `(1, 12, 5, 13)` inference works with added people | **PASS** |
| 15 | `test_reset_preserves_person_counter` | Validates monotonic ID continuity across session resets | **PASS** |
| 16 | `test_reset_restarts_counter_when_requested` | Validates reset to `P0001` when explicitly requested | **PASS** |

### Complete Test Suite Summary
```text
tests/test_dashboard.py .............. [ 7 passed ]
tests/test_interactive_population.py . [ 16 passed ]
tests/test_pipeline.py ............... [ 9 passed ]
tests/test_sensors.py ................ [ 10 passed ]
tests/test_simulation.py ............. [ 14 passed ]
======================= 56 passed in 29.04s =======================
```

---

## 6. Boundary Preservation Verification

- **Model Weights**: `joint_lstm_gnn_all_5_rooms.keras` was untouched and not retrained.
- **Preprocessing Scalers**: `artifacts/scalers.joblib` was not modified.
- **Historical Datasets**: All 5 CSV files in `dataset/` remain read-only and unmodified.
- **Tensor Architecture**: Rolling buffer shape `(1, 12, 5, 13)` and 15-minute forecast horizon are strictly preserved.

