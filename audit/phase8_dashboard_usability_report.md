# Phase 8: Professional Building Energy Management System (BEMS) Usability & Interactivity Report

## 1. Executive Summary & Objectives

The primary objective of Phase 8 was to transform the existing research-grade Streamlit occupancy dashboard into an industrial-grade, highly interactive **Building Energy Management System (BEMS) Control Centre**.

Prior to Phase 8, the dashboard was organized as a 988-line monolithic script with controls scattered across fragmented tabs, duplicate widgets in both the sidebar and main body, no continuous simulation playback or step duration configuration, no visual room selection on the 2D floor plan, and missing contextual operational controls.

Phase 8 introduces a modern industrial IoT control-room architecture:
1. **Global BEMS Simulation Toolbar**: Persistent at the top of the interface, providing Play/Pause auto-stepping, configurable step sizes (+5m, +15m horizon, +30m, +60m lookback), confirmation dialogs for resets, simulated logical clock, and clear data origin indicators.
2. **Interactive 2D Floor Plan as Central Focus**: Highlights the active room with high-contrast cyan borders, room status badges (Normal, High Occupancy, Elevated CO₂, Eco Mode, High Demand), and rich telemetry tooltips.
3. **Selected Room Control Centre**: Unified contextual panel synchronized with the active room, consolidating Occupancy, Environmental multi-physics, HVAC setpoints, Daylight-harvesting lighting, and real-time physical explainability ("Why is this room consuming power?").
4. **Synchronized Room Summary Cards**: One-click room selection cards updating the floor plan and control centre simultaneously.
5. **Streamlined People Flow Operations**: Intuitive Add (with entrance queue preview), Move (with topological doorway path validation), and Exit controls.
6. **BEMS Demonstration Scenarios**: Six reproducible presets (Normal Workday, Low Occupancy, Peak Surge, Empty Building, Aggressive Energy Saving, Summer Heatwave) with preview and deterministic execution.
7. **Modular Software Architecture**: Refactored `dashboard/` into clean, maintainable view modules (`components.py`, `room_detail.py`, `overview.py`, `people_flow.py`, `sensors_view.py`, `energy_view.py`, `prediction_view.py`, `exports_view.py`, `scenarios.py`).

---

## 2. Usability & Architecture Audit Findings

| Identified Usability Issue | Previous Implementation | Phase 8 Resolution |
| :--- | :--- | :--- |
| **Monolithic Codebase** | `app.py` was 988 lines containing all UI rendering and state mutations. | Decomposed into 9 modular packages under `dashboard/`. `app.py` reduced to a clean 140-line orchestrator. |
| **Duplicate & Confusing Controls** | Add/Exit people was duplicated in sidebar and Tab 2; lighting controls duplicated in Tab 3 and Tab 4. | Unified into dedicated operational workspaces: `People Flow` and `Selected Room Control Centre`. |
| **No Visual Room Selection** | Floor plan was purely static; clicking rooms or seeing active focus was impossible. | Added `selected_room_id` highlight (border width 5, cyan `#00F0FF`) and 5 synchronized room selector cards. |
| **Fragmented Room Information** | Operators had to browse 4 separate tabs to see one room's occupancy, sensors, HVAC, and energy. | Built unified **Selected Room Control Centre** contextual panel presenting all 5 dimensions together. |
| **Lack of Simulation Stepping Control** | Stepping was fixed to 5m in sidebar; continuous playback was unsupported. | Added Global Toolbar with Play/Pause, configurable duration (+5m, +15m, +30m, +60m), and auto-playback interval. |
| **Accidental Simulation Resets** | Clicking reset immediately cleared all state without confirmation. | Implemented native `@st.dialog` confirmation dialog displaying warning and parameter sliders before reset. |
| **No Preconfigured Scenarios** | Demonstrating edge cases required manual tedious clicking and occupant manipulation. | Implemented 6 reproducible BEMS scenario presets with impact preview and deterministic one-click application. |
| **Opaque Energy Consumption** | High power consumption was reported without physical explanation. | Added real-time **Physical Explainability** calculating occupant sensible heat (+100 W/occ), thermal delta ($T_{in} - T_{sp}$), and daylight deficit. |

---

## 3. Major Features & Interactions Implemented

### 3.1 Global BEMS Simulation Toolbar (`dashboard/components.py`)
- **System Badges**: Displays current engine state (`▶ RUNNING (Auto-Step)`, `⏸ PAUSED`, `⏹ IDLE`), Physics Engine indicator, and STGNN Forecasting status.
- **Continuous Playback**: Play/Pause button initiates controlled auto-stepping with selectable speed (0.5s, 1.0s, 2.0s, 3.0s per step) without infinite blocking loops.
- **Configurable Step Multiplier**:
  - `5 min (1 simulation tick)`
  - `15 min (3 ticks — 1 STGNN Forecast Horizon)`
  - `30 min (6 ticks — 0.5 hour)`
  - `60 min (12 ticks — 1 STGNN Lookback Window)`
- **Logical Simulation Clock**: Formatted UTC timestamp, discrete step count, and active occupancy count.
- **Safe Reset Dialog**: Native confirmation popup preventing accidental data loss.

### 3.2 Interactive Floor Plan & Room Selection (`dashboard/floorplan.py`)
- **Visual Room Highlighting**: Selected room receives a bright `#00F0FF` border (width 5) and `[SELECTED]` marker.
- **Rich Room Status Badges**:
  - `🟢 Normal` (<50% utilization, nominal CO₂)
  - `🟠 Moderate Load` (50–85% utilization)
  - `🔴 High Occupancy` (≥85% capacity utilization)
  - `🟠 Elevated CO₂` (>1000 ppm)
  - `🌿 Eco Mode` (Setback setpoint active)
  - `⚡ High Demand` (Active cooling refrigeration)
- **Synchronized Room Summary Cards**: 5 interactive cards below the floor plan displaying Occupancy, Temp, CO₂, HVAC Mode, Power, and Cumulative Energy with one-click "Focus Room" action.

### 3.3 Selected Room Control Centre (`dashboard/room_detail.py`)
Provides contextual operational controls for `st.session_state.selected_room`:
- **Occupancy & Actions**: Live count/capacity bar, detailed occupant roster (ID, type, activity, state), quick inter-room transfer to connected doors, quick exit.
- **Environmental Telemetry**: Multi-physics cards grouped into Immediate (Lux, Sound), Dynamic (Indoor Temp, RH, CO₂), and Outdoor boundary conditions (Dry-Bulb Temp, RH).
- **HVAC & Setpoint**: Mode status (AUTO, ECO, HIGH_DEMAND, OFF), indoor temp vs setpoint, cooling thermal demand (W), electrical power (W), setpoint slider (18.0–28.0°C), manual mode override radio.
- **Lighting Control**: Mode radio (Auto Daylight Harvesting, Force ON, Force OFF), measured lux, daylight vs artificial luminaire lux, vacancy timer (countdown to 10 min off).
- **Energy & Explainability**: Power breakdown, cumulative kWh, and real-time physical drivers explaining why the room consumes electrical power.

### 3.4 BEMS Demonstration Scenarios (`dashboard/scenarios.py`)
| Preset Key | Scenario Name | Population | Setpoint | Outdoor Temp | Operational Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `normal` | Normal Workday | 40 occ | 24.5 °C | 28.03 °C | Nominal business hours with balanced occupancy and standard schedules. |
| `low_occupancy` | Low Occupancy (Off-Peak) | 15 occ | 24.5 °C | 28.03 °C | Off-peak operation; vacant rooms trigger 10-minute vacancy setbacks. |
| `peak_surge` | Peak Occupancy Surge | 80 occ | 24.0 °C | 29.50 °C | High thermal and metabolic loads testing maximum HVAC cooling capacity. |
| `empty_building` | Empty Building (Night/Weekend) | 0 occ | 28.0 °C | 28.03 °C | 0 occupants; verifies 10-min vacancy timeouts, all rooms drop to ECO mode. |
| `energy_saving` | Aggressive Energy Saving | 40 occ | 26.5 °C | 28.03 °C | 26.5°C setpoints and strict daylight harvesting saving 35–45% energy. |
| `heatwave` | Summer Heatwave Stress Test | 40 occ | 24.0 °C | 35.00 °C | Extreme ambient weather (35°C dry-bulb) stressing building chiller envelope. |

---

## 4. Software Architecture & File Structure

```
Digital_Twin/
├── app.py                            # Streamlit entrypoint orchestrating modular BEMS views
├── dashboard/
│   ├── __init__.py                   # Package exports
│   ├── components.py                 # Theme CSS, Global Toolbar, Dialogs, Feedback banners
│   ├── floorplan.py                  # 2D Floor Plan with selection highlight and status badges
│   ├── room_detail.py                # Selected Room Control Centre (Occupancy, Sensors, HVAC, Energy)
│   ├── overview.py                   # Building Overview workspace tab
│   ├── people_flow.py                # People Flow Control workspace tab
│   ├── sensors_view.py               # Environmental Telemetry workspace tab
│   ├── energy_view.py                # Smart HVAC & Energy Management workspace tab
│   ├── prediction_view.py            # STGNN Predictive Evaluation workspace tab
│   ├── exports_view.py               # Data Export & Audit Center workspace tab
│   ├── scenarios.py                  # 6 BEMS operational demonstration presets
│   ├── state.py                      # Session state management (selected_room, feedback, stepping)
│   └── charts.py                     # Plotly time-series and comparative charts
├── simulation/                       # Physical building, sensor, and energy simulation engine
├── tests/
│   ├── test_dashboard.py             # Phase 4 original dashboard tests (7 passed)
│   ├── test_dashboard_bems.py        # Phase 8 new BEMS interactivity tests (11 passed)
│   ├── test_energy_management.py     # Phase 7 energy tests (20 passed)
│   ├── test_phase6_sensor_physics.py # Phase 6 sensor physics tests (15 passed)
│   ├── test_manual_people_flow.py    # Manual flow tests (11 passed)
│   ├── test_interactive_population.py# Interactive population tests (16 passed)
│   ├── test_sensors.py               # Sensor tests (10 passed)
│   ├── test_simulation.py            # Building simulation tests (14 passed)
│   └── test_pipeline.py              # End-to-end inference pipeline tests (9 passed)
```

---

## 5. Verification & Testing

### Test Suite Execution
The automated test suite was executed across all existing and new test modules:

```powershell
pytest tests/test_dashboard_bems.py
pytest
```

**Results:**
- `tests/test_dashboard_bems.py`: **11 passed in 14.2s**
- Full test suite: **113 passed in 42.1s** (100% passing across all 9 test suites, zero regressions).

---

## 6. How to Run the BEMS Dashboard

To start the interactive BEMS application:

```powershell
streamlit run app.py
```

### Quick Operator Walkthrough:
1. **Focus a Room**: Click on any room pill above the floor plan (e.g. `Room 2`) or click `Focus Room` on a Room Summary Card. The floor plan highlights `Room 2` in bright cyan, and the **Selected Room Control Centre** updates with Room 2's live occupancy, sensors, HVAC controls, and energy explainability.
2. **Step the Simulation**: In the top toolbar, select `15 min` duration and click `⏭ Step (+15m)`. The clock advances 15 minutes (3 ticks), sensors dynamically evolve, and STGNN outputs a new 15-minute ahead forecast.
3. **Dispatch Occupants**: Navigate to **🚶 People Flow Control**. Select Room 1 as source and Room 2 as destination. Enter 3 occupants. The interface validates doorway connectivity and destination capacity. Click `Move Occupant(s)`.
4. **Inspect Energy Drivers**: Open **⚡ Smart HVAC & Energy**. Note the "Highest Consuming Room" callout explaining occupant sensible heat (+100 W/occ) and temperature delta.
5. **Load a Scenario**: Click `🎬 Scenario` in the toolbar. Select `Summer Heatwave Stress Test` or `Empty Building`. Preview the engineering impacts and click `Apply Scenario`.

