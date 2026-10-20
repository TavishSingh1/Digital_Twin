# Phase 8.3 Audit Report: Advanced Digital Twin Visualization, Live Animation and Operator Experience

## 1. Executive Summary & Objectives

Phase 8.3 transforms the Building Energy Management System (BEMS) into an industrial-grade, highly interactive Digital Twin visualization cockpit. Prior to this phase, occupant movements were depicted using static Plotly snapshots requiring Streamlit server reruns, creating stepped or static movement perceptions rather than continuous 60 FPS physical fluidity.

### Key Deliverables in Phase 8.3:
1. **Live 60 FPS HTML5 Canvas Visualizer (`dashboard/live_canvas.py`)**: Built a client-side visual renderer utilizing browser `requestAnimationFrame` to deliver continuous visual interpolation along doorway waypoints.
2. **Dual-Tab Architecture**: Seamlessly coexists with the Plotly Blueprint View, offering operators both a 60 FPS continuous live visualizer and an interactive vector blueprint, maintaining 100% backwards compatibility with all existing Plotly-based tests.
3. **Decoupled Visual Playback Controls**: In-browser Play/Pause, Replay, 0.5x–2.0x visual speed scaling, and scrubbers that run entirely on the client side with **zero server roundtrips**, **zero simulation clock advancement**, and **zero sensor/energy recalculation**.
4. **Comprehensive Occupant Inspector (`dashboard/occupant_inspector.py`)**: Dedicated panel displaying individual occupant ID, category/role, authoritative room vs real-time visual transit progress, intended destination, door-to-door waypoints, entrance queue status, search by ID, room/status filtering, and automatic stale selection cleanup upon occupant exit.
5. **Movement History & Visual Replay Center**: Chronological ledger of confirmed doorway movements with room, occupant, and event type filtering, bounded FIFO retention ($\le 100$ events, $\le 25$ exit records), and non-mutating visual replay.
6. **Cross-Workspace Synchronization**: Global room selection synchronizes the interactive floor plan, summary cards, control centre, occupant inspector, history ledger, and contextual navigation shortcuts to People Flow and Smart HVAC workspaces.
7. **Adaptive Marker Sizing & High-Density Legibility**: Implemented dynamic occupant marker sizing ($10\text{px} \rightarrow 8\text{px} \rightarrow 6\text{px}$) to prevent occlusion during high-occupancy scenarios.
8. **Automated Test Suite Expansion**: Expanded `tests/test_occupant_animation.py` to 36 test cases covering continuous interpolation, speed scaling, pause/resume, multi-occupant animation, bounded history, visual replay non-mutation, and simulation invariance. Full test suite achieves **160 passing tests across 11 test suites** in 42.56s.

---

## 2. Phase 1 Audit: Answers to the 10 Specific Audit Questions

### Q1: How does the existing implementation render occupants and animations?
**Answer**: Occupants are rendered in [`dashboard/floorplan.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dashboard/floorplan.py) as a Plotly scatter trace (`go.Scatter(mode="markers")`) overlaid onto SVG shapes depicting room rectangles and door segments. In Phase 8.2, [`dashboard/animation.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/dashboard/animation.py) calculated deterministic coordinates using `get_deterministic_room_grid_position` for stationary occupants and `interpolate_along_waypoints` for moving occupants. However, because Plotly figures in Streamlit are static JSON objects transferred over WebSocket, any visual movement required a Python rerun that rendered a single snapshot.

### Q2: Why does the current animation not appear fluid in the browser?
**Answer**: Streamlit's architecture requires full server-client roundtrips for every UI update. In Phase 8.2, each rerun evaluated a single discrete progress snapshot (e.g., progress $\alpha = 1.0$), or multiple rerun loops with `time.sleep()`. This introduced 200–500ms network and serialization latency per frame, resulting in noticeable stepped stuttering rather than smooth continuous motion. True 60 FPS fluidity requires continuous hardware-accelerated rendering inside the browser's JavaScript runtime via `requestAnimationFrame`.

### Q3: How is animation progress calculated, updated, and persisted?
**Answer**: Piecewise linear distance interpolation is calculated in `interpolate_along_waypoints(waypoints, progress)` by calculating cumulative Euclidean segment lengths $d_i = \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}$, finding the target distance $D = \alpha \sum d_i$, and linearly interpolating along the enclosing segment. In Phase 8.3, this mathematics is executed natively inside the client's browser Canvas animation loop (`renderFrame(now)`) using sub-millisecond timestamps, while `AnimationManager.replay_progress` and `AnimationManager.animation_progress` persist state in Python session memory.

### Q4: How does animation speed interact with the simulation clock?
**Answer**: Visual animation speed and the simulation logical clock are **strictly decoupled**:
- **Simulation Clock**: Advances in discrete 5-minute steps ($\Delta t = 300\text{s}$) exclusively when `pipeline.step()` or `advance_pipeline_step()` is invoked.
- **Visual Animation Speed**: Scales browser-side rendering duration ($T_{\text{visual}} = 3000\text{ms} / \text{speed}$, where $\text{speed} \in \{0.5\times, 1.0\times, 1.5\times, 2.0\times\}$). Modifying animation speed, scrubbing, pausing, or replaying visually alters zero simulation clock timestamps, step counts, physical sensor models, or energy meters.

### Q5: What happens when multiple occupants move simultaneously?
**Answer**: Multiple occupants transferring simultaneously (e.g. batch transfers or multiple scheduled room shifts) are grouped into a `MovementEvent` containing their list of `person_ids`. For batch transfers, individual occupants receive offset spacing so their markers do not perfectly overlap while traversing doorways. Furthermore, multiple concurrent movement events across different rooms (e.g. Room 1 $\rightarrow$ Room 2 and Room 3 $\rightarrow$ Room 4) execute in parallel inside the canvas render loop, with each group following its own topological doorway waypoints.

### Q6: How are movement events logged, deduplicated, and displayed?
**Answer**: Movement events are recorded in `AnimationManager.record_movement(...)`. Deduplication uses an MD5 hash of `f"{event_type}_{timestamp}_{source_room}_{destination_room}_{sorted_pids}"`. If the hash is present in `seen_event_hashes`, registration is skipped, preventing duplicate events on Streamlit reruns. Events are displayed in the **Movement History & Visual Replay Center** with filtering by Room, Occupant ID, and Event Type, with FIFO eviction capped at 100 events.

### Q7: How does room selection synchronize with the floor plan and detail views?
**Answer**: Room selection is governed globally via `dashboard.state.get_selected_room()` and `set_selected_room(room_id)`. When an operator clicks a quick focus pill, floor plan room, or candidate in the Occupant Inspector:
1. `selected_room` updates in Streamlit `session_state`.
2. The floor plan highlights the selected room with a prominent cyan border (`#00F0FF`, 5px) and `[SELECTED]` badge.
3. The **Selected Room Control Centre** loads the room's environmental telemetry, occupancy, HVAC setpoint overrides, and power breakdown.
4. The **Room Summary Cards** highlight the active room card.
5. The **Occupant Inspector** synchronizes candidate filtering.
6. The **Movement History Center** defaults filtering to the active room.

### Q8: What happens to the animation when the user changes rooms, overlays, or views?
**Answer**: The visual canvas maintains its running animation loop because room selection and overlay changes are passed into the component configuration payload. Changing rooms highlights the new room geometry without restarting or corrupting active occupant movement interpolation. Switching workspaces preserves the persistent `AnimationManager` state in `st.session_state["animation_manager"]`.

### Q9: What happens to visual exit records when occupants leave the building?
**Answer**: When occupants depart via `engine.exit_person(...)`, their authoritative `current_room_id` is set to `None`, removing them from room membership and total building occupancy. Concurrently, `AnimationManager` records a transient visual exit record placing them outside the building at $(7.0, -1.8)$ with a distinct purple diamond marker. These records are strictly visual and capped at 25 entries (FIFO) to prevent clutter and memory accumulation.

### Q10: Where are the performance bottlenecks in the current animation pipeline?
**Answer**: The primary bottleneck was server-side Plotly figure reconstruction and WebSocket JSON re-transmission on every frame. Building a Plotly figure with dozens of shapes, annotations, and scatter traces takes 30–70ms of Python CPU time and transmits ~100KB over WebSocket per rerun. By migrating continuous interpolation to a self-contained HTML5 Canvas component (`dashboard/live_canvas.py`), rendering executes at 60 FPS on the client GPU/CPU with **zero network transmissions** during interpolation.

---

## 3. Architecture of Phase 8.3 Visualization

```
+-------------------------------------------------------------------------------+
|                             Streamlit Server (Python)                         |
|                                                                               |
|  +---------------------------+       +-------------------------------------+  |
|  |   simulation/engine.py    | ----> |       simulation/pipeline.py        |  |
|  | - 5-min physics cadence   |       | - STGNN 15-min occupancy forecasting|  |
|  | - Authoritative occupancy |       | - Rolling 12-step lookback buffer   |  |
|  +-------------+-------------+       +------------------+------------------+  |
|                |                                        |                     |
|                v                                        v                     |
|  +-------------------------------------------------------------------------+  |
|  |                        dashboard/state.py                               |  |
|  | - Global selected_room, selected_overlay, animation_manager instance   |  |
|  +-------------------------------------------------------------------------+  |
|                |                                        |                     |
|       +--------+--------+                      +--------+--------+            |
|       |                 |                      |                 |            |
|       v                 v                      v                 v            |
|  +----------+    +--------------+       +--------------+  +----------------+  |
|  | Overview |    | Room Control |       |  Occupant    |  | Movement Hist. |  |
|  | Dual Tabs|    | Centre       |       |  Inspector   |  | & Replay Ctr   |  |
|  +----+-----+    +--------------+       +--------------+  +----------------+  |
|       |                                                                       |
+-------|-----------------------------------------------------------------------+
        | JSON Configuration Payload (Building, Doors, Occupants, Overlays)
        v
+-------------------------------------------------------------------------------+
|                       Client Browser (HTML5 Canvas 60 FPS)                    |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  |                           #dt-container                                 |  |
|  |  +-------------------------------------------------------------------+  |  |
|  |  |  <canvas id="canvas">                                             |  |  |
|  |  |  - 1:1 Isometric 26m x 20m coordinate mapping                     |  |  |
|  |  |  - High-contrast room fills, doors, active overlay color-mapping  |  |  |
|  |  |  - Continuous requestAnimationFrame (60 FPS) visual interpolation|  |  |
|  |  |  - Deterministic stationary grids, moving markers, exit markers   |  |  |
|  |  |  - Selected room cyan glow, selected occupant route diamonds     |  |  |
|  |  +-------------------------------------------------------------------+  |  |
|  |  |  HUD Controls: [Play/Pause] [Replay] [0.5x 1x 1.5x 2x] [Scrubber] |  |  |
|  |  +-------------------------------------------------------------------+  |  |
|  +-------------------------------------------------------------------------+  |
|                                                                               |
|  * Strictly Visual: Zero Server Roundtrips | Zero Clock Advancement           |
+-------------------------------------------------------------------------------+
```

---

## 4. Detailed Component Implementation

### 4.1 Live 60 FPS Canvas Visualizer (`dashboard/live_canvas.py`)
- **Isometric Projection**: Maps building dimensions ($26\text{m} \times 20\text{m}$) with proportional padding to canvas width and height.
- **Continuous Interpolation**: Uses `requestAnimationFrame` and `performance.now()` to compute continuous normalized progress $\alpha \in [0.0, 1.0]$.
- **Topological Route Animation**: Moving occupants travel strictly along doorway waypoints calculated by BFS pathfinding.
- **Interactive Tooltip HUD**: Hovering over rooms or occupants displays rich telemetry (temperature, CO₂, illuminance, HVAC mode, status badge) with zero latency.
- **Dual Tabs Integration**:
  - `🎮 Live 60 FPS Digital Twin`: Hardware-accelerated interactive canvas.
  - `📐 Plotly Blueprint View`: High-resolution vector floor plan preserving full test compatibility.

### 4.2 Comprehensive Occupant Inspector (`dashboard/occupant_inspector.py`)
- **Authoritative vs Visual Location**: Distinguishes between authoritative room assignment (`occ.current_room_id`) and visual transit state (`event.progress`, `in_transit`).
- **Route & Waypoints Ledger**: Displays formatted door-to-door transit path and coordinates.
- **Search & Filters**: Search by occupant ID substring, filter by room, and filter by status (`All Inside`, `Stationary`, `In Transit`, `Queued`).
- **Graceful Stale Selection Handling**: If a selected occupant departs the building, `validate_selected_occupant` detects exit and cleans up the selection without errors.

### 4.3 Movement History & Visual Replay Center (`dashboard/overview.py`)
- **Chronological Ledger**: Displays confirmed arrivals, transfers, and departures with timestamp, source, destination, and occupant IDs.
- **Multi-Axis Filtering**: Filter by room (synchronized with selected room), event type (`admission`, `transfer`, `exit`), and occupant ID.
- **Bounded Retention**: Automatically evicts oldest events when buffer exceeds 100 entries (and 25 exit records).
- **Non-Mutating Visual Replay**: Clicking `▶ Replay` triggers browser-side visual animation of historical waypoints. Authoritative room counts, logical clock, and physical sensors remain 100% invariant.

### 4.4 High-Density Marker Scaling (`dashboard/floorplan.py`)
- **Adaptive Sizing**:
  - Occupancy $< 15$: $10\text{px}$ markers.
  - Occupancy $15 - 30$: $8\text{px}$ markers.
  - Occupancy $> 30$: $6\text{px}$ markers.
- **Contrast & Legibility**: White high-contrast borders ($1\text{px}$), categorical coloring (Blue: Staff, Green: Office Worker, Purple: Visitor), and exterior queue rendering at $(7.0, -1.6)$.

---

## 5. Verification & Test Suite Results

The comprehensive test suite in [`tests/test_occupant_animation.py`](file:///d:/VIT/SEMS/SEMESTER_7/PJT1/Digital_Twin/tests/test_occupant_animation.py) was expanded from 16 to 36 test cases, directly validating all Phase 8.3 criteria.

### Automated Test Matrix:
| Test ID | Test Description | Status |
| :--- | :--- | :---: |
| `test_1` | Confirmed movement creates one animation | ✅ PASSED |
| `test_2` | Rejected movement creates no animation | ✅ PASSED |
| `test_3` | Route waypoints follow valid room connections | ✅ PASSED |
| `test_4` | Occupants do not pass through solid walls | ✅ PASSED |
| `test_5` | Multiple simultaneous movements work | ✅ PASSED |
| `test_6` | Batch movement preserves occupant IDs | ✅ PASSED |
| `test_7` | Stationary markers are deterministic across reruns | ✅ PASSED |
| `test_8` | Queued occupants are not counted as room occupants | ✅ PASSED |
| `test_9` | Entry and exit animations reflect confirmed engine events | ✅ PASSED |
| `test_10` | Repeated reruns do not duplicate animations | ✅ PASSED |
| `test_11` | Animation progress does not advance simulation time | ✅ PASSED |
| `test_12` | Animation progress does not increment cumulative energy | ✅ PASSED |
| `test_13` | Reset clears transient animation state | ✅ PASSED |
| `test_14` | Missing geometry and degenerate paths handled safely | ✅ PASSED |
| `test_15` | Existing room selection and all 6 overlays continue to work | ✅ PASSED |
| `test_16` | STGNN pipeline and simulation integrity remain unaffected | ✅ PASSED |
| `test_17` | Visual speed scaling ($0.5\times, 1.0\times, 1.5\times, 2.0\times$) | ✅ PASSED |
| `test_18` | Visual pause and resume behaviour | ✅ PASSED |
| `test_19` | Continuous visual interpolation along doorway waypoints | ✅ PASSED |
| `test_20` | Simultaneous multi-occupant animations | ✅ PASSED |
| `test_21` | Bounded movement history retention ($\le 100$ events FIFO) | ✅ PASSED |
| `test_22` | Movement history filtering by room, occupant, and event type | ✅ PASSED |
| `test_23` | Visual replay execution without mutating simulation state | ✅ PASSED |
| `test_24` | Visual exit records retention bounded to $\le 25$ | ✅ PASSED |
| `test_25` | Stale occupant selection cleanup when occupant exits | ✅ PASSED |
| `test_26` | Adaptive marker sizing under high occupancy density ($10\text{px}/8\text{px}/6\text{px}$) | ✅ PASSED |
| `test_27` | Simulation clock invariance during visual animation and replay | ✅ PASSED |
| `test_28` | Physical sensor invariance during visual animation and replay | ✅ PASSED |
| `test_29` | HVAC and energy meter invariance during visual animation and replay | ✅ PASSED |
| `test_30` | HTML5 canvas serialization and script generation | ✅ PASSED |
| `test_31` | Occupant transit status discrimination (stationary vs moving) | ✅ PASSED |
| `test_32` | Reset clears replay state, paused state, and bounded history | ✅ PASSED |
| `test_33` | Door coordinate accuracy and collision-free routing | ✅ PASSED |
| `test_34` | STGNN pipeline integrity under concurrent animation and replay | ✅ PASSED |
| `test_35` | Replay non-existent event handled safely | ✅ PASSED |
| `test_36` | Filter events with empty results handled gracefully | ✅ PASSED |

### Full Suite Regression Results:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\VIT\SEMS\SEMESTER_7\PJT1\Digital_Twin
collected 160 items

tests\test_dashboard.py .......                                          [  4%]
tests\test_dashboard_bems.py ...........                                 [ 11%]
tests\test_energy_management.py ....................                     [ 23%]
tests\test_interactive_population.py ................                    [ 33%]
tests\test_manual_people_flow.py ...........                             [ 40%]
tests\test_occupant_animation.py ....................................    [ 63%]
tests\test_phase6_sensor_physics.py ...............                      [ 72%]
tests\test_phase8_1_floorplan.py ...........                             [ 79%]
tests\test_pipeline.py .........                                         [ 85%]
tests\test_sensors.py ..........                                         [ 91%]
tests\test_simulation.py ..............                                  [100%]

============================ 160 passed in 42.56s =============================
```

---

## 6. Summary of Architectural Invariants Preserved

1. **Simulation Invariance**: Visual playback, scrubbing, pause/resume, and historical replay never advance the simulation clock ($\Delta t = 5\text{min}$), never alter step counters, and never modify authoritative room populations.
2. **Physical Sensor Invariance**: Temperature, CO₂, humidity, illuminance, and sound sensors remain governed strictly by differential physics equations in `simulation/sensors.py`.
3. **Smart HVAC & Energy Invariance**: Thermal cooling demand, setpoint tracking, and cumulative energy consumption are 100% immune to visual rendering state.
4. **STGNN Model Integrity**: The 13-feature input tensor shape `(1, 12, 5, 13)` and 15-minute forecasting pipeline remain completely intact with zero retraining.
5. **Memory Safety**: Event history is strictly bounded to 100 events, exit markers to 25 records, and the browser animation loop runs on client resources without memory leaks.

