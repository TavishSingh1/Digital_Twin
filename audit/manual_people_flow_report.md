# Manual People Flow Control & Movement Audit Report

**Status**: VERIFIED & COMPLETE  
**Date**: 2026-10-16  
**Scope**: Interactive manual people flow control, direct doorway topology enforcement, capacity validation, immediate sensor synchronization, and automated test verification.  
**Total Automated Tests**: 67 passing (11 new flow control tests + 56 existing regression tests).  

---

## 1. Executive Summary

A fully operational **Manual People Flow Control** system has been integrated into the 5-room building digital twin simulation and interactive dashboard.

Key capabilities delivered:
1. **Interactive Movement Interface**: Added a dedicated dashboard tab titled **People Flow Control** with selectable operational modes:
   - **Manual**: Direct, real-time command of occupant transfers between connected rooms.
   - **Scheduled**: Standard circadian schedule execution where occupants follow behavioral goals.
   - **Automatic**: Multi-step automated background simulation.
2. **Topological Doorway Enforcement**: Movement strictly checks the physical room connectivity configured in `config/building.json`. Occupants cannot pass through solid walls in one step. When non-adjacent rooms are selected (e.g. Room 1 to Room 4), the interface and engine reject the movement and explain the intermediate room required (e.g. Room 2 or Room 3).
3. **Destination Capacity Verification**: Movement validates that the destination room has sufficient capacity before moving occupants. If full or partially full, the exact available spots and capacity limits are reported.
4. **Occupant Exclusivity & State Integrity**: Occupants are atomically transferred between rooms. An occupant can never exist in two rooms simultaneously, nor be lost or duplicated.
5. **Immediate Sensor Telemetry Recalculation**: Acoustic sound pressure levels ($L_p$) and lighting recalculate immediately in both the source and destination rooms via logarithmic decibel power summation, while continuous thermodynamic variables (temperature, CO₂, humidity) evolve dynamically over subsequent 5-minute ticks.
6. **Rolling Lookback Buffer Synchronization**: The rolling buffer in `DigitalTwinPipeline` updates immediately after movement, ensuring subsequent STGNN model input tensors of shape `(1, 12, 5, 13)` reflect the updated state.
7. **Test Coverage**: 11 new automated test cases in `tests/test_manual_people_flow.py` pass, achieving 67/67 passing tests building-wide.

---

## 2. Room Connectivity & Topology

From `config/building.json`, the building's physical doorway adjacency graph is:

```mermaid
flowchart LR
    R1["Room 1 (Capacity: 40)<br>Main Entrance / Lobby"] <-->|Door (14.0, 5.0)| R2["Room 2 (Capacity: 25)<br>Open Office A"]
    R1 <-->|Door (4.0, 10.0)| R3["Room 3 (Capacity: 15)<br>Executive Suite"]
    R2 <-->|Door (14.0, 10.0)| R4["Room 4 (Capacity: 20)<br>Open Office B"]
    R2 <-->|Door (22.0, 10.0)| R5["Room 5 (Capacity: 20)<br>Cafeteria / Lounge"]
    R3 <-->|Door (8.0, 15.0)| R4
    R4 <-->|Door (18.0, 15.0)| R5
```

### Direct Doorway Matrix
| Source | Directly Connected Destinations | Disconnected Destinations (Requires Intermediate Room) |
| :--- | :--- | :--- |
| **Room 1** | Room 2, Room 3 | Room 4 (via Room 2 or 3), Room 5 (via Room 2) |
| **Room 2** | Room 1, Room 4, Room 5 | Room 3 (via Room 1 or 4) |
| **Room 3** | Room 1, Room 4 | Room 2 (via Room 1 or 4), Room 5 (via Room 4) |
| **Room 4** | Room 2, Room 3, Room 5 | Room 1 (via Room 2 or 3) |
| **Room 5** | Room 2, Room 4 | Room 1 (via Room 2), Room 3 (via Room 4) |

---

## 3. Engine Implementation (`SimulationEngine.move_people`)

The `move_people` method in `simulation/engine.py` executes with strict validation:

1. **Existence & Distinction**: Checks that `source_room != destination_room` and both rooms exist in `self.building.rooms`.
2. **Door Adjacency**: Checks `destination_room in self.building.adjacency[source_room]`. If false, uses `find_path()` to find the shortest BFS route and returns a detailed message specifying the intermediate room required.
3. **Occupant Selection**:
   - By specific IDs: Verifies each ID exists in `self.occupants` and is currently inside `source_room`.
   - By count: Validates `count <= src.occupant_count` and deterministically takes the first `count` occupants.
4. **Capacity Enforcement**: Validates `len(targets) <= (dst.capacity - dst.occupant_count)`.
5. **Atomic State Transition**:
   - `src.remove_occupant(pid)`
   - `dst.add_occupant(pid)`
   - `occ.current_room_id = destination_room`
   - `occ.position = dst.clamp_position(door_pos[0], door_pos[1])`
6. **Telemetry & Ground Truth**:
   - Calls `sensor_manager.refresh_immediate_sensors()`
   - Records outgoing and incoming transitions in ground truth
   - Conserves cumulative arrivals and exits (`self.total_arrivals` and `self.total_exits` remain unchanged).

---

## 4. Dashboard User Interface (`app.py`)

In the **People Flow Control** tab:
1. **Mode Selector**: Choose between `Manual`, `Scheduled`, and `Automatic`.
2. **Source Room Selector**: Displays current occupancy, capacity, and an interactive multi-select list of all individual occupant IDs currently in the source room.
3. **Number to Move**: User can pick specific IDs or set a numeric quantity.
4. **Destination Room Selector**: Displays destination capacity and real-time validation:
   - Green banner if direct doorway exists and capacity is available.
   - Red banner if non-adjacent, explaining which intermediate room to use.
   - Red banner if destination capacity is insufficient.
5. **Move People Button**: Dynamically enabled only when the proposed move is valid. Clicking executes the transfer, displays a success toast/alert with moved IDs, and calls `st.rerun()` to refresh the floor plan, room statistics, and live sensors.

---

## 5. Automated Test Suite Results

A dedicated automated test suite (`tests/test_manual_people_flow.py`) was created with 11 test cases:

| Test ID | Test Function | Verified Behavior | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_move_single_person_between_connected_rooms` | Moves 1 person from Room 1 to Room 2; updates counts and locations | **PASS** |
| 2 | `test_move_selected_group_of_people` | Moves specific IDs (`P0002`, `P0004`, `P0007`); checks correct relocation | **PASS** |
| 3 | `test_reject_movement_between_unconnected_rooms` | Rejects Room 1 -> Room 4 move, explaining intermediate room needed | **PASS** |
| 4 | `test_reject_movement_into_full_room` | Rejects move into full Room 3 (15/15); state remains untouched | **PASS** |
| 5 | `test_reject_nonexistent_occupant_ids` | Rejects invalid IDs (`P9999`) and occupants not in source room | **PASS** |
| 6 | `test_occupant_exclusivity_no_dual_residency` | Verifies disjoint occupant sets across all rooms (no dual presence) | **PASS** |
| 7 | `test_updating_source_and_destination_occupancy` | Multi-hop transfer sequence verifies exact room occupancy conservation | **PASS** |
| 8 | `test_sensor_state_updates_after_movement` | Source SPL drops, destination SPL jumps immediately upon occupant move | **PASS** |
| 9 | `test_preserving_arrivals_and_exits` | Internal transfers do not alter building-wide cumulative arrival/exit metrics | **PASS** |
| 10 | `test_reject_same_source_and_destination` | Rejects movement from a room to itself | **PASS** |
| 11 | `test_pipeline_integration_and_stgnn_lookback` | Moving occupants updates rolling buffer and feeds valid tensor to STGNN | **PASS** |

### Complete Test Suite Summary
```text
tests/test_dashboard.py .............. [ 7 passed ]
tests/test_interactive_population.py . [ 16 passed ]
tests/test_manual_people_flow.py ..... [ 11 passed ]
tests/test_pipeline.py ............... [ 9 passed ]
tests/test_sensors.py ................ [ 10 passed ]
tests/test_simulation.py ............. [ 14 passed ]
======================= 67 passed in 31.73s =======================
```

---

## 6. Safety Compliance & Boundary Verification

- **Model Integrity**: `joint_lstm_gnn_all_5_rooms.keras` was not modified or retrained.
- **Artifacts**: `artifacts/scalers.joblib` and raw CSV datasets in `dataset/` remain read-only and unmodified.
- **Inference Invariants**: Input tensor shape `(1, 12, 5, 13)` and 15-minute forecasting horizon are strictly preserved.

