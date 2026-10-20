# Phase 8.1 Audit Report: Interactive Building Floor Plan and Room Selection

## 1. Executive Summary & Objectives

The goal of Phase 8.1 was to upgrade the five-room 2D building visualization into an authoritative, geometrically accurate, multi-physics interactive floor plan.

Key achievements in Phase 8.1:
1. **Authoritative Building Geometry**: Verified strict mapping to `config/building.json` with a 1:1 isometric aspect ratio ($26\text{m} \times 20\text{m}$ envelope), preventing visual distortion.
2. **Doorways & Wall Boundaries**: Clear differentiation between solid wall boundaries, interior connecting doorways (width $1.2\text{m}$), the exterior main exit/entrance at Room 1, and the active entrance waiting queue.
3. **First-Class Room Selection**: Selection highlights room boundaries in bright cyan (`#00F0FF`, border width 5), marks the room with `★ [SELECTED]`, and synchronizes the contextual **Selected Room Control Centre** and room summary cards.
4. **Multi-Physics Environmental & Energy Overlays**:
   - 👥 **Occupancy**: Empty (0 occ), Partial (<50%), Moderate (50-85%), Near-Capacity (≥85%).
   - 🌡️ **Indoor Temperature**: Cool (<22.0°C), Comfort (22.0-25.0°C), Warm (25.0-27.0°C), Hot (>27.0°C).
   - 💨 **CO₂ Concentration**: Excellent (<600 ppm), Good (600-800 ppm), Elevated (800-1000 ppm), High (>1000 ppm limit).
   - 💡 **Illuminance**: Dim (<100 lux), Sub-target (100-300 lux), Target Workplane (300-600 lux), High Daylight (>600 lux).
   - ❄️ **HVAC Power**: Standby (<300 W), Eco/Low (300-1200 W), Moderate (1200-2500 W), High Demand (>2500 W).
   - ⚡ **Total Room Power**: Idle (<500 W), Low (500-1500 W), Moderate (1500-3000 W), Peak (>3000 W).
5. **Pure Presentation Invariance**: Overlay switching and floor plan rendering strictly mutate zero simulation state: clock, step count, occupant positions, sensor history, and energy integration meters are completely untouched.
6. **Geometry Validation**: Explicit check (`validate_building_geometry`) rejecting corrupted, non-positive, or negative dimensions with clean `ValueError` exceptions.
7. **Comprehensive Test Suite**: 124 passing tests across 10 test modules (100% pass rate).

---

## 2. Geometry Source & Coordinate Reference

All physical coordinates and bounding dimensions originate directly from `config/building.json`:

| Room ID | Canonical Name | X (m) | Y (m) | Width (m) | Height (m) | Area ($m^2$) | Capacity | Exterior Door |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Room 1** | Main Conference / Lobby | 0.0 | 0.0 | 14.0 | 10.0 | 140.0 | 40 | Yes [7.0, 0.0] |
| **Room 2** | Open Office A | 14.0 | 0.0 | 12.0 | 10.0 | 120.0 | 25 | No |
| **Room 3** | Executive Office Suite | 0.0 | 10.0 | 8.0 | 10.0 | 80.0 | 15 | No |
| **Room 4** | Open Office B | 8.0 | 10.0 | 10.0 | 10.0 | 100.0 | 20 | No |
| **Room 5** | Cafeteria / Lounge | 18.0 | 10.0 | 8.0 | 10.0 | 80.0 | 20 | No |

### Interior Door Connections ($1.2\text{m}$ width):
1. **Room 1 ↔ Room 2**: Door position `[14.0, 5.0]`
2. **Room 1 ↔ Room 3**: Door position `[4.0, 10.0]`
3. **Room 2 ↔ Room 4**: Door position `[14.0, 10.0]`
4. **Room 3 ↔ Room 4**: Door position `[8.0, 15.0]`
5. **Room 4 ↔ Room 5**: Door position `[18.0, 15.0]`
6. **Room 2 ↔ Room 5**: Door position `[22.0, 10.0]`

---

## 3. Files Created & Modified

| File | Change Description |
| :--- | :--- |
| `dashboard/floorplan.py` | Added geometry validation (`validate_building_geometry`), 6 multi-physics overlays with threshold scales (`get_overlay_color_and_text`, `get_overlay_legend_info`), strict 1:1 aspect ratio (`scaleanchor="y"`, `scaleratio=1`), and selected room highlight. |
| `dashboard/overview.py` | Added interactive overlay selector, rich colored legend banner, and cross-room physical/energy comparison table. |
| `dashboard/state.py` | Added persistent session state helpers: `get_selected_overlay()`, `set_selected_overlay()`. |
| `dashboard/__init__.py` | Exported `OVERLAY_OPTIONS`, `get_overlay_legend_info`, `validate_building_geometry`. |
| `tests/test_phase8_1_floorplan.py` | Created automated test suite covering all 12 Phase 8.1 requirements (11 tests). |
| `audit/phase8_1_floorplan_report.md` | Comprehensive Phase 8.1 audit and completion report. |

---

## 4. Test Verification & Results

### Baseline Before Phase 8.1:
- 113 passed tests across 9 test files.

### New Phase 8.1 Test Suite (`tests/test_phase8_1_floorplan.py`):
1. `test_all_five_configured_rooms_render`: 5 bounding shapes, 1:1 isometric aspect ratio.
2. `test_room_names_and_capacities_match_config`: Name and capacity labels match `config/building.json`.
3. `test_room_selection_highlight`: Border color `#00F0FF`, border width 5, `[SELECTED]` tag.
4. `test_selection_persists_across_state`: Session state retains room and overlay selections.
5. `test_all_overlays_render_cleanly`: All 6 overlays generate valid figures and complete legends.
6. `test_overlay_switching_preserves_simulation`: Step count, logical clock, and energy are untouched.
7. `test_missing_telemetry_graceful_handling`: Missing readings render clean `N/A` without exceptions.
8. `test_occupancy_and_power_match_simulation`: Rendered metrics match live simulation state.
9. `test_invalid_geometry_raises_value_error`: Non-positive dimensions and negative coordinates raise `ValueError`.
10. `test_entrance_queue_and_doorways`: Doors and entrance queue traces render accurately.
11. `test_stgnn_pipeline_unaffected_by_rendering`: Rolling buffer and prediction logs are invariant.

### Full Test Suite Execution:
- Total tests: **124 passed in 40.5s** (100% passing across all 10 test modules).

---

## 5. Known Limitations & Next Steps

1. **Static Occupant Markers**: In accordance with Phase 8.1 instructions, occupant positions are rendered cleanly as static or dispersed markers. Animation of step transitions is deferred to Phase 8.2 once floor plan ergonomics are verified.
2. **Door Open/Close State**: Doors are currently rendered as static spatial passageways with clear widths ($1.2\text{m}$). Dynamic door swing animations can be integrated alongside occupant movement path traces.

