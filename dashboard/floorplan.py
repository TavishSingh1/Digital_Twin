"""
2D Building Floor Plan Visualizer & Overlay Engine
==================================================
Renders an interactive, geometrically accurate 2D top-down floor plan for the 5-room facility.

Key Architectural Guarantees:
1. Strict 1:1 isometric aspect ratio preserving building proportions (26m x 20m).
2. Visible room walls, distinct doorway connections, exterior main exit, and entrance queue.
3. Multi-physics environmental and energy overlay coloring:
   - Occupancy (empty, partial, moderate, near-capacity)
   - Indoor Temperature (°C)
   - CO2 Concentration (ppm)
   - Illuminance (lux)
   - HVAC Compressor Power (W)
   - Total Room Electrical Power (W)
4. Highlighted selected room focus (cyan glowing border and [SELECTED] tag).
5. Comprehensive legend metadata and threshold scales.
6. Pure presentation: zero simulation mutation or clock advancement during rendering.
7. Explicit validation rejecting invalid room geometry with clean errors.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import plotly.graph_objects as go

from simulation.building import Building
from simulation.occupant import Occupant
from dashboard.animation import (
    AnimationManager,
    get_deterministic_room_grid_position,
    calculate_route_waypoints,
)


OVERLAY_OPTIONS = [
    "Occupancy",
    "Indoor Temperature",
    "CO2 Concentration",
    "Illuminance",
    "HVAC Power",
    "Total Room Power",
]


def validate_building_geometry(building: Building):
    """
    Validates building room geometry.
    Raises ValueError if any room has invalid coordinates or non-positive dimensions.
    """
    if not building or not building.rooms:
        raise ValueError("Building does not contain any configured rooms.")

    for r_id, room in building.rooms.items():
        if room.width <= 0:
            raise ValueError(f"Room '{r_id}' has non-positive width: {room.width}")
        if room.height <= 0:
            raise ValueError(f"Room '{r_id}' has non-positive height: {room.height}")
        if room.capacity <= 0:
            raise ValueError(f"Room '{r_id}' has invalid capacity: {room.capacity}")
        if room.x < 0 or room.y < 0:
            raise ValueError(f"Room '{r_id}' has negative coordinates: ({room.x}, {room.y})")


def get_utilization_color(occupant_count: int, capacity: int) -> str:
    """Returns rgba color representing room capacity utilization."""
    if capacity <= 0:
        return "rgba(189, 195, 199, 0.35)"  # Gray
    if occupant_count == 0:
        return "rgba(100, 116, 139, 0.25)"  # Empty slate
    ratio = occupant_count / capacity
    if ratio < 0.50:
        return "rgba(46, 204, 113, 0.35)"  # Green (< 50%)
    elif ratio < 0.85:
        return "rgba(243, 156, 18, 0.40)"  # Amber (50% - 85%)
    else:
        return "rgba(231, 76, 60, 0.45)"  # Red (>= 85%)


def get_utilization_border_color(occupant_count: int, capacity: int) -> str:
    """Returns border stroke color for room boundaries."""
    if capacity <= 0:
        return "rgb(127, 140, 141)"
    if occupant_count == 0:
        return "rgb(148, 163, 184)"
    ratio = occupant_count / capacity
    if ratio < 0.50:
        return "rgb(39, 174, 96)"
    elif ratio < 0.85:
        return "rgb(211, 84, 0)"
    else:
        return "rgb(192, 57, 43)"


def get_overlay_color_and_text(
    room_id: str,
    room: Any,
    overlay: str,
    energy_states: Optional[dict[str, Any]] = None,
    sensor_readings: Optional[dict[str, Any]] = None,
) -> Tuple[str, str, str, str, str]:
    """
    Computes fill color, border color, primary label, status badge, and tooltip text
    for a room under the active overlay.
    Returns: (fill_color, border_color, primary_metric_str, status_badge, tooltip_str)
    """
    count = room.occupant_count
    cap = room.capacity
    util_pct = (count / cap * 100.0) if cap > 0 else 0.0

    # Sensor data extraction
    s_dict = (sensor_readings.get(room_id) or {}) if sensor_readings else {}
    t_in = s_dict.get("indoor_temperature")
    co2 = s_dict.get("indoor_co2")
    lux = s_dict.get("illuminance")

    # Energy data extraction
    e_state = (energy_states.get(room_id) or None) if energy_states else None
    hvac_w = getattr(e_state, "hvac_power_w", None) if e_state else None
    tot_w = getattr(e_state, "total_power_w", None) if e_state else None
    hvac_mode = getattr(e_state, "hvac_mode", "AUTO") if e_state else "AUTO"
    cum_kwh = getattr(e_state, "cumulative_total_kwh", 0.0) if e_state else 0.0

    # 1. Occupancy Overlay
    if overlay == "Occupancy":
        fill = get_utilization_color(count, cap)
        border = get_utilization_border_color(count, cap)
        metric_str = f"👥 {count} / {cap} ({util_pct:.0f}%)"
        if count == 0:
            badge = "⚪ Empty"
        elif util_pct >= 85.0:
            badge = "🔴 Near Capacity"
        elif co2 is not None and co2 > 1000.0:
            badge = "🟠 Elevated CO₂"
        elif util_pct >= 50.0:
            badge = "🟠 Moderate Load"
        else:
            badge = "🟢 Partial Load"
        tip = f"Occupancy: <b>{count}/{cap}</b> ({util_pct:.1f}%)<br>Status: <b>{badge}</b>"

    # 2. Indoor Temperature Overlay
    elif overlay == "Indoor Temperature":
        if t_in is not None:
            metric_str = f"🌡️ {t_in:.2f} °C"
            if t_in < 22.0:
                fill = "rgba(56, 189, 248, 0.35)"
                border = "rgb(14, 165, 233)"
                badge = "❄️ Cool (<22°C)"
            elif t_in <= 25.0:
                fill = "rgba(16, 185, 129, 0.35)"
                border = "rgb(5, 150, 105)"
                badge = "🟢 Comfort (22-25°C)"
            elif t_in <= 27.0:
                fill = "rgba(245, 158, 11, 0.40)"
                border = "rgb(217, 119, 6)"
                badge = "🟠 Warm (25-27°C)"
            else:
                fill = "rgba(239, 68, 68, 0.45)"
                border = "rgb(220, 38, 38)"
                badge = "🔴 Hot (>27°C)"
            tip = f"Indoor Temp: <b>{t_in:.2f} °C</b><br>Evaluation: <b>{badge}</b>"
        else:
            fill = "rgba(100, 116, 139, 0.25)"
            border = "rgb(100, 116, 139)"
            metric_str = "🌡️ Temp: Unavailable"
            badge = "⚪ No Reading"
            tip = "Indoor Temp: <i>Unavailable</i>"

    # 3. CO2 Concentration Overlay
    elif overlay == "CO2 Concentration":
        if co2 is not None:
            metric_str = f"💨 {co2:.0f} ppm"
            if co2 < 600.0:
                fill = "rgba(16, 185, 129, 0.35)"
                border = "rgb(5, 150, 105)"
                badge = "🟢 Excellent (<600 ppm)"
            elif co2 <= 800.0:
                fill = "rgba(6, 182, 212, 0.35)"
                border = "rgb(8, 145, 178)"
                badge = "🔵 Good (600-800 ppm)"
            elif co2 <= 1000.0:
                fill = "rgba(245, 158, 11, 0.40)"
                border = "rgb(217, 119, 6)"
                badge = "🟠 Elevated (800-1000 ppm)"
            else:
                fill = "rgba(239, 68, 68, 0.45)"
                border = "rgb(220, 38, 38)"
                badge = "🔴 High (>1000 ppm)"
            tip = f"Indoor CO₂: <b>{co2:.1f} ppm</b><br>Air Quality: <b>{badge}</b>"
        else:
            fill = "rgba(100, 116, 139, 0.25)"
            border = "rgb(100, 116, 139)"
            metric_str = "💨 CO₂: Unavailable"
            badge = "⚪ No Reading"
            tip = "Indoor CO₂: <i>Unavailable</i>"

    # 4. Illuminance Overlay
    elif overlay == "Illuminance":
        if lux is not None:
            metric_str = f"💡 {lux:.0f} lux"
            if lux < 100.0:
                fill = "rgba(99, 102, 241, 0.30)"
                border = "rgb(79, 70, 229)"
                badge = "🌑 Dim (<100 lx)"
            elif lux <= 300.0:
                fill = "rgba(14, 165, 233, 0.35)"
                border = "rgb(2, 132, 199)"
                badge = "🔵 Sub-target (100-300 lx)"
            elif lux <= 600.0:
                fill = "rgba(234, 179, 8, 0.40)"
                border = "rgb(202, 138, 4)"
                badge = "🟡 Target Workplane (300-600 lx)"
            else:
                fill = "rgba(249, 115, 22, 0.45)"
                border = "rgb(234, 88, 12)"
                badge = "🟠 High Daylight (>600 lx)"
            tip = f"Illuminance: <b>{lux:.1f} lux</b><br>Lighting: <b>{badge}</b>"
        else:
            fill = "rgba(100, 116, 139, 0.25)"
            border = "rgb(100, 116, 139)"
            metric_str = "💡 Lux: Unavailable"
            badge = "⚪ No Reading"
            tip = "Illuminance: <i>Unavailable</i>"

    # 5. HVAC Power Overlay
    elif overlay == "HVAC Power":
        if hvac_w is not None:
            metric_str = f"❄️ {hvac_w:.0f} W ({hvac_w/1000.0:.2f} kW)"
            if hvac_w < 300.0:
                fill = "rgba(100, 116, 139, 0.25)"
                border = "rgb(100, 116, 139)"
                badge = "⚪ Standby (<300W)"
            elif hvac_w <= 1200.0:
                fill = "rgba(16, 185, 129, 0.35)"
                border = "rgb(5, 150, 105)"
                badge = "🟢 Eco / Low Load"
            elif hvac_w <= 2500.0:
                fill = "rgba(245, 158, 11, 0.40)"
                border = "rgb(217, 119, 6)"
                badge = "🟠 Moderate Load"
            else:
                fill = "rgba(239, 68, 68, 0.45)"
                border = "rgb(220, 38, 38)"
                badge = "🔴 High Demand (>2.5 kW)"
            tip = f"HVAC Power: <b>{hvac_w:.1f} W</b><br>Mode: <b>{hvac_mode}</b> ({badge})"
        else:
            fill = "rgba(100, 116, 139, 0.25)"
            border = "rgb(100, 116, 139)"
            metric_str = "❄️ HVAC: Standby"
            badge = "⚪ Standby"
            tip = "HVAC: <i>Standby</i>"

    # 6. Total Room Power Overlay
    else:  # Total Room Power
        if tot_w is not None:
            metric_str = f"⚡ {tot_w:.0f} W ({tot_w/1000.0:.2f} kW)"
            if tot_w < 500.0:
                fill = "rgba(100, 116, 139, 0.25)"
                border = "rgb(100, 116, 139)"
                badge = "⚪ Idle (<500W)"
            elif tot_w <= 1500.0:
                fill = "rgba(16, 185, 129, 0.35)"
                border = "rgb(5, 150, 105)"
                badge = "🟢 Low Demand"
            elif tot_w <= 3000.0:
                fill = "rgba(245, 158, 11, 0.40)"
                border = "rgb(217, 119, 6)"
                badge = "🟠 Moderate Demand"
            else:
                fill = "rgba(239, 68, 68, 0.45)"
                border = "rgb(220, 38, 38)"
                badge = "🔴 Peak Demand (>3 kW)"
            tip = f"Total Power: <b>{tot_w:.1f} W</b><br>Energy: <b>{cum_kwh:.2f} kWh</b>"
        else:
            fill = "rgba(100, 116, 139, 0.25)"
            border = "rgb(100, 116, 139)"
            metric_str = "⚡ Power: N/A"
            badge = "⚪ N/A"
            tip = "Total Power: <i>Unavailable</i>"

    return fill, border, metric_str, badge, tip


def get_overlay_legend_info(overlay: str) -> dict[str, Any]:
    """Returns metadata, metric unit, description, and color ranges for the active overlay."""
    if overlay == "Occupancy":
        return {
            "name": "Room Occupancy & Capacity",
            "unit": "occupants / %",
            "desc": "Human occupant load relative to maximum room design capacity.",
            "ranges": [
                {"label": "Empty (0 occ)", "color": "#64748B", "desc": "Room vacant"},
                {"label": "Partial (<50%)", "color": "#10B981", "desc": "Normal light load"},
                {"label": "Moderate (50-85%)", "color": "#F59E0B", "desc": "Elevated density"},
                {"label": "Near Capacity (≥85%)", "color": "#EF4444", "desc": "Design capacity threshold"},
            ],
        }
    elif overlay == "Indoor Temperature":
        return {
            "name": "Indoor Dry-Bulb Temperature",
            "unit": "°C",
            "desc": "Air temperature governed by building envelope, outdoor weather, and HVAC refrigeration.",
            "ranges": [
                {"label": "Cool (<22.0°C)", "color": "#38BDF8", "desc": "Below comfort setpoint"},
                {"label": "Comfort (22.0-25.0°C)", "color": "#10B981", "desc": "Optimal thermal comfort"},
                {"label": "Warm (25.0-27.0°C)", "color": "#F59E0B", "desc": "Eco mode setback range"},
                {"label": "Hot (>27.0°C)", "color": "#EF4444", "desc": "High thermal load"},
            ],
        }
    elif overlay == "CO2 Concentration":
        return {
            "name": "Indoor Carbon Dioxide (CO₂)",
            "unit": "ppm",
            "desc": "Metabolic occupant respiration and fresh-air air change per hour (ACH) decay.",
            "ranges": [
                {"label": "Excellent (<600 ppm)", "color": "#10B981", "desc": "Near-outdoor ambient air"},
                {"label": "Good (600-800 ppm)", "color": "#06B6D4", "desc": "Healthy indoor environment"},
                {"label": "Elevated (800-1000 ppm)", "color": "#F59E0B", "desc": "Ventilation ramp-up indicator"},
                {"label": "High (>1000 ppm)", "color": "#EF4444", "desc": "ASHRAE comfort guideline limit"},
            ],
        }
    elif overlay == "Illuminance":
        return {
            "name": "Workplane Illuminance",
            "unit": "lux",
            "desc": "Natural solar daylight contribution plus artificial luminaire illuminance.",
            "ranges": [
                {"label": "Dim (<100 lux)", "color": "#6366F1", "desc": "Night or unlit room"},
                {"label": "Sub-target (100-300 lux)", "color": "#0EA5E9", "desc": "Requires artificial light"},
                {"label": "Target (300-600 lux)", "color": "#EAB308", "desc": "Optimal office task lighting"},
                {"label": "High Daylight (>600 lux)", "color": "#F97316", "desc": "Direct exterior window daylight"},
            ],
        }
    elif overlay == "HVAC Power":
        return {
            "name": "HVAC Compressor Electrical Power",
            "unit": "W",
            "desc": "Compressor electrical draw maintaining room comfort setpoints via COP 3.2 refrigeration.",
            "ranges": [
                {"label": "Standby (<300 W)", "color": "#64748B", "desc": "Idling / within deadband"},
                {"label": "Eco (<1200 W)", "color": "#10B981", "desc": "Low modulation cooling"},
                {"label": "Moderate (1200-2500 W)", "color": "#F59E0B", "desc": "Active heat extraction"},
                {"label": "High Demand (>2500 W)", "color": "#EF4444", "desc": "Near maximum cooling capacity"},
            ],
        }
    else:  # Total Room Power
        return {
            "name": "Total Room Electrical Demand",
            "unit": "W",
            "desc": "Sum of HVAC refrigeration, daylight-harvested lighting, and auxiliary base loads.",
            "ranges": [
                {"label": "Idle (<500 W)", "color": "#64748B", "desc": "Base auxiliary load only"},
                {"label": "Low (500-1500 W)", "color": "#10B981", "desc": "Eco mode / low occupancy"},
                {"label": "Moderate (1500-3000 W)", "color": "#F59E0B", "desc": "Standard business operation"},
                {"label": "Peak (>3000 W)", "color": "#EF4444", "desc": "High cooling and luminaire load"},
            ],
        }


def create_floorplan_figure(
    building: Building,
    occupants: Optional[dict[str, Occupant]] = None,
    show_occupants: bool = True,
    dark_mode: bool = True,
    entrance_queue_count: int = 0,
    energy_states: Optional[dict[str, Any]] = None,
    sensor_readings: Optional[dict[str, Any]] = None,
    selected_room_id: Optional[str] = None,
    active_overlay: str = "Occupancy",
    animation_manager: Optional[AnimationManager] = None,
    show_paths: bool = True,
    selected_occupant_id: Optional[str] = None,
) -> go.Figure:
    """
    Renders an interactive, geometrically verified 2D floor plan of the facility.
    Parameters:
        building: Building instance with rooms and door connections.
        occupants: Dictionary of Occupant instances.
        show_occupants: Whether to plot individual occupant markers.
        dark_mode: Whether to style figure for dark UI themes.
        entrance_queue_count: Number of occupants waiting in entrance queue.
        energy_states: Optional dictionary mapping room_id to RoomEnergyState.
        sensor_readings: Optional dictionary mapping room_id to latest sensor dict.
        selected_room_id: Optional ID of the currently selected room for highlight.
        active_overlay: Active physical metric overlay ("Occupancy", "Indoor Temperature", etc.).
        animation_manager: Optional AnimationManager providing waypoint routes, interpolation & deduplication.
        show_paths: Whether to render trajectory paths through doorways.
        selected_occupant_id: Optional ID of occupant to highlight with golden halo and planned route.
    """
    # 0. Validate building geometry
    validate_building_geometry(building)

    fig = go.Figure()

    # Room bounding polygons and labels
    for room in building.rooms.values():
        x0, y0 = room.x, room.y
        x1, y1 = room.x + room.width, room.y + room.height
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

        count = room.occupant_count
        cap = room.capacity
        util_pct = (count / cap * 100.0) if cap > 0 else 0.0

        is_selected = (selected_room_id is not None) and (
            room.room_id == selected_room_id or room.room_id.replace(" ", "_") == selected_room_id.replace(" ", "_")
        )

        # Compute overlay-driven colors and labels
        fill_color, border_color, primary_metric, badge, overlay_tip = get_overlay_color_and_text(
            room_id=room.room_id,
            room=room,
            overlay=active_overlay,
            energy_states=energy_states,
            sensor_readings=sensor_readings,
        )

        if is_selected:
            final_border = "#00F0FF"  # High-contrast cyan selection highlight
            final_width = 5
        else:
            final_border = border_color
            final_width = 3

        # 1. Add room boundary shape
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            fillcolor=fill_color,
            line=dict(color=final_border, width=final_width),
            layer="below",
        )

        # 2. Add invisible hover scatter for room tooltip
        room_info = (
            f"<b>{room.room_id}</b>: {room.name}<br>"
            f"Dimensions: {room.width:.1f}m × {room.height:.1f}m<br>"
            f"Occupancy: <b>{count} / {cap}</b> ({util_pct:.1f}%)<br>"
            f"Overlay ({active_overlay}): {overlay_tip}<br>"
            f"Doors: {len(room.doors)}"
        )

        # If sensor readings present, include quick snapshot in tooltip
        if sensor_readings and room.room_id in sensor_readings:
            s = sensor_readings[room.room_id]
            t_val = s.get("indoor_temperature")
            co2_val = s.get("indoor_co2")
            if t_val is not None and co2_val is not None:
                room_info += f"<br>Temp: <b>{t_val:.1f}°C</b> | CO₂: <b>{co2_val:.0f} ppm</b>"

        # Build clean room card text annotation (Phase 8.4 Blueprint Simplification)
        prefix = "<span style='color:#00F0FF; font-weight:bold;'>[SELECTED]</span><br>" if is_selected else ""
        display_text = f"{prefix}<b>{room.room_id}</b><br><span style='font-size:10px;'>{room.name}</span><br><b>{primary_metric}</b>"

        # Show concise sensor reading and status only for the single selected room
        if is_selected:
            if active_overlay not in ["Indoor Temperature", "CO2 Concentration"]:
                if sensor_readings and room.room_id in sensor_readings:
                    s = sensor_readings[room.room_id]
                    t_in = s.get("indoor_temperature")
                    co2 = s.get("indoor_co2")
                    if t_in is not None and co2 is not None:
                        display_text += f"<br><span style='font-size:10px;'>🌡️ {t_in:.1f}°C | 💨 {co2:.0f} ppm</span>"
            if badge and "Elevated" in badge:
                display_text += f"<br><span style='font-size:9px; color:#F59E0B;'>Status: {badge}</span>"

        fig.add_trace(
            go.Scatter(
                x=[cx], y=[cy],
                mode="text",
                text=[display_text],
                textposition="middle center",
                textfont=dict(
                    family="sans-serif",
                    size=11,
                    color="#FFFFFF" if dark_mode else "#1A1A1A",
                ),
                hoverinfo="text",
                hovertext=[room_info],
                showlegend=False,
            )
        )

    # 3. Add interior door connections and door opening lines
    door_xs = []
    door_ys = []
    door_texts = []
    for (r1, r2), pos in building.door_positions.items():
        door_xs.append(pos[0])
        door_ys.append(pos[1])
        door_texts.append(f"Interior Doorway: {r1} ↔ {r2}<br>Clear passage width: 1.2m")

    if door_xs:
        fig.add_trace(
            go.Scatter(
                x=door_xs, y=door_ys,
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=13,
                    color="#F1C40F",  # Amber doorway marker
                    line=dict(color="#000000", width=1.5),
                ),
                name="Interior Doors",
                hoverinfo="text",
                hovertext=door_texts,
            )
        )

    # 4. Add exterior exit marker (Room 1)
    r1 = building.get_room("Room 1")
    if r1 and r1.has_exterior_exit and r1.exterior_door_position:
        ex, ey = r1.exterior_door_position
        fig.add_trace(
            go.Scatter(
                x=[ex], y=[ey],
                mode="markers+text",
                text=["MAIN EXIT"],
                textposition="bottom center",
                textfont=dict(size=10, color="#E74C3C", family="sans-serif"),
                marker=dict(
                    symbol="triangle-down",
                    size=16,
                    color="#E74C3C",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                name="Exterior Exit",
                hoverinfo="text",
                hovertext=["Building Main Entry/Exit Door (Room 1 Gateway)"],
            )
        )

    # 5. Add entrance queue indicator if occupants are waiting
    queue_count = entrance_queue_count
    if queue_count == 0 and occupants:
        queue_count = sum(1 for o in occupants.values() if o.movement_state == "waiting_at_entrance")

    if queue_count > 0:
        fig.add_trace(
            go.Scatter(
                x=[7.0],
                y=[-1.6],
                mode="markers+text",
                text=[f"WAITING QUEUE: {queue_count}"],
                textposition="bottom center",
                textfont=dict(size=11, color="#E67E22", family="sans-serif"),
                marker=dict(
                    symbol="hexagon",
                    size=16,
                    color="#E67E22",
                    line=dict(color="#FFFFFF", width=2),
                ),
                name="Entrance Queue",
                hoverinfo="text",
                hovertext=[
                    f"<b>Entrance Waiting Queue</b><br>"
                    f"Occupants waiting: <b>{queue_count}</b><br>"
                    f"Room 1 capacity is full. Admitted FIFO as capacity frees."
                ],
            )
        )

    # 6. Add individual occupant markers
    if show_occupants and occupants:
        occ_xs = []
        occ_ys = []
        occ_texts = []
        occ_colors = []
        occ_sizes = []

        type_color_map = {
            "regular": "#3498DB",  # Blue
            "visitor": "#9B59B6",  # Purple
            "staff": "#1ABC9C",    # Teal
        }

        # Deterministic positioning for stationary and moving occupants
        if animation_manager is not None:
            positions = animation_manager.get_occupant_positions(building, occupants)
        else:
            # Fallback deterministic grid positioning
            positions = {}
            room_occupants: dict[str, list[str]] = {r: [] for r in building.rooms.keys()}
            for pid, occ in sorted(occupants.items()):
                if occ.is_inside_building and occ.current_room_id in room_occupants:
                    room_occupants[occ.current_room_id].append(pid)
            for r_id, pids in room_occupants.items():
                rm = building.get_room(r_id)
                total = len(pids)
                for idx, pid in enumerate(pids):
                    positions[pid] = get_deterministic_room_grid_position(pid, rm, idx, total)

        for occ_id, occ in sorted(occupants.items()):
            if occ.is_inside_building and occ.current_room_id:
                x, y = positions.get(occ.occupant_id, occ.position)
                occ_xs.append(x)
                occ_ys.append(y)
                # Adaptive marker size for high occupant density readability
                rm_pop = len([o for o in occupants.values() if o.is_inside_building and o.current_room_id == occ.current_room_id])
                if rm_pop > 30:
                    m_size = 6
                elif rm_pop > 15:
                    m_size = 8
                else:
                    m_size = 10
                occ_sizes.append(m_size)
                occ_colors.append(type_color_map.get(occ.occupant_type, "#3498DB"))
                occ_texts.append(
                    f"Occupant: <b>{occ.occupant_id}</b><br>"
                    f"Type: {occ.occupant_type.capitalize()}<br>"
                    f"Room: {occ.current_room_id}<br>"
                    f"Activity: {occ.current_activity}<br>"
                    f"State: {occ.movement_state}"
                )

        if occ_xs:
            fig.add_trace(
                go.Scatter(
                    x=occ_xs, y=occ_ys,
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=occ_sizes,
                        color=occ_colors,
                        opacity=0.85,
                        line=dict(color="#FFFFFF", width=1),
                    ),
                    name="Occupants",
                    hoverinfo="text",
                    hovertext=occ_texts,
                )
            )

        # 6a. Highlight selected occupant (if specified and inside building)
        if selected_occupant_id and selected_occupant_id in occupants:
            sel_occ = occupants[selected_occupant_id]
            if sel_occ.is_inside_building:
                sel_x, sel_y = positions.get(selected_occupant_id, sel_occ.position)
                fig.add_trace(
                    go.Scatter(
                        x=[sel_x], y=[sel_y],
                        mode="markers+text",
                        text=[f"★ {selected_occupant_id}"],
                        textposition="top center",
                        textfont=dict(color="#F1C40F", size=12, family="sans-serif"),
                        marker=dict(
                            symbol="circle",
                            size=20,
                            color="rgba(241, 196, 15, 0.4)",
                            line=dict(color="#F1C40F", width=3),
                        ),
                        name=f"Selected ({selected_occupant_id})",
                        hoverinfo="text",
                        hovertext=[
                            f"<b>Selected Occupant: {selected_occupant_id}</b><br>"
                            f"Room: {sel_occ.current_room_id}<br>"
                            f"Activity: {sel_occ.current_activity}<br>"
                            f"State: {sel_occ.movement_state}<br>"
                            f"Destination: {sel_occ.intended_destination or 'None'}"
                        ],
                    )
                )

                # Route waypoints for selected occupant
                if animation_manager and sel_occ.intended_destination and sel_occ.intended_destination != sel_occ.current_room_id:
                    sel_route = animation_manager.get_selected_route_waypoints(building, sel_occ)
                    if len(sel_route) >= 2:
                        rx = [p[0] for p in sel_route]
                        ry = [p[1] for p in sel_route]
                        fig.add_trace(
                            go.Scatter(
                                x=rx, y=ry,
                                mode="lines+markers",
                                line=dict(color="#F1C40F", width=3, dash="dot"),
                                marker=dict(size=7, symbol="diamond", color="#F1C40F"),
                                name=f"Route: {selected_occupant_id}",
                                hoverinfo="text",
                                hovertext=[f"Planned Path: {sel_occ.current_room_id} → {sel_occ.intended_destination}"] * len(rx),
                            )
                        )

    # 6b. Movement Trajectories (Clean waypoint visualization for active or latest event)
    if show_paths and animation_manager and animation_manager.events:
        active_replay = animation_manager.get_active_replay_event()
        events_to_show = [active_replay] if active_replay else [animation_manager.events[-1]]
        for ev in events_to_show:
            if len(ev.waypoints) >= 2:
                wx = [p[0] for p in ev.waypoints]
                wy = [p[1] for p in ev.waypoints]
                if ev.event_type == "admission":
                    p_color = "#2ECC71"  # Emerald green
                elif ev.event_type == "exit":
                    p_color = "#E74C3C"  # Crimson red
                else:
                    p_color = "#3498DB"  # Sky blue
                fig.add_trace(
                    go.Scatter(
                        x=wx, y=wy,
                        mode="lines+markers",
                        line=dict(color=p_color, width=2.5, dash="dash"),
                        marker=dict(size=5, symbol="circle", color=p_color),
                        name=f"Path: {ev.source_room or 'Outside'} → {ev.destination_room or 'Exit'}",
                        hoverinfo="text",
                        hovertext=[
                            f"<b>Movement Event ({ev.event_type.upper()})</b><br>"
                            f"Occupants: {', '.join(ev.person_ids)}<br>"
                            f"Route: {ev.source_room or 'Outside'} → {ev.destination_room or 'Exit'}<br>"
                            f"Timestamp: {ev.timestamp}"
                        ] * len(wx),
                        showlegend=False,
                    )
                )

    # 6c. Transient visual exit records (outside building, minimal markers)
    if animation_manager and animation_manager.exit_records:
        recent_exits = animation_manager.exit_records[-3:]
        exit_xs = []
        exit_ys = []
        exit_texts = []
        for i, rec in enumerate(recent_exits):
            off_x = (i - 1) * 0.4
            off_y = -0.4
            exit_xs.append(rec["position"][0] + off_x)
            exit_ys.append(rec["position"][1] + off_y)
            exit_texts.append(
                f"<b>Visual Exit Record</b> (Not an active occupant)<br>"
                f"ID: <b>{rec['occupant_id']}</b><br>"
                f"Departed from: {rec['from_room']}<br>"
                f"Time: {rec['timestamp']}"
            )
        if exit_xs:
            fig.add_trace(
                go.Scatter(
                    x=exit_xs, y=exit_ys,
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=8,
                        color="#E74C3C",
                        line=dict(color="#C0392B", width=1),
                    ),
                    name="Visual Exits (Outside)",
                    hoverinfo="text",
                    hovertext=exit_texts,
                    showlegend=False,
                )
            )

    # 7. Coordinate axes & strict 1:1 isometric aspect ratio
    fig.update_layout(
        title=dict(
            text=f"2D Facility Layout — Active Overlay: {active_overlay}",
            font=dict(size=14, color="#ECEFF1"),
        ),
        xaxis=dict(
            title="Building Width (meters)",
            range=[-2, 28],
            showgrid=True,
            zeroline=False,
            gridcolor="#2C3E50" if dark_mode else "#E0E0E0",
            scaleanchor="y",
            scaleratio=1,  # Strict 1:1 aspect ratio ensures zero geometric distortion
        ),
        yaxis=dict(
            title="Building Height (meters)",
            range=[-3, 22],
            showgrid=True,
            zeroline=False,
            gridcolor="#2C3E50" if dark_mode else "#E0E0E0",
        ),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=20, r=20, t=35, b=20),
        height=490,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )

    return fig
