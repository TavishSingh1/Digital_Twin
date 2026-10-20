"""
SVG-Based Interactive Building Floor Plan Engine (Pixel-Scale Vector Architecture)
==================================================================================
High-performance, crisp vector floor plan for the 5-room cyber-physical facility.

Key Architectural Guarantees:
1. Strict 1:1 isometric proportions (26m x 20m building bounds) via viewBox.
2. Crisp room boundaries, interior doorways, exterior exit, and entrance queue.
3. Multi-physics overlay shading (Occupancy, Temperature, CO2, Illuminance, Power).
4. High-contrast selected room highlight with glowing cyan border and badge.
5. Strict visual decluttering: room name, occupancy ratio, and subtle status only.
   (NO wall of raw sensor telemetry covering every room).
6. Deterministic occupant placement at authoritative simulation coordinates.
7. Pure presentation: zero mutations to simulation clock, sensors, or energy.
8. Interactive room selection support with native Streamlit callbacks.
"""

import html
import math
from typing import Any, Dict, List, Optional, Tuple

from simulation.building import Building
from simulation.occupant import Occupant
from dashboard.animation import AnimationManager, get_deterministic_room_grid_position
from dashboard.floorplan import get_overlay_color_and_text

# Coordinate scale factor: 40 pixels per meter
SCALE = 40.0
BUILDING_WIDTH_M = 26.0
BUILDING_HEIGHT_M = 20.0


def to_svg_x(sim_x: float) -> float:
    return sim_x * SCALE


def to_svg_y(sim_y: float) -> float:
    return (BUILDING_HEIGHT_M - sim_y) * SCALE


def generate_svg_floorplan_markup(
    building: Building,
    occupants: Optional[Dict[str, Occupant]] = None,
    selected_room_id: Optional[str] = "Room 1",
    active_overlay: str = "Occupancy",
    sensor_readings: Optional[Dict[str, Any]] = None,
    energy_states: Optional[Dict[str, Any]] = None,
    animation_manager: Optional[AnimationManager] = None,
    show_occupants: bool = True,
    selected_occupant_id: Optional[str] = None,
    entrance_queue_count: int = 0,
    container_id: str = "bems-svg-floorplan",
) -> str:
    """
    Generates a standards-compliant, responsive SVG string representing the 5-room building.
    Building dimensions: 26m wide x 20m high (1040px x 800px at SCALE=40).
    ViewBox: -60 -40 1160 980 (preserves exact 1:1 isometric aspect ratio).
    """
    svg_elements = []

    # 1. Defs: Glow filters and marker styles
    defs = """
  <defs>
    <!-- Cyan glow for selected room -->
    <filter id="cyan-glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    <!-- Gold glow for selected occupant -->
    <filter id="gold-glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    <!-- HVAC Supply Airflow Arrowhead Markers -->
    <marker id="hvac-arrow-cooling" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#00F0FF" />
    </marker>
    <marker id="hvac-arrow-idle" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B" />
    </marker>
    <marker id="hvac-arrow-return" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94A3B8" />
    </marker>
  </defs>
"""
    svg_elements.append(defs)

    # 2. Architectural 2m Grid
    grid_lines = []
    for gx in range(0, 28, 2):
        x_px = to_svg_x(gx)
        grid_lines.append(f'<line x1="{x_px}" y1="0" x2="{x_px}" y2="{to_svg_y(0)}" stroke="#1E293B" stroke-width="1.5" stroke-dasharray="6,6" />')
    for gy in range(0, 22, 2):
        y_px = to_svg_y(gy)
        grid_lines.append(f'<line x1="0" y1="{y_px}" x2="{to_svg_x(26)}" y2="{y_px}" stroke="#1E293B" stroke-width="1.5" stroke-dasharray="6,6" />')

    svg_elements.append(f"""
  <!-- Architectural Grid -->
  <g class="bems-grid" opacity="0.6">
    {''.join(grid_lines)}
  </g>
""")

    # 3. Room Polygons, Fill Overlays, and Clean Typography
    norm_sel = selected_room_id.replace(" ", "_") if selected_room_id else ""

    for r_id, room in building.rooms.items():
        is_sel = (r_id == selected_room_id or r_id.replace(" ", "_") == norm_sel)

        # SVG bounding rect for room
        rx = to_svg_x(room.x)
        ry = to_svg_y(room.y + room.height)
        rw = room.width * SCALE
        rh = room.height * SCALE
        cx = rx + rw / 2.0
        cy = ry + rh / 2.0

        fill_col, border_col, metric_str, badge_str, tip_str = get_overlay_color_and_text(
            room_id=r_id,
            room=room,
            overlay=active_overlay,
            energy_states=energy_states,
            sensor_readings=sensor_readings,
        )

        if is_sel:
            stroke_color = "#00F0FF"
            stroke_width = "6"
            filter_attr = 'filter="url(#cyan-glow)"'
            bg_fill = "rgba(0, 240, 255, 0.12)"
        else:
            stroke_color = border_col
            stroke_width = "2.5"
            filter_attr = ""
            bg_fill = fill_col

        count = room.occupant_count
        cap = room.capacity
        util_pct = (count / cap * 100.0) if cap > 0 else 0.0

        if count == 0:
            status_text = "Empty"
            status_color = "#64748B"
        elif util_pct >= 85.0:
            status_text = "High Load"
            status_color = "#EF4444"
        else:
            status_text = "Normal"
            status_color = "#10B981"

        room_title = f"{r_id}: {room.name}"
        room_tooltip = (
            f"{room_title}\n"
            f"Occupancy: {count} / {cap} ({util_pct:.0f}%)\n"
            f"Status: {status_text} | Overlay: {active_overlay} ({metric_str})"
        )

        sel_badge = ' <tspan fill="#00F0FF" font-weight="900">★</tspan>' if is_sel else ""

        svg_elements.append(f"""
  <!-- Room: {r_id} -->
  <a href="?selected_room={r_id.replace(' ', '+')}" target="_parent" style="text-decoration:none; cursor:pointer;">
    <g class="room-group" data-room-id="{r_id}" cursor="pointer">
      <title>{html.escape(room_tooltip)}</title>
      <!-- Room Floor Base -->
      <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}"
            fill="{bg_fill}" stroke="{stroke_color}" stroke-width="{stroke_width}"
            rx="6" {filter_attr} class="room-rect" />

      <!-- Corner Selection Accent -->
      {'<polygon points="' + f'{rx},{ry} {rx+40},{ry} {rx},{ry+40}' + '" fill="#00F0FF" />' if is_sel else ''}

      <!-- Room ID and Name -->
      <text x="{cx}" y="{ry + 55}" text-anchor="middle"
            fill="#F8FAFC" font-size="26" font-weight="800" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif">
        {r_id}{sel_badge}
      </text>
      <text x="{cx}" y="{ry + 90}" text-anchor="middle"
            fill="#94A3B8" font-size="16" font-weight="600" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif">
        {html.escape(room.name)}
      </text>

      <!-- Compact Occupancy Pill Indicator -->
      <g transform="translate({cx - 90}, {cy - 22})">
        <rect x="0" y="0" width="180" height="44" rx="22"
              fill="rgba(15, 23, 42, 0.90)" stroke="{status_color}" stroke-width="2.5" />
        <text x="90" y="28" text-anchor="middle"
              fill="#F8FAFC" font-size="18" font-weight="700" font-family="monospace">
          👥 {count} / {cap}
        </text>
      </g>

      <!-- Status Tag below Occupancy -->
      <text x="{cx}" y="{cy + 60}" text-anchor="middle"
            fill="{status_color}" font-size="15" font-weight="700" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif">
        ● {status_text}
      </text>
    </g>
  </a>
""")

    # 3.5 HVAC Equipment Terminals, Airflow Vectors & Conceptual Distribution Network
    hvac_equipment_locations = {
        "Room 1": {"x": 10.5, "y": 9.4, "dy_sign": 1},   # North wall, supply points DOWN (+SVG y)
        "Room 2": {"x": 20.0, "y": 0.6, "dy_sign": -1},  # South wall, supply points UP (-SVG y)
        "Room 3": {"x": 4.0, "y": 19.4, "dy_sign": 1},   # North wall, supply points DOWN (+SVG y)
        "Room 4": {"x": 13.0, "y": 19.4, "dy_sign": 1},  # North wall, supply points DOWN (+SVG y)
        "Room 5": {"x": 22.0, "y": 19.4, "dy_sign": 1},  # North wall, supply points DOWN (+SVG y)
    }

    # Conceptual Chilled / Refrigerant Distribution Line
    # Illustrates central supply trunk along interior dividing boundary (y=10.0m -> SVG y=400)
    svg_elements.append("""
  <!-- Conceptual Central HVAC Distribution Loop (Chilled Water / Refrigerant Spine) -->
  <g class="hvac-distribution-spine" opacity="0.75">
    <title>Conceptual Central HVAC Distribution Loop (Refrigerant / Chilled Water Distribution Spine)</title>
    <!-- Main distribution trunk line -->
    <line x1="120" y1="400" x2="920" y2="400" stroke="#0284C7" stroke-width="2.5" stroke-dasharray="8,6" />
    <!-- Feeder branches to room units -->
    <line x1="420" y1="400" x2="420" y2="412" stroke="#0284C7" stroke-width="2" stroke-dasharray="4,4" />
    <line x1="800" y1="400" x2="800" y2="764" stroke="#0284C7" stroke-width="1.8" stroke-dasharray="6,6" opacity="0.5" />
    <line x1="160" y1="400" x2="160" y2="36" stroke="#0284C7" stroke-width="1.8" stroke-dasharray="6,6" opacity="0.5" />
    <line x1="520" y1="400" x2="520" y2="36" stroke="#0284C7" stroke-width="1.8" stroke-dasharray="6,6" opacity="0.5" />
    <line x1="880" y1="400" x2="880" y2="36" stroke="#0284C7" stroke-width="1.8" stroke-dasharray="6,6" opacity="0.5" />
    <!-- Spine identification badge -->
    <rect x="440" y="392" width="160" height="16" rx="4" fill="rgba(15, 23, 42, 0.90)" stroke="#0284C7" stroke-width="1" />
    <text x="520" y="403" text-anchor="middle" fill="#38BDF8" font-size="9" font-weight="700" font-family="monospace">
      [CONCEPTUAL CHILLED LOOP]
    </text>
  </g>
""")

    for r_id, eq in hvac_equipment_locations.items():
        e_st = (energy_states.get(r_id) or energy_states.get(r_id.replace(" ", "_"))) if energy_states else None
        s_rd = (sensor_readings.get(r_id) or sensor_readings.get(r_id.replace(" ", "_"))) if sensor_readings else None

        op_st = getattr(e_st, "operating_state", "IDLE") if e_st else "IDLE"
        hvac_m = getattr(e_st, "hvac_mode", "NORMAL") if e_st else "NORMAL"
        if hvac_m == "OFF":
            op_st = "OFF"

        p_w = getattr(e_st, "hvac_power_w", 0.0) if e_st else 0.0
        q_w = getattr(e_st, "cooling_delivered_w", getattr(e_st, "cooling_demand_w", 0.0)) if e_st else 0.0
        airflow = getattr(e_st, "airflow_m3_h", 0.0) if e_st else 0.0
        fan_st = getattr(e_st, "fan_state", "LOW" if op_st == "IDLE" else ("OFF" if op_st == "OFF" else "MED")) if e_st else "LOW"
        target_sp = getattr(e_st, "hvac_target_setpoint", 24.5) if e_st else 24.5
        t_in = getattr(s_rd, "indoor_temperature", getattr(e_st, "indoor_temperature", 24.5)) if s_rd else (getattr(e_st, "indoor_temperature", 24.5) if e_st else 24.5)
        t_err = t_in - target_sp

        ux = to_svg_x(eq["x"])
        uy = to_svg_y(eq["y"])
        bw = 82
        bh = 24
        bx = ux - bw / 2.0
        by = uy - bh / 2.0
        dy_sign = eq["dy_sign"]  # +1: points downward (increasing y), -1: points upward (decreasing y)

        if op_st == "COOLING":
            box_stroke = "#00F0FF"
            box_fill = "rgba(14, 165, 233, 0.40)"
            led_color = "#00F0FF"
            state_label = "❄ COOL"
            state_text_color = "#00F0FF"
            airflow_text = f"{airflow:.0f} m³/h"
            airflow_color = "#38BDF8"
        elif op_st == "IDLE":
            box_stroke = "#64748B"
            box_fill = "rgba(30, 41, 59, 0.70)"
            led_color = "#94A3B8"
            state_label = "⏸ IDLE"
            state_text_color = "#94A3B8"
            airflow_text = f"{airflow:.0f} m³/h"
            airflow_color = "#64748B"
        else:  # OFF
            box_stroke = "#475569"
            box_fill = "rgba(15, 23, 42, 0.90)"
            led_color = "#475569"
            state_label = "⏹ OFF"
            state_text_color = "#64748B"
            airflow_text = "0 m³/h (OFF)"
            airflow_color = "#475569"

        hvac_tooltip = (
            f"{r_id} Virtual HVAC Terminal Unit\n"
            f"Operating State: {op_st} | Control Policy: {hvac_m}\n"
            f"Target Setpoint: {target_sp:.1f}°C (Error: {t_err:+.1f}°C)\n"
            f"Indoor Temp: {t_in:.1f}°C | Power: {p_w:.0f} W (Elec)\n"
            f"Thermal Cooling: {q_w:.0f} W (COP: 3.2)\n"
            f"Airflow: {airflow:.0f} m³/h (Fan: {fan_st})"
        )

        # Airflow vectors:
        flow_lines = []
        if op_st == "COOLING":
            y_start = (by + bh) if dy_sign > 0 else by
            y_end = y_start + (dy_sign * 34)
            for ox in [-22, 0, 22]:
                flow_lines.append(
                    f'<line x1="{ux + ox}" y1="{y_start}" x2="{ux + ox}" y2="{y_end}" '
                    f'stroke="#00F0FF" stroke-width="2" class="hvac-supply-active" marker-end="url(#hvac-arrow-cooling)" />'
                )
            ret_x = bx + bw + 14
            ret_y_start = y_end
            ret_y_end = y_start
            flow_lines.append(
                f'<line x1="{ret_x}" y1="{ret_y_start}" x2="{ret_x}" y2="{ret_y_end}" '
                f'stroke="#94A3B8" stroke-width="1.8" class="hvac-return-active" marker-end="url(#hvac-arrow-return)" />'
            )
        elif op_st == "IDLE":
            y_start = (by + bh) if dy_sign > 0 else by
            y_end = y_start + (dy_sign * 16)
            for ox in [-12, 12]:
                flow_lines.append(
                    f'<line x1="{ux + ox}" y1="{y_start}" x2="{ux + ox}" y2="{y_end}" '
                    f'stroke="#64748B" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.45" marker-end="url(#hvac-arrow-idle)" />'
                )

        svg_elements.append(f"""
  <!-- HVAC Terminal Equipment: {r_id} -->
  <g class="hvac-terminal" data-hvac-room="{r_id}">
    <title>{html.escape(hvac_tooltip)}</title>

    <!-- Airflow Supply & Return Strokes -->
    <g class="hvac-airflow-group">
      {''.join(flow_lines)}
    </g>

    <!-- Equipment Terminal Box -->
    <rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="4"
          fill="{box_fill}" stroke="{box_stroke}" stroke-width="1.8" />

    <!-- Terminal Unit Louver Slats Detail -->
    <line x1="{bx + 6}" y1="{by + 6}" x2="{bx + 26}" y2="{by + 6}" stroke="{box_stroke}" stroke-width="1.2" opacity="0.7" />
    <line x1="{bx + 6}" y1="{by + 11}" x2="{bx + 26}" y2="{by + 11}" stroke="{box_stroke}" stroke-width="1.2" opacity="0.7" />
    <line x1="{bx + 6}" y1="{by + 16}" x2="{bx + 26}" y2="{by + 16}" stroke="{box_stroke}" stroke-width="1.2" opacity="0.7" />

    <!-- Status LED Indicator -->
    <circle cx="{bx + bw - 10}" cy="{by + 12}" r="3.5" fill="{led_color}" />

    <!-- Operational State Text Label -->
    <text x="{bx + 30}" y="{by + 16}" text-anchor="start"
          fill="{state_text_color}" font-size="11" font-weight="800" font-family="-apple-system, sans-serif">
      {state_label}
    </text>

    <!-- Compact Volumetric Airflow Badge -->
    <text x="{ux}" y="{by + (35 if dy_sign > 0 else -6)}" text-anchor="middle"
          fill="{airflow_color}" font-size="10" font-weight="700" font-family="monospace">
      {airflow_text}
    </text>
  </g>
""")

    # 4. Interior Doors (Doorway passages at configured connections)
    doors_drawn = set()
    for (r1, r2), pos in building.door_positions.items():
        pair_key = tuple(sorted([r1, r2]))
        if pair_key in doors_drawn:
            continue
        doors_drawn.add(pair_key)

        dx = to_svg_x(pos[0])
        dy = to_svg_y(pos[1])

        is_vertical = any(abs(room.x - pos[0]) < 0.2 or abs(room.x + room.width - pos[0]) < 0.2 for room in building.rooms.values())

        if is_vertical:
            x1, y1 = dx, dy - 24
            x2, y2 = dx, dy + 24
            rx_d, ry_d = dx - 10, dy - 24
            rw_d, rh_d = 20, 48
        else:
            x1, y1 = dx - 24, dy
            x2, y2 = dx + 24, dy
            rx_d, ry_d = dx - 24, dy - 10
            rw_d, rh_d = 48, 20

        door_tip = f"Doorway: {r1} ↔ {r2}\nClear passage width: 1.2m"

        svg_elements.append(f"""
  <!-- Door {r1} <-> {r2} -->
  <g class="door-group">
    <title>{html.escape(door_tip)}</title>
    <!-- Door Cutout Clearance -->
    <rect x="{rx_d}" y="{ry_d}" width="{rw_d}" height="{rh_d}" fill="#0F172A" />
    <!-- Doorway Accent Line -->
    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
          stroke="#F59E0B" stroke-width="6" stroke-linecap="round" />
  </g>
""")

    # 5. Exterior Main Exit (Room 1 Gateway at sim 7.0, 0.0 -> SVG 280, 800)
    r1 = building.get_room("Room 1")
    if r1 and r1.has_exterior_exit and r1.exterior_door_position:
        ex_x = to_svg_x(r1.exterior_door_position[0])
        ex_y = to_svg_y(r1.exterior_door_position[1])

        svg_elements.append(f"""
  <!-- Exterior Main Exit -->
  <g class="exit-marker" transform="translate({ex_x}, {ex_y})">
    <title>Building Main Exit / Entrance Gateway (Room 1)</title>
    <rect x="-28" y="-6" width="56" height="12" fill="#0F172A" />
    <polygon points="0,28 -20,-4 20,-4" fill="#EF4444" stroke="#FFFFFF" stroke-width="2" />
    <text x="0" y="52" text-anchor="middle" fill="#EF4444" font-size="16" font-weight="800" font-family="-apple-system, sans-serif">
      MAIN EXIT
    </text>
  </g>
""")

    # 6. Entrance Queue (sim 7.0, -1.6 -> SVG 280, 864)
    queue_count = entrance_queue_count
    if queue_count == 0 and occupants:
        queue_count = sum(1 for o in occupants.values() if o.movement_state == "waiting_at_entrance")

    if queue_count > 0:
        q_x = to_svg_x(7.0)
        q_y = to_svg_y(-1.6)
        svg_elements.append(f"""
  <!-- Entrance Waiting Queue -->
  <g class="queue-marker" transform="translate({q_x}, {q_y})">
    <title>Entrance Waiting Queue\nOccupants waiting for admission: {queue_count}\nAdmitted FIFO as capacity frees</title>
    <circle cx="0" cy="0" r="18" fill="#F97316" stroke="#FFFFFF" stroke-width="2.5" />
    <text x="0" y="6" text-anchor="middle" fill="#FFFFFF" font-size="15" font-weight="900" font-family="-apple-system, sans-serif">
      Q
    </text>
    <text x="0" y="38" text-anchor="middle" fill="#F97316" font-size="16" font-weight="800" font-family="monospace">
      QUEUE: {queue_count}
    </text>
  </g>
""")

    # 7. Occupant Markers (Positioned at Authoritative Simulation Coordinates)
    if show_occupants and occupants:
        if animation_manager is not None:
            positions = animation_manager.get_occupant_positions(building, occupants)
        else:
            positions = {}
            room_occupants: Dict[str, List[str]] = {r: [] for r in building.rooms.keys()}
            for pid, occ in sorted(occupants.items()):
                if occ.is_inside_building and occ.current_room_id in room_occupants:
                    room_occupants[occ.current_room_id].append(pid)
            for rm_id, pids in room_occupants.items():
                rm = building.get_room(rm_id)
                tot = len(pids)
                for idx, pid in enumerate(pids):
                    positions[pid] = get_deterministic_room_grid_position(pid, rm, idx, tot)

        # Highlighted route for selected occupant
        if selected_occupant_id and selected_occupant_id in occupants:
            sel_occ = occupants[selected_occupant_id]
            if sel_occ.is_inside_building and animation_manager and sel_occ.intended_destination:
                sel_route = animation_manager.get_selected_route_waypoints(building, sel_occ)
                if len(sel_route) >= 2:
                    pts = " ".join([f"{to_svg_x(p[0])},{to_svg_y(p[1])}" for p in sel_route])
                    svg_elements.append(f"""
  <!-- Selected Occupant Route -->
  <polyline points="{pts}" fill="none" stroke="#FBBF24" stroke-width="5" stroke-dasharray="8,6" stroke-linecap="round" opacity="0.9">
    <title>Planned Route: {sel_occ.current_room_id} ➔ {sel_occ.intended_destination}</title>
  </polyline>
""")

        # Trajectory for active movement event
        if animation_manager and animation_manager.events:
            active_replay = animation_manager.get_active_replay_event()
            ev_to_show = active_replay if active_replay else animation_manager.events[-1]
            if len(ev_to_show.waypoints) >= 2:
                pts = " ".join([f"{to_svg_x(p[0])},{to_svg_y(p[1])}" for p in ev_to_show.waypoints])
                ev_col = "#10B981" if ev_to_show.event_type == "admission" else ("#EF4444" if ev_to_show.event_type == "exit" else "#38BDF8")
                svg_elements.append(f"""
  <!-- Active Movement Event Trajectory -->
  <polyline points="{pts}" fill="none" stroke="{ev_col}" stroke-width="4.5" stroke-dasharray="10,6" stroke-linecap="round" opacity="0.85">
    <title>Active Movement ({ev_to_show.event_type.upper()}): {ev_to_show.source_room or 'Outside'} ➔ {ev_to_show.destination_room or 'Exit'}</title>
  </polyline>
""")

        # Occupant marker circles
        for occ_id, occ in sorted(occupants.items()):
            if occ.is_inside_building and occ.current_room_id:
                pos = positions.get(occ_id, occ.position)
                ox = to_svg_x(max(0.2, min(25.8, pos[0])))
                oy = to_svg_y(max(0.2, min(19.8, pos[1])))

                is_sel_occ = (occ_id == selected_occupant_id)

                if occ.occupant_type == "visitor":
                    fill_col = "#A855F7"
                elif occ.occupant_type == "staff":
                    fill_col = "#10B981"
                else:
                    fill_col = "#38BDF8"

                occ_tip = (
                    f"Occupant: {occ_id}\n"
                    f"Type: {occ.occupant_type.capitalize()}\n"
                    f"Room: {occ.current_room_id}\n"
                    f"Activity: {occ.current_activity}\n"
                    f"State: {occ.movement_state}"
                )

                if is_sel_occ:
                    svg_elements.append(f"""
  <circle cx="{ox}" cy="{oy}" r="20" fill="rgba(251, 191, 36, 0.25)" stroke="#FBBF24" stroke-width="4" filter="url(#gold-glow)" />
""")

                svg_elements.append(f"""
  <circle cx="{ox}" cy="{oy}" r="11" fill="{fill_col}" stroke="#FFFFFF" stroke-width="2" class="occ-marker" data-occ-id="{occ_id}">
    <title>{html.escape(occ_tip)}</title>
  </circle>
""")

    # 8. Assemble full SVG wrapper
    svg_body = "\n".join(svg_elements)
    full_svg = f"""<svg id="{container_id}" xmlns="http://www.w3.org/2000/svg"
     viewBox="-60 -40 1160 980"
     preserveAspectRatio="xMidYMid meet"
     style="width: 100%; height: auto; max-height: 520px; background-color: #0B1120; border-radius: 8px; border: 1px solid #1E293B;">
  <style>
    .room-rect {{
      transition: fill 0.25s ease, stroke 0.25s ease, stroke-width 0.25s ease;
    }}
    .room-group:hover .room-rect {{
      filter: brightness(1.2);
    }}
    .occ-marker {{
      transition: cx 0.35s ease, cy 0.35s ease;
      cursor: pointer;
    }}
    .occ-marker:hover {{
      r: 16;
      stroke: #FBBF24;
      stroke-width: 4;
    }}
    @keyframes hvac-supply-pulse {{
      0% {{ stroke-dashoffset: 16; }}
      100% {{ stroke-dashoffset: 0; }}
    }}
    .hvac-supply-active {{
      animation: hvac-supply-pulse 1.2s linear infinite;
      stroke-dasharray: 6, 4;
    }}
    @keyframes hvac-return-pulse {{
      0% {{ stroke-dashoffset: 0; }}
      100% {{ stroke-dashoffset: 16; }}
    }}
    .hvac-return-active {{
      animation: hvac-return-pulse 1.5s linear infinite;
      stroke-dasharray: 4, 4;
    }}
    .hvac-terminal {{
      cursor: pointer;
      transition: filter 0.2s ease;
    }}
    .hvac-terminal:hover {{
      filter: drop-shadow(0 0 6px rgba(0, 240, 255, 0.6));
    }}
  </style>
  {svg_body}
</svg>"""
    return full_svg
