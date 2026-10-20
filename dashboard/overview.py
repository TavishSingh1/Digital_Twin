"""
Building Overview Workspace View (Phase 8.5 Redesign)
=====================================================
Professional Building Energy Management System (BEMS) Digital Twin Command Centre:
1. Header: Facility identity, operational status, logical clock timestamp, and step count.
2. Summary Row: 4 key metrics (Total Occupants, Occupied Rooms, Avg Indoor Temp, Total Power).
3. Main Workspace: Large interactive SVG floor plan side-by-side with Selected Room Details.
4. Secondary Workspace: Clean tabbed access for Occupant Inspector, Replay, and Forecast.
5. Strict Decluttering: Clean typography, 1:1 isometric proportions, restrained overlays.
6. Pure Presentation: Zero simulation state or energy mutations during rendering.
"""

from typing import Any
import streamlit as st
import pandas as pd

from simulation.config import EXPECTED_ROOM_IDS
from dashboard.state import (
    get_selected_room,
    set_selected_room,
    get_selected_overlay,
    set_selected_overlay,
    get_animation_manager,
)
from dashboard.floorplan import (
    create_floorplan_figure,
    OVERLAY_OPTIONS,
    get_overlay_legend_info,
)
from dashboard.svg_floorplan import generate_svg_floorplan_markup
from dashboard.components import render_room_summary_cards
from dashboard.room_detail import render_selected_room_control_centre
from dashboard.charts import create_occupancy_chart
from dashboard.live_canvas import render_live_floorplan_component
from dashboard.occupant_inspector import render_occupant_inspector


def render_overview_view(pipeline: Any):
    """Renders the redesigned, professional Building Overview workspace."""
    if not pipeline:
        st.warning("⚠️ Simulation pipeline is not initialized.")
        return

    selected_room = get_selected_room()
    anim_mgr = get_animation_manager()
    engine = pipeline.engine
    building = engine.building
    occupants = engine.occupants

    # Safe telemetry extraction
    b_cap = building.total_capacity
    b_occ = building.total_occupants
    b_util = (b_occ / b_cap * 100.0) if b_cap > 0 else 0.0

    # Count occupied rooms
    occupied_rooms = sum(1 for r in building.rooms.values() if r.occupant_count > 0)
    total_rooms = len(building.rooms)

    # Sensor data extraction for average indoor temp
    sensor_df = engine.get_sensor_readings_df()
    latest_sensors_dict = {}
    indoor_temps = []
    realtime_sensors = engine.sensor_manager.get_latest_readings() if hasattr(engine, "sensor_manager") else {}
    if realtime_sensors:
        for r_id in EXPECTED_ROOM_IDS:
            s_obj = realtime_sensors.get(r_id)
            if s_obj:
                latest_sensors_dict[r_id] = s_obj.to_dict() if hasattr(s_obj, "to_dict") else vars(s_obj)
                t_val = latest_sensors_dict[r_id].get("indoor_temperature")
                if t_val is not None:
                    indoor_temps.append(t_val)
    elif not sensor_df.empty:
        for r_id in EXPECTED_ROOM_IDS:
            rdf = sensor_df[sensor_df["room_id"] == r_id]
            if not rdf.empty:
                latest_sensors_dict[r_id] = rdf.iloc[-1].to_dict()
                t_val = latest_sensors_dict[r_id].get("indoor_temperature")
                if t_val is not None:
                    indoor_temps.append(t_val)

    avg_indoor_temp = (sum(indoor_temps) / len(indoor_temps)) if indoor_temps else 24.5

    # Energy states extraction for total power demand
    e_states = engine.energy_manager.get_latest_states() if hasattr(engine, "energy_manager") else {}
    tot_building_power_w = sum(getattr(s, "total_power_w", 0.0) for s in e_states.values())

    # -------------------------------------------------------------------------
    # 1. HEADER: FACILITY IDENTITY & OPERATIONAL STATUS
    # -------------------------------------------------------------------------
    curr_dt = engine.current_time
    is_auto = st.session_state.get("is_auto_playing", False)
    status_badge = "🟢 LIVE RUNNING" if is_auto else "⚪ STANDBY (PAUSED)"

    h_col1, h_col2 = st.columns([2.5, 1.5])
    with h_col1:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:2px;'>"
            f"<h3 style='margin:0; color:#F8FAFC; font-weight:800; letter-spacing:-0.5px;'>🏢 Facility Operations Digital Twin</h3>"
            f"<span style='background:#1E293B; border:1px solid #334155; color:#38BDF8; font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:12px;'>"
            f"{status_badge}</span>"
            f"</div>"
            f"<span style='color:#94A3B8; font-size:0.85rem;'>5-Room Cyber-Physical Facility — Intelligent HVAC, Lighting & STGNN Telemetry</span>",
            unsafe_allow_html=True,
        )
    with h_col2:
        st.markdown(
            f"<div style='text-align:right; font-size:0.85rem; color:#94A3B8; padding-top:4px;'>"
            f"Logical Clock: <b style='color:#F8FAFC; font-family:monospace;'>{curr_dt.strftime('%H:%M:%S UTC')}</b> | "
            f"Step: <b style='color:#38BDF8; font-family:monospace;'>#{engine.step_count}</b><br>"
            f"Queue: <b style='color:#F97316;'>{engine.waiting_at_entrance} waiting</b> | "
            f"Exits: <b>{engine.total_exits}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. SMALL SUMMARY ROW (4 Focused Key Performance Indicators)
    # -------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        label="Total Facility Occupancy",
        value=f"{b_occ} / {b_cap}",
        delta=f"{b_util:.1f}% Capacity",
        help="Current occupants inside building versus total physical capacity.",
    )
    k2.metric(
        label="Active Occupied Rooms",
        value=f"{occupied_rooms} / {total_rooms}",
        delta=f"{occupied_rooms} populated",
        help="Rooms currently containing one or more occupants.",
    )
    k3.metric(
        label="Average Indoor Temperature",
        value=f"{avg_indoor_temp:.1f} °C",
        delta="Nominal Comfort",
        help="Area-weighted mean dry-bulb temperature across all 5 zones.",
    )
    k4.metric(
        label="Total Electrical Demand",
        value=f"{tot_building_power_w:.0f} W",
        delta=f"{tot_building_power_w / 1000.0:.2f} kW Load",
        help="Real-time aggregate power draw of HVAC compressors, fans, and luminaires.",
    )

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. MAIN WORKSPACE: LARGE FLOOR PLAN & SELECTED ROOM DETAILS
    # -------------------------------------------------------------------------
    col_floorplan, col_detail = st.columns([1.65, 1.15])

    with col_floorplan:
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
            f"<span style='font-size:1.05rem; font-weight:700; color:#F8FAFC;'>🗺️ Interactive Building Floor Plan</span>"
            f"<span style='font-size:0.8rem; color:#94A3B8;'>Focused Zone: <b style='color:#00F0FF;'>{selected_room}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Quick Room Selector Pills (1-Click instant room focus)
        pill_cols = st.columns(5)
        room_nicknames = {
            "Room 1": "Lobby",
            "Room 2": "Office A",
            "Room 3": "Suite",
            "Room 4": "Office B",
            "Room 5": "Lounge",
        }
        for idx, r_id in enumerate(EXPECTED_ROOM_IDS):
            is_active = (r_id == selected_room)
            nick = room_nicknames.get(r_id, "")
            btn_label = f"★ {r_id}" if is_active else f"{r_id} ({nick})"
            if pill_cols[idx].button(
                btn_label,
                key=f"bems_pill_{r_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                set_selected_room(r_id)
                st.rerun()

        # Overlay Selector & Display Mode Selector
        c_ov, c_disp = st.columns([1.8, 1.2])
        with c_ov:
            current_overlay = get_selected_overlay()
            def_idx = OVERLAY_OPTIONS.index(current_overlay) if current_overlay in OVERLAY_OPTIONS else 0
            chosen_overlay = st.selectbox(
                "🎨 Multi-Physics Overlay Layer",
                options=OVERLAY_OPTIONS,
                index=def_idx,
                help="Select physical telemetry overlay to project onto 2D room layouts.",
            )
            if chosen_overlay != current_overlay:
                set_selected_overlay(chosen_overlay)

        with c_disp:
            display_mode = st.radio(
                "Visualization Engine",
                options=["Primary SVG", "Technical Blueprint", "Live Canvas"],
                index=0,
                horizontal=True,
                key="rb_floorplan_view_mode",
                help="Select floor plan visualizer engine.",
            )

        # Render Active Visualization Engine
        if display_mode == "Technical Blueprint":
            # Plotly Blueprint figure (preserves full Phase 8.1 / 8.4 test parity)
            floorplan_fig = create_floorplan_figure(
                building=building,
                occupants=occupants,
                show_occupants=anim_mgr.show_occupants,
                dark_mode=True,
                entrance_queue_count=engine.waiting_at_entrance,
                energy_states=e_states,
                sensor_readings=latest_sensors_dict,
                selected_room_id=selected_room,
                active_overlay=chosen_overlay,
                animation_manager=anim_mgr,
                show_paths=anim_mgr.show_paths,
                selected_occupant_id=anim_mgr.selected_occupant_id,
            )
            st.plotly_chart(floorplan_fig, use_container_width=True)

        elif display_mode == "Live Canvas":
            # Stabilized HTML5 Canvas visualizer
            render_live_floorplan_component(
                building=building,
                occupants=occupants,
                anim_mgr=anim_mgr,
                selected_room_id=selected_room,
                active_overlay=chosen_overlay,
                entrance_queue_count=engine.waiting_at_entrance,
                sensor_readings=latest_sensors_dict,
                energy_states=e_states,
                height=490,
            )

        else:
            # PRIMARY SVG Floor Plan Experience (Phase 8.5)
            svg_markup = generate_svg_floorplan_markup(
                building=building,
                occupants=occupants,
                selected_room_id=selected_room,
                active_overlay=chosen_overlay,
                sensor_readings=latest_sensors_dict,
                energy_states=e_states,
                animation_manager=anim_mgr,
                show_occupants=anim_mgr.show_occupants,
                selected_occupant_id=anim_mgr.selected_occupant_id,
                entrance_queue_count=engine.waiting_at_entrance,
            )
            html_container = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0B1120; overflow: hidden; }}
  .svg-wrapper {{ width: 100%; height: 490px; display: flex; align-items: center; justify-content: center; }}
</style>
</head>
<body>
<div class="svg-wrapper">
{svg_markup}
</div>
</body>
</html>"""
            import streamlit.components.v1 as components
            components.html(html_container, height=490)

        # Restrained Legend Bar
        legend_info = get_overlay_legend_info(chosen_overlay)
        pill_htmls = []
        for r in legend_info["ranges"]:
            pill_htmls.append(
                f"<span style='display:inline-block; padding:2px 8px; border-radius:4px; margin-right:6px; "
                f"background:{r['color']}2B; border:1px solid {r['color']}; color:#F8FAFC; font-size:0.75rem; font-weight:600;'>"
                f"{r['label']}"
                f"</span>"
            )
        st.markdown(
            f"<div style='background:#1E293B; border:1px solid #334155; border-radius:6px; padding:6px 12px; margin-bottom:12px; font-size:0.8rem;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>"
            f"<span><b>Overlay:</b> {legend_info['name']} (<code>{legend_info['unit']}</code>)</span>"
            f"<span style='color:#94A3B8; font-size:0.75rem;'>{legend_info['desc']}</span>"
            f"</div>"
            f"<div>{''.join(pill_htmls)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_detail:
        # Contextual Selected Room Control Centre
        render_selected_room_control_centre(pipeline, selected_room)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. SECONDARY WORKSPACE: OPTIONAL PREDICTION, OCCUPANTS & DIAGNOSTICS
    # -------------------------------------------------------------------------
    st.markdown("#### 🔍 Facility Diagnostics & Operational Workspaces")
    tab_inspect, tab_matrix, tab_ops = st.tabs([
        "👤 Occupant Spatial Inspector & Replay",
        "📊 Room Comparison Matrix & Forecast Timeline",
        "🧭 Contextual System Operations",
    ])

    with tab_inspect:
        render_occupant_inspector(pipeline, anim_mgr, selected_room)

    with tab_matrix:
        st.markdown("##### 📊 Cross-Room Physical & Electrical Comparison Matrix")
        comp_rows = []
        for r_id in EXPECTED_ROOM_IDS:
            rm = building.get_room(r_id)
            e_st = e_states.get(r_id)
            s_row = latest_sensors_dict.get(r_id, {})

            comp_rows.append({
                "Room": r_id,
                "Name": rm.name,
                "Occupancy": f"{rm.occupant_count} / {rm.capacity}",
                "Utilization": f"{(rm.occupant_count / rm.capacity * 100.0) if rm.capacity > 0 else 0.0:.0f}%",
                "Indoor Temp": f"{s_row.get('indoor_temperature', 24.5):.2f} °C",
                "Indoor CO₂": f"{s_row.get('indoor_co2', 450.0):.0f} ppm",
                "Illuminance": f"{s_row.get('illuminance', 300.0):.0f} lux",
                "Sound Level": f"{s_row.get('sound_pressure_level', 50.0):.1f} dBA",
                "HVAC Mode": getattr(e_st, "hvac_mode", "AUTO") if e_st else "AUTO",
                "HVAC Power": f"{getattr(e_st, 'hvac_power_w', 0.0):.0f} W" if e_st else "0 W",
                "Total Power": f"{getattr(e_st, 'total_power_w', 0.0):.0f} W" if e_st else "0 W",
                "Cumulative Energy": f"{getattr(e_st, 'cumulative_total_kwh', 0.0):.2f} kWh" if e_st else "0.00 kWh",
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.markdown("##### 📈 Occupancy Dynamics & 15-Minute STGNN Forecast Timeline")
        c_sel, _ = st.columns([1, 2])
        with c_sel:
            selected_room_timeline = st.selectbox(
                "Select Zone for Timeline Chart",
                options=["All Rooms (Total Building)"] + EXPECTED_ROOM_IDS,
                index=0,
                key="sb_redesign_timeline_room",
            )
        filter_room = None if selected_room_timeline == "All Rooms (Total Building)" else selected_room_timeline
        occ_timeline_fig = create_occupancy_chart(
            gt_df=engine.recorder.to_dataframe(),
            pred_df=pipeline.get_predictions_df(),
            room_id=filter_room,
            dark_mode=True,
        )
        st.plotly_chart(occ_timeline_fig, use_container_width=True)

    with tab_ops:
        c_nav1, c_nav2 = st.columns(2)
        with c_nav1:
            st.markdown(
                f"<div style='background:#1E293B; border:1px solid #3B82F6; border-radius:8px; padding:14px 18px;'>"
                f"<h5 style='margin:0 0 6px 0; color:#60A5FA;'>🚶 Dispatch People Flow</h5>"
                f"<p style='margin:0 0 8px 0; font-size:0.85rem; color:#94A3B8;'>Perform multi-person batch transfers, manual entry, or route evacuation for <b>{selected_room}</b> in the People Flow Control workspace.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c_nav2:
            st.markdown(
                f"<div style='background:#1E293B; border:1px solid #10B981; border-radius:8px; padding:14px 18px;'>"
                f"<h5 style='margin:0 0 6px 0; color:#34D399;'>⚡ Manage Smart HVAC & Energy</h5>"
                f"<p style='margin:0 0 8px 0; font-size:0.85rem; color:#94A3B8;'>Analyze cooling loads, adjust temperature deadbands, or toggle daylight harvesting in <b>{selected_room}</b> in the Smart HVAC & Energy workspace.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
