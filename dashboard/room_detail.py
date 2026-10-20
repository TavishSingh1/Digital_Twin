"""
Selected Room Control Centre
============================
Contextual operational panel for the globally selected room:
- Occupancy monitoring, occupant list, and quick entry/transfer/exit actions
- Environmental multi-physics telemetry with classification tags
- Smart HVAC mode controls, setpoint adjustments, and thermal demand
- Lighting controls (Daylight harvesting vs Manual override)
- Real-time power breakdown, cumulative kWh, and physical explainability
"""

from typing import Any, Optional
import streamlit as st
import pandas as pd

from simulation.config import EXPECTED_ROOM_IDS
from dashboard.state import (
    get_pipeline,
    add_person_action,
    exit_person_action,
    move_people_action,
)


def render_selected_room_control_centre(pipeline: Any, selected_room: str):
    """
    Renders the unified Selected Room Control Centre for the active room.
    """
    room = pipeline.engine.building.get_room(selected_room)
    if not room:
        st.warning(f"Room '{selected_room}' not found in building topology.")
        return

    # Extract state records
    energy_mgr = pipeline.engine.energy_manager
    e_state = energy_mgr.get_latest_states().get(selected_room)
    sensor_meta = pipeline.engine.sensor_manager.get_sensor_metadata(selected_room)
    sensor_df = pipeline.engine.get_sensor_readings_df()

    last_sensor = {}
    sensor_obj = pipeline.engine.sensor_manager.get_latest_readings().get(selected_room)
    if sensor_obj:
        last_sensor = sensor_obj.to_dict() if hasattr(sensor_obj, "to_dict") else vars(sensor_obj)
    elif not sensor_df.empty:
        rdf = sensor_df[sensor_df["room_id"] == selected_room]
        if not rdf.empty:
            last_sensor = rdf.iloc[-1].to_dict()

    connected_doors = sorted(list(pipeline.engine.building.adjacency.get(selected_room, [])))
    doors_str = ", ".join(connected_doors) if connected_doors else "None (Corridor)"

    # Panel Container Header
    st.markdown(
        f"<div class='panel-container'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<div>"
        f"<span style='font-size:1.2rem; font-weight:700; color:#F8FAFC;'>🎛️ Control Centre: {selected_room}</span>"
        f"<span style='color:#94A3B8; font-size:0.9rem; margin-left:8px;'>({room.name})</span>"
        f"</div>"
        f"<span style='font-size:0.8rem; color:#94A3B8;'>Doors: <b>{doors_str}</b> | Size: <b>{room.width:.1f}m × {room.height:.1f}m</b></span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Extract primary physical variables for instant visibility
    count = room.occupant_count
    cap = room.capacity
    util_pct = (count / cap * 100.0) if cap > 0 else 0.0

    t_in = last_sensor.get("indoor_temperature", 24.5)
    t_out = last_sensor.get("dry_bulb_temperature", 28.0)
    rh_in = last_sensor.get("indoor_relative_humidity", 55.0)
    rh_out = last_sensor.get("outdoor_relative_humidity", 87.0)
    co2 = last_sensor.get("indoor_co2", 450.0)
    lux = last_sensor.get("illuminance", 350.0)
    snd = last_sensor.get("sound_pressure_level", 50.0)

    hvac_mode = getattr(e_state, "hvac_mode", "AUTO") if e_state else "AUTO"
    tot_power = getattr(e_state, "total_power_w", 0.0) if e_state else 0.0
    cum_energy = getattr(e_state, "cumulative_total_kwh", 0.0) if e_state else 0.0

    # 1. Primary Metrics Grid for Selected Room (Task 3 Selective Telemetry)
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    m_c1.metric("Occupancy", f"{count} / {cap}", f"{util_pct:.0f}% load")
    m_c2.metric("Indoor Temp", f"{t_in:.1f} °C", f"{t_in - t_out:+.1f} °C vs Ext")
    m_c3.metric("Indoor CO₂", f"{co2:.0f} ppm", "Elevated" if co2 > 1000 else "Nominal", delta_color="inverse" if co2 > 1000 else "normal")
    m_c4.metric("Indoor Humidity", f"{rh_in:.1f} %")

    m2_c1, m2_c2, m2_c3, m2_c4 = st.columns(4)
    m2_c1.metric("Illuminance", f"{lux:.0f} lux")
    m2_c2.metric("Sound Level", f"{snd:.1f} dBA")
    m2_c3.metric("HVAC Mode", hvac_mode)
    m2_c4.metric("Power Demand", f"{tot_power:.0f} W", f"{cum_energy:.2f} kWh total")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Operational Sub-tabs for contextual inspection & command
    sub_occ, sub_env, sub_hvac, sub_light, sub_energy = st.tabs([
        "👥 Occupants & Flow",
        "🌡️ Environmental Sensors",
        "❄️ HVAC Controls",
        "💡 Lighting Policy",
        "⚡ Energy Diagnostics",
    ])

    # -------------------------------------------------------------------------
    # SUB-TAB 1: OCCUPANCY & ACTIONS
    # -------------------------------------------------------------------------
    with sub_occ:
        count = room.occupant_count
        cap = room.capacity
        util_pct = (count / cap * 100.0) if cap > 0 else 0.0

        o_c1, o_c2, o_c3 = st.columns([1, 1, 1])
        o_c1.metric("Occupancy", f"{count} / {cap} people", delta=f"{cap - count} free capacity")
        o_c2.metric("Utilization", f"{util_pct:.1f}%")
        status_label = "Normal" if util_pct < 85 else "High Load (≥85%)"
        o_c3.metric("Load Status", status_label)

        st.progress(min(1.0, count / cap if cap > 0 else 0.0))

        # Occupants list
        occ_list = sorted(list(room.occupants))
        with st.expander(f"📋 Occupants Currently in {selected_room} ({len(occ_list)} individuals)", expanded=False):
            if occ_list:
                occ_rows = []
                for pid in occ_list:
                    o = pipeline.engine.occupants.get(pid)
                    if o:
                        occ_rows.append({
                            "ID": o.occupant_id,
                            "Type": o.occupant_type.capitalize(),
                            "Activity": o.current_activity,
                            "State": o.movement_state,
                            "Destination": o.intended_destination or "None",
                        })
                st.dataframe(pd.DataFrame(occ_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No occupants currently inside this room.")

        # Contextual Quick Actions for this room
        st.markdown("##### ⚡ Quick Occupancy Actions")
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            st.markdown(f"**Move from {selected_room} to Connected Room**")
            if connected_doors and count > 0:
                target_dest = st.selectbox(
                    "Connected Destination",
                    options=connected_doors,
                    key=f"quick_move_dest_{selected_room}",
                )
                dst_obj = pipeline.engine.building.get_room(target_dest)
                dst_avail = dst_obj.capacity - dst_obj.occupant_count

                move_cnt = st.number_input(
                    "Count to Move",
                    min_value=1,
                    max_value=max(1, min(count, dst_avail if dst_avail > 0 else 1)),
                    value=1,
                    step=1,
                    disabled=(dst_avail <= 0),
                    key=f"quick_move_cnt_{selected_room}",
                )

                if st.button(f"Transfer to {target_dest}", use_container_width=True, disabled=(dst_avail <= 0)):
                    res = move_people_action(
                        source_room=selected_room,
                        destination_room=target_dest,
                        count=int(move_cnt),
                    )
                    st.rerun()
            elif count == 0:
                st.caption("Cannot move: room is currently vacant.")
            else:
                st.caption("No direct interior doors connect this room.")

        with act_col2:
            st.markdown(f"**Exit from {selected_room}**")
            if count > 0:
                exit_cnt = st.number_input(
                    "Count to Exit",
                    min_value=1,
                    max_value=count,
                    value=1,
                    step=1,
                    key=f"quick_exit_cnt_{selected_room}",
                )
                if st.button(f"Exit {exit_cnt} from Building", use_container_width=True):
                    res = exit_person_action(count=int(exit_cnt), room_id=selected_room)
                    st.rerun()
            else:
                st.caption("No occupants to exit.")

    # -------------------------------------------------------------------------
    # SUB-TAB 2: ENVIRONMENTAL SENSORS
    # -------------------------------------------------------------------------
    with sub_env:
        t_in = last_sensor.get("indoor_temperature", 24.5)
        t_out = last_sensor.get("dry_bulb_temperature", 28.0)
        rh_in = last_sensor.get("indoor_relative_humidity", 55.0)
        rh_out = last_sensor.get("outdoor_relative_humidity", 87.0)
        co2 = last_sensor.get("indoor_co2", 450.0)
        lux = last_sensor.get("illuminance", 350.0)
        snd = last_sensor.get("sound_pressure_level", 50.0)

        st.markdown("##### ⚡ Immediate Physical Signals")
        imm_c1, imm_c2 = st.columns(2)
        imm_c1.metric("Illuminance", f"{lux:.1f} lux", help="Workplane daylight + artificial luminaires")
        imm_c2.metric("Sound Pressure Level", f"{snd:.1f} dBA", help="Logarithmic acoustic sum of noise floor + occupants")

        st.markdown("##### ⏳ Dynamic Mass & Heat Balance")
        dyn_c1, dyn_c2, dyn_c3 = st.columns(3)
        dyn_c1.metric("Indoor Temperature", f"{t_in:.2f} °C", delta=f"{t_in - t_out:+.2f} °C vs Outdoor")
        dyn_c2.metric("Indoor Relative Humidity", f"{rh_in:.1f} %")
        dyn_c3.metric("Indoor CO₂", f"{co2:.1f} ppm", delta="Elevated" if co2 > 1000 else "Nominal", delta_color="inverse" if co2 > 1000 else "normal")

        st.markdown("##### ⛅ Ambient Outdoor Boundary Conditions")
        out_c1, out_c2 = st.columns(2)
        out_c1.metric("Dry-Bulb Ambient Temp", f"{t_out:.2f} °C")
        out_c2.metric("Outdoor Relative Humidity", f"{rh_out:.1f} %")

    # -------------------------------------------------------------------------
    # SUB-TAB 3: HVAC & SETPOINT
    # -------------------------------------------------------------------------
    with sub_hvac:
        hvac_ctrl = energy_mgr.hvac_controllers[selected_room]
        curr_override = hvac_ctrl.manual_override

        curr_mode = getattr(e_state, "hvac_mode", "AUTO")
        curr_op_st = getattr(e_state, "operating_state", "IDLE")
        curr_sp = getattr(e_state, "hvac_target_setpoint", 24.5)
        curr_p = getattr(e_state, "hvac_power_w", 0.0)
        curr_q = getattr(e_state, "cooling_delivered_w", getattr(e_state, "cooling_demand_w", 0.0))
        curr_air = getattr(e_state, "airflow_m3_h", 0.0)
        curr_fan = getattr(e_state, "fan_state", "LOW")
        curr_cum_kwh = getattr(e_state, "cumulative_hvac_kwh", 0.0)
        curr_expl = getattr(e_state, "controller_explanation", "")

        t_err = t_in - curr_sp

        # Top row: 4 key equipment metrics
        h_c1, h_c2, h_c3, h_c4 = st.columns(4)
        h_c1.metric("Operating State", f"❄ {curr_op_st}" if curr_op_st == "COOLING" else (f"⏸ {curr_op_st}" if curr_op_st == "IDLE" else f"⏹ {curr_op_st}"))
        h_c2.metric("Target Setpoint", f"{curr_sp:.1f} °C", delta=f"{t_err:+.1f} °C error", delta_color="inverse" if abs(t_err) > 0.5 else "normal")
        h_c3.metric("HVAC Electrical Power", f"{curr_p:.0f} W", delta=f"{curr_cum_kwh:.3f} kWh cum")
        h_c4.metric("Thermal Cooling / COP", f"{curr_q:.0f} W (COP {hvac_ctrl.cop:.1f})")

        # Second row: Airflow & Policy
        a_c1, a_c2, a_c3, a_c4 = st.columns(4)
        a_c1.metric("Volumetric Airflow", f"{curr_air:.0f} m³/h", delta=f"Fan: {curr_fan}")
        a_c2.metric("Control Policy", curr_mode)
        a_c3.metric("Rated Capacity", f"{hvac_ctrl.rated_cooling_capacity_w/1000.0:.1f} kW th")
        a_c4.metric("Heating Mode", "Unavailable", help="Cooling-only tropical building system")

        # Live controller explanation box
        if curr_expl:
            st.markdown(
                f"<div style='background:#0F172A; border-left:4px solid #00F0FF; padding:8px 12px; margin:8px 0; border-radius:4px; font-size:0.85rem; color:#E2E8F0;'>"
                f"<b>🤖 Active Controller Decision:</b> {curr_expl}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns(2)

        with ctrl_col1:
            hvac_override_idx = 1 if curr_override == "ECO" else (2 if curr_override == "FORCE_OFF" else 0)
            chosen_hvac_mode = st.radio(
                "HVAC Operational Policy",
                options=["Auto (Occupancy Responsive)", "Eco (Forced Setback 28°C)", "Force OFF (Manual)"],
                index=hvac_override_idx,
                key=f"ctrl_hvac_mode_{selected_room}",
                help="Auto modulates between Normal comfort and Eco setback based on occupancy.",
            )
            desired_override = "ECO" if "Eco" in chosen_hvac_mode else ("FORCE_OFF" if "Force OFF" in chosen_hvac_mode else None)
            if desired_override != curr_override:
                pipeline.engine.set_room_hvac_mode(selected_room, desired_override)
                st.session_state.action_feedback = ("info", f"Updated {selected_room} HVAC mode to: {chosen_hvac_mode}")
                st.rerun()

        with ctrl_col2:
            chosen_setpoint = st.slider(
                "Target Comfort Temperature (°C)",
                min_value=18.0,
                max_value=28.0,
                value=float(hvac_ctrl.target_setpoint),
                step=0.5,
                key=f"ctrl_hvac_sp_{selected_room}",
                help="Target indoor temperature maintained by proportional thermal cooling.",
            )
            if abs(chosen_setpoint - hvac_ctrl.target_setpoint) >= 0.25:
                pipeline.engine.set_room_hvac_setpoint(selected_room, chosen_setpoint)
                st.session_state.action_feedback = ("info", f"Updated {selected_room} HVAC setpoint to: {chosen_setpoint:.1f} °C")
                st.rerun()

    # -------------------------------------------------------------------------
    # SUB-TAB 4: LIGHTING CONTROL
    # -------------------------------------------------------------------------
    with sub_light:
        light_ctrl = energy_mgr.lighting_controllers[selected_room]
        curr_light_override = light_ctrl.manual_override

        l_c1, l_c2, l_c3 = st.columns(3)
        l_c1.metric("Lamp State", "💡 ON" if sensor_meta.get("is_light_on") else "🌑 OFF")
        l_pow = getattr(e_state, "lighting_power_w", 0.0)
        l_c2.metric("Lighting Power", f"{l_pow:.1f} W")
        vac_timer = sensor_meta.get("vacancy_timer_minutes", 0.0)
        l_c3.metric("Vacancy Timer", f"{vac_timer:.1f} / 10.0 min")

        st.markdown("---")
        light_override_idx = 1 if curr_light_override == "FORCE_ON" else (2 if curr_light_override == "FORCE_OFF" else 0)
        chosen_light_mode = st.radio(
            "Lighting Operational Policy",
            options=["Auto (Daylight Harvesting + Vacancy Timeout)", "Force ON (Manual)", "Force OFF (Manual)"],
            index=light_override_idx,
            horizontal=True,
            key=f"ctrl_light_mode_{selected_room}",
            help="Auto dims artificial lighting according to exterior natural daylight.",
        )
        desired_light_override = True if "Force ON" in chosen_light_mode else (False if "Force OFF" in chosen_light_mode else None)
        if desired_light_override != sensor_meta.get("manual_light_override"):
            pipeline.engine.set_room_lighting(selected_room, desired_light_override)
            st.session_state.action_feedback = ("info", f"Updated {selected_room} lighting override to: {chosen_light_mode}")
            st.rerun()

        st.caption(
            f"• Daylight Contribution: `{sensor_meta.get('daylight_lux', 0.0):.1f} lux` | "
            f"Artificial Lamp: `{sensor_meta.get('artificial_lux', 0.0):.1f} lux` | "
            f"Target: `300 lux`"
        )

    # -------------------------------------------------------------------------
    # SUB-TAB 5: ENERGY & DIAGNOSTIC EXPLAINABILITY
    # -------------------------------------------------------------------------
    with sub_energy:
        tot_w = getattr(e_state, "total_power_w", 0.0)
        tot_kwh = getattr(e_state, "cumulative_total_kwh", 0.0)
        cool_w = getattr(e_state, "cooling_demand_w", 0.0)

        e_k1, e_k2, e_k3 = st.columns(3)
        e_k1.metric("Instantaneous Power", f"{tot_w:.0f} W", delta=f"{tot_w/1000.0:.2f} kW")
        e_k2.metric("Cumulative Energy", f"{tot_kwh:.3f} kWh")
        e_k3.metric("Cooling Demand (Thermal)", f"{cool_w:.0f} W", help="Thermal heat extraction rate required by comfort setpoint")

        # Explainability Section
        st.markdown("##### 🔍 Why is this room consuming power?")
        reasons = []
        if count > 0:
            sensible_gain = count * 100.0
            reasons.append(f"• **Occupant Sensible Heat:** {count} person(s) contribute ~{sensible_gain:.0f} W of internal thermal heat gain.")
        else:
            reasons.append("• **Room is Vacant:** No internal occupant metabolic load.")

        t_diff = t_in - curr_sp
        if t_diff > 0.5:
            reasons.append(f"• **Thermal Delta:** Indoor temperature ({t_in:.2f}°C) exceeds target setpoint ({curr_sp:.2f}°C) by {t_diff:+.2f}°C, driving active refrigeration.")
        elif t_diff < -0.5:
            reasons.append(f"• **Within Deadband:** Temperature is below comfort threshold; compressor idling in standby.")
        else:
            reasons.append(f"• **Equilibrium:** Temperature is within ±0.5°C deadband of setpoint.")

        daylight = sensor_meta.get("daylight_lux", 0.0)
        if daylight < 300.0 and sensor_meta.get("is_light_on"):
            reasons.append(f"• **Daylight Deficit:** Ambient daylight ({daylight:.0f} lx) is below the 300 lx workplane target, requiring artificial luminaire supplement.")
        elif not sensor_meta.get("is_light_on"):
            reasons.append("• **Lighting Off:** Room is vacant or lights forced off, consuming 0 W lighting power.")

        if getattr(e_state, "is_eco_mode", False):
            reasons.append("• **Eco Setback Active:** Setpoint relaxed to 28.0°C to conserve energy.")

        if curr_override:
            reasons.append(f"• **Manual HVAC Override Active:** Controller is operating under forced `{curr_override}` policy.")

        st.markdown(
            f"<div class='explain-box'>" + "<br>".join(reasons) + "</div>",
            unsafe_allow_html=True,
        )

