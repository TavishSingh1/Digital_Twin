"""
Smart HVAC & Energy Management Workspace View
=============================================
Operational energy management dashboard:
- Building-level power & energy KPIs (electrical kW, thermal kW, airflow m³/h)
- 5-Room HVAC Status Matrix (temperatures, setpoint deadband, fan & airflow, COP, operating states)
- Highest-consuming room callout with physical explainability
- Dedicated Time-Series HVAC Analytics (Temperature vs Setpoint with deadband, Power & Energy dual-axis)
- Filterable and sortable room-wise power & energy table
- Synchronized comparative Plotly charts
- Selected room energy overrides & setpoint controls
- Reproducible multi-policy simulation experiment (AUTO vs ECO vs FORCE_OFF) with zero active state mutation
- Baseline comparator and engineering disclaimers
"""

from typing import Any, Optional
import streamlit as st
import pandas as pd

from simulation.config import EXPECTED_ROOM_IDS
from dashboard.state import get_selected_room, set_selected_room
from dashboard.charts import (
    create_room_power_breakdown_chart,
    create_cumulative_energy_trend_chart,
    create_occupancy_vs_power_chart,
    create_energy_savings_comparison_chart,
    create_temperature_vs_setpoint_chart,
    create_hvac_power_and_energy_chart,
    create_policy_experiment_chart,
)
from simulation.experiments import run_hvac_mode_experiment


def render_energy_view(pipeline: Any):
    """Renders the comprehensive Smart HVAC & Energy Management workspace."""
    if not pipeline:
        st.warning("⚠️ Simulation pipeline is not initialized.")
        return

    st.subheader("⚡ Smart HVAC & Building Energy Management System")
    st.caption(
        "Cyber-physical HVAC telemetry, occupancy-responsive modulation, deadband hysteresis (±0.5°C), "
        "volumetric airflow dynamics, and physical electrical-to-thermal power decomposition."
    )

    energy_mgr = pipeline.engine.energy_manager
    latest_states = energy_mgr.get_latest_states()
    energy_df = energy_mgr.to_dataframe()
    kpis = energy_mgr.get_building_kpis()
    comparison = energy_mgr.get_comparison_metrics()

    # 1. Top-Level Facility HVAC & Energy KPIs
    k_c1, k_c2, k_c3, k_c4, k_c5, k_c6 = st.columns(6)
    k_c1.metric(
        "Total Power Demand",
        f"{kpis['total_power_kw']:.2f} kW",
        help="Instantaneous electrical power drawn by HVAC, lighting, and base loads across all 5 rooms",
    )
    k_c2.metric(
        "HVAC Electrical Power",
        f"{kpis['hvac_power_kw']:.2f} kW",
        help="Total electrical power consumed by AC compressor refrigeration and fan circuits",
    )
    k_c3.metric(
        "Thermal Cooling Delivered",
        f"{kpis.get('total_cooling_delivered_kw', 0.0):.2f} kW th",
        help="Total thermal heat extraction rate across conditioned rooms (Q_thermal = COP × P_elec)",
    )
    k_c4.metric(
        "Total Volumetric Airflow",
        f"{kpis.get('total_airflow_m3_h', 0.0):.0f} m³/h",
        help="Aggregate conditioned supply airflow circulated across terminal units",
    )
    k_c5.metric(
        "Active Terminal Units",
        f"{kpis.get('rooms_cooling_count', 0)} Cooling",
        delta=f"{kpis.get('rooms_idle_count', 0)} Idle, {kpis.get('rooms_off_count', 0)} Off",
        delta_color="off",
        help="Compressor operating distribution across all 5 thermal zones",
    )
    k_c6.metric(
        "Cumulative Energy",
        f"{kpis['cumulative_total_kwh']:.3f} kWh",
        delta=f"Avg Error: {kpis.get('avg_temp_error_celsius', 0.0):.2f} °C",
        delta_color="off",
        help="Integrated electrical energy consumption and mean absolute setpoint tracking error",
    )

    st.markdown("---")

    # 2. Dedicated 5-Room HVAC Status Matrix
    st.markdown("#### ❄️ Five-Room HVAC Terminal Status Matrix")
    st.caption(
        "Zone-level thermodynamic states, thermostat deadband tracking (±0.5°C hysteresis), "
        "fan volumetric airflow rates, electrical vs thermal power, and operating compressor states."
    )

    if latest_states:
        matrix_rows = []
        for r_id in EXPECTED_ROOM_IDS:
            s = latest_states.get(r_id)
            if not s:
                continue

            # Deadband evaluation
            t_err = s.indoor_temperature - s.hvac_target_setpoint
            if s.hvac_mode == "FORCE_OFF":
                db_status = "⏹ SHUTDOWN"
            elif t_err > 0.5:
                db_status = f"❄ COOLING REQ (+{t_err:.2f}°C)"
            elif t_err < -0.5:
                db_status = f"⏸ OVERCOOLED ({t_err:.2f}°C)"
            else:
                db_status = f"✅ IN DEADBAND (±0.5°C)"

            # State badge
            if s.operating_state == "COOLING":
                op_badge = "❄ COOLING"
            elif s.operating_state == "IDLE":
                op_badge = "⏸ IDLE"
            else:
                op_badge = "⏹ OFF"

            # Mode badge
            if s.hvac_mode == "HIGH_DEMAND":
                mode_badge = "🔴 HIGH_DEMAND"
            elif s.hvac_mode == "ECO" or s.is_eco_mode:
                mode_badge = "🟢 ECO"
            elif s.hvac_mode == "FORCE_OFF":
                mode_badge = "⚪ FORCE_OFF"
            else:
                mode_badge = "🔵 AUTO"

            matrix_rows.append({
                "Room": f"{r_id} - {s.room_name}",
                "Occupancy": f"{s.occupant_count} / {s.capacity}",
                "Indoor Temp": f"{s.indoor_temperature:.2f} °C",
                "Setpoint": f"{s.hvac_target_setpoint:.1f} °C",
                "Deadband Status": db_status,
                "Control Mode": mode_badge,
                "Operating State": op_badge,
                "Volumetric Airflow": f"{s.airflow_m3_h:.0f} m³/h ({s.fan_state})",
                "Electrical Power": f"{s.hvac_power_w:.0f} W",
                "Thermal Cooling": f"{s.cooling_delivered_w:.0f} W (COP {s.hvac_cop:.1f})",
                "Heating Status": "Unavailable (Cooling-Only)",
            })

        df_matrix = pd.DataFrame(matrix_rows)
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    # 3. Highest Consuming Room Callout & Physical Explainability
    if latest_states:
        sorted_states = sorted(latest_states.values(), key=lambda s: s.total_power_w, reverse=True)
        top_room = sorted_states[0]
        top_name = top_room.room_name
        top_rid = top_room.room_id
        top_p = top_room.total_power_w
        top_hvac_p = top_room.hvac_power_w
        top_cool_w = top_room.cooling_delivered_w
        top_occ = top_room.occupant_count
        sensible_w = top_occ * 100.0
        t_delta = top_room.indoor_temperature - top_room.hvac_target_setpoint

        st.markdown(
            f"<div style='background:#1E293B; border:1px solid #E74C3C; border-left:6px solid #EF4444; border-radius:6px; padding:12px; margin-top:10px; margin-bottom:14px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div>"
            f"<span style='font-size:1.05rem; font-weight:700; color:#F8FAFC;'>🔥 Highest Consuming Zone: {top_rid} ({top_name})</span>"
            f"</div>"
            f"<span style='font-size:1.1rem; font-weight:700; color:#F87171;'>{top_p:.0f} W Total ({top_hvac_p:.0f} W HVAC elec)</span>"
            f"</div>"
            f"<div style='font-size:0.85rem; color:#CBD5E1; margin-top:6px;'>"
            f"<b>Physical Drivers:</b> Occupants: <b>{top_occ}</b> contributing <b>~{sensible_w:.0f} W</b> internal sensible heat. "
            f"Indoor temp: <b>{top_room.indoor_temperature:.2f} °C</b> (Target: <b>{top_room.hvac_target_setpoint:.1f} °C</b>, Error: <b>{t_delta:+.2f} °C</b>). "
            f"Thermal Cooling: <b>{top_cool_w:.0f} W th</b> at <b>{top_room.airflow_m3_h:.0f} m³/h</b> airflow. "
            f"HVAC Mode: <b>{top_room.hvac_mode}</b> (Operating: <b>{top_room.operating_state}</b>). "
            f"<br><b>Controller Rationale:</b> <i>{top_room.controller_explanation}</i>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 4. Time-Series HVAC Analytics (Temperature vs Setpoint & Power vs Energy)
    st.markdown("#### 📈 Time-Series HVAC Analytics & Thermostat Tracking")
    chart_col1, chart_col2 = st.columns([1, 3])
    with chart_col1:
        selected_zone_for_chart = st.selectbox(
            "Select Thermal Zone",
            options=["All Rooms (Total Building)"] + EXPECTED_ROOM_IDS,
            index=1 if get_selected_room() in EXPECTED_ROOM_IDS else 0,
            key="sb_hvac_chart_zone",
        )
    with chart_col2:
        st.caption(
            f"Visualizing continuous thermodynamic tracking for **{selected_zone_for_chart}**. "
            "Shaded band on temperature chart represents the inactive deadband (±0.5°C) preventing compressor cycling."
        )

    an_c1, an_c2 = st.columns(2)
    with an_c1:
        st.plotly_chart(
            create_temperature_vs_setpoint_chart(energy_df, room_id=selected_zone_for_chart, dark_mode=True),
            use_container_width=True,
        )
    with an_c2:
        st.plotly_chart(
            create_hvac_power_and_energy_chart(energy_df, room_id=selected_zone_for_chart, dark_mode=True),
            use_container_width=True,
        )

    st.markdown("---")

    # 5. Comparative Visualizations (Energy breakdown & savings)
    st.markdown("#### 📊 Comparative Facility Energy Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_room_power_breakdown_chart(energy_df, dark_mode=True), use_container_width=True)
    with c2:
        st.plotly_chart(create_cumulative_energy_trend_chart(energy_df, dark_mode=True), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(create_occupancy_vs_power_chart(energy_df, dark_mode=True), use_container_width=True)
    with c4:
        st.plotly_chart(create_energy_savings_comparison_chart(comparison, dark_mode=True), use_container_width=True)

    st.markdown("---")

    # 6. Selected Room Operational Controls & Overrides
    st.markdown("#### 🎛️ Selected Room Energy & HVAC Overrides")

    curr_sel = get_selected_room()
    sel_c1, sel_c2, sel_c3 = st.columns([1, 1.2, 1.2])

    with sel_c1:
        target_ctrl_room = st.selectbox(
            "Select Room for Overrides",
            options=EXPECTED_ROOM_IDS,
            index=EXPECTED_ROOM_IDS.index(curr_sel) if curr_sel in EXPECTED_ROOM_IDS else 0,
            key="sb_energy_ctrl_room",
        )
        if target_ctrl_room != curr_sel:
            set_selected_room(target_ctrl_room)

        r_st = latest_states.get(target_ctrl_room)
        if r_st:
            st.write(f"**{r_st.room_name}**")
            st.caption(f"Occupancy: **{r_st.occupant_count} / {r_st.capacity}** ({r_st.occupancy_ratio*100:.1f}%)")
            st.caption(f"Current HVAC Mode: `{r_st.hvac_mode}` ({r_st.operating_state})")
            st.caption(f"Cooling Delivered: `{r_st.cooling_delivered_w:.1f} W th`")
            st.caption(f"Volumetric Airflow: `{r_st.airflow_m3_h:.0f} m³/h ({r_st.fan_state})`")

    hvac_ctrl = energy_mgr.hvac_controllers[target_ctrl_room]
    curr_hvac_override = hvac_ctrl.manual_override
    hvac_idx = 1 if curr_hvac_override == "ECO" else (2 if curr_hvac_override == "FORCE_OFF" else 0)

    with sel_c2:
        st.markdown("##### ❄️ HVAC Mode & Setpoint")
        chosen_hvac_mode = st.radio(
            "HVAC Control Mode",
            options=["Auto (Occupancy Responsive)", "Eco (Forced Setback)", "Force OFF (Manual)"],
            index=hvac_idx,
            horizontal=True,
            key=f"bems_hvac_mode_{target_ctrl_room}",
        )
        desired_hvac_override = "ECO" if "Eco" in chosen_hvac_mode else ("FORCE_OFF" if "Force OFF" in chosen_hvac_mode else None)
        if desired_hvac_override != curr_hvac_override:
            pipeline.engine.set_room_hvac_mode(target_ctrl_room, desired_hvac_override)
            st.session_state.action_feedback = ("info", f"Set {target_ctrl_room} HVAC mode to {chosen_hvac_mode}")
            st.rerun()

        chosen_setpoint = st.slider(
            "Target Comfort Temperature (°C)",
            min_value=18.0,
            max_value=28.0,
            value=float(hvac_ctrl.target_setpoint),
            step=0.5,
            key=f"bems_hvac_sp_{target_ctrl_room}",
        )
        if abs(chosen_setpoint - hvac_ctrl.target_setpoint) >= 0.25:
            pipeline.engine.set_room_hvac_setpoint(target_ctrl_room, chosen_setpoint)
            st.session_state.action_feedback = ("info", f"Set {target_ctrl_room} setpoint to {chosen_setpoint:.1f} °C")
            st.rerun()

        st.caption("ℹ️ **Heating Status:** Unavailable (Cooling-Only System - DX units with cooling coils only)")

    light_ctrl = energy_mgr.lighting_controllers[target_ctrl_room]
    curr_light_override = light_ctrl.manual_override
    light_idx = 1 if curr_light_override == "FORCE_ON" else (2 if curr_light_override == "FORCE_OFF" else 0)

    with sel_c3:
        st.markdown("##### 💡 Lighting Policy")
        chosen_light_mode = st.radio(
            "Lighting Mode",
            options=["Auto (Daylight Harvesting)", "Force ON (Manual)", "Force OFF (Manual)"],
            index=light_idx,
            horizontal=True,
            key=f"bems_light_mode_{target_ctrl_room}",
        )
        desired_light_override = True if "Force ON" in chosen_light_mode else (False if "Force OFF" in chosen_light_mode else None)
        sensor_meta = pipeline.engine.sensor_manager.get_sensor_metadata(target_ctrl_room)
        if desired_light_override != sensor_meta.get("manual_light_override"):
            pipeline.engine.set_room_lighting(target_ctrl_room, desired_light_override)
            st.session_state.action_feedback = ("info", f"Set {target_ctrl_room} lighting override to {chosen_light_mode}")
            st.rerun()

        st.caption(
            "⚠️ **Policy Note:** Manual overrides bypass automated occupancy setbacks and daylight harvesting, "
            "directly influencing compressor load and lighting kWh accumulation."
        )

    st.markdown("---")

    # 7. Reproducible Multi-Policy HVAC Experiment (AUTO vs ECO vs FORCE_OFF)
    st.markdown("#### 🔬 Reproducible Multi-Policy HVAC Simulation Experiment")
    st.caption(
        "Benchmarking control strategies (Smart AUTO, Continuous ECO Setback, and FORCE OFF Baseline) "
        "under identical initial conditions, random seed (42), and stochastic weather patterns."
    )

    exp_col1, exp_col2, exp_col3 = st.columns([1.2, 1, 2])
    with exp_col1:
        exp_duration = st.slider(
            "Experiment Duration (Steps)",
            min_value=12,
            max_value=48,
            value=24,
            step=6,
            help="12 steps = 1 hr, 24 steps = 2 hrs, 48 steps = 4 hrs at 5-minute sampling interval",
            key="slider_hvac_exp_steps",
        )
    with exp_col2:
        st.write("")
        st.write("")
        run_exp_btn = st.button("▶️ Run Experiment Benchmark", key="btn_run_hvac_policy_exp")

    # Run or initialize experiment results in session state
    if run_exp_btn or ("hvac_experiment_results" not in st.session_state):
        with st.spinner("Executing isolated simulation benchmarks across policies..."):
            st.session_state.hvac_experiment_results = run_hvac_mode_experiment(
                duration_steps=exp_duration,
                seed=42,
            )
        if run_exp_btn:
            st.success(f"Benchmarking complete for {exp_duration} steps ({exp_duration*5/60:.1f} hours). Active simulation unaffected.")

    exp_res = st.session_state.hvac_experiment_results
    if exp_res:
        summary_table = exp_res.get("summary_table", pd.DataFrame())
        display_summary = [c for c in summary_table.columns if not c.startswith("_")]
        st.dataframe(summary_table[display_summary], use_container_width=True, hide_index=True)

        st.plotly_chart(create_policy_experiment_chart(exp_res, dark_mode=True), use_container_width=True)

        st.info(
            "⚖️ **Comfort vs Energy Trade-off Disclosure:** Continuous ECO Setback enforces a constant 28.0°C setback target "
            "rather than maintaining occupied comfort (24.5°C). While ECO achieves dramatic energy savings (~96%), "
            "it does so by relaxing thermal comfort (+3.5°C warmer indoor temperatures). "
            "This is a **comfort-versus-energy trade-off scenario**, NOT an equivalent-comfort energy saving. "
            "For equivalent-comfort efficiency savings, refer to Section 8 below where Smart AUTO is benchmarked "
            "against the fixed 24.0°C baseline policy."
        )

        st.caption(
            "🔒 **Strict Isolation Guarantee:** This experiment executes inside dedicated, transient `SimulationEngine` "
            "instances in isolated memory. Active operator setpoints, room schedules, and live telemetry remain strictly unmutated."
        )

    st.markdown("---")

    # 8. Baseline Savings Evaluation & Disclaimers
    st.markdown("#### 📊 Smart vs Baseline Comparative Efficiency")
    sav_c1, sav_c2, sav_c3, sav_c4 = st.columns(4)
    sav_c1.metric("Baseline Unmodulated Energy", f"{comparison['baseline_total_kwh']:.3f} kWh")
    sav_c2.metric("Smart BEMS Control Energy", f"{comparison['smart_total_kwh']:.3f} kWh")
    sav_c3.metric(
        "Energy Saved",
        f"{comparison['savings_kwh']:.3f} kWh",
        delta=f"{comparison['savings_percent']:.1f}% reduction",
    )
    sav_c4.metric(
        "Demand Profile",
        f"Peak: {comparison['peak_smart_demand_kw']:.2f} kW",
        delta=f"Avg: {comparison['avg_smart_demand_kw']:.2f} kW",
        delta_color="off",
    )

    st.info(
        "ℹ️ **Engineering Disclaimer:** Energy and thermodynamic calculations reflect physical engineering approximations "
        "(sensible human heat gain 100 W/occ, COP 3.2, 120 m³/h per kW cooling, daylight harvesting). "
        "Heating is unavailable in this cooling-only installation. Telemetry does not represent revenue-grade utility billing meters."
    )
