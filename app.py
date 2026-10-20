"""
Interactive Building Energy Management System (BEMS) & Occupancy Digital Twin
=============================================================================
Pairing discrete-time building multi-physics simulation, environmental telemetry,
real-time smart HVAC & lighting energy management, and a Joint Improved Spatio-Temporal
Graph Neural Network (STGNN) forecasting room occupancy 15 minutes ahead.
"""

import time
import streamlit as st

from dashboard.state import (
    init_session_state,
    get_pipeline,
    advance_pipeline_step,
    reset_simulation,
    get_selected_room,
)
from dashboard.components import (
    inject_bems_theme,
    render_simulation_toolbar,
    render_action_feedback,
    scenario_selector_dialog,
    confirm_reset_dialog,
    bems_help_dialog,
)
from dashboard.overview import render_overview_view
from dashboard.people_flow import render_people_flow_view
from dashboard.sensors_view import render_sensors_view
from dashboard.energy_view import render_energy_view
from dashboard.prediction_view import render_prediction_view
from dashboard.exports_view import render_exports_view
from dashboard.scenarios import SCENARIO_PRESETS


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Facility BEMS — Digital Twin Command Centre",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="auto",
)

# Persistent session state initialization
init_session_state()
pipeline = get_pipeline()

# Inject industrial IoT control-room styling
inject_bems_theme()


# -----------------------------------------------------------------------------
# 2. SIDEBAR: FACILITY & SYSTEM STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Facility BEMS System")
    st.caption("5-Room Cyber-Physical Building Digital Twin")

    if st.session_state.model_load_error:
        st.error(f"⚠️ Model Loading Error:\n{st.session_state.model_load_error}")
    else:
        st.success("STGNN Neural Model: Loaded & Active")

    if pipeline:
        st.markdown("---")
        st.markdown("#### ⚡ System Telemetry")
        curr_dt = pipeline.engine.current_time
        st.write(f"- **Logical Clock:** `{curr_dt.strftime('%H:%M:%S UTC')}`")
        st.write(f"- **Date:** `{curr_dt.strftime('%Y-%m-%d')}`")
        st.write(f"- **Step Count:** `Step {pipeline.engine.step_count}`")
        st.write(f"- **Total Inside:** `{pipeline.engine.building.total_occupants} / {pipeline.engine.building.total_capacity}`")
        st.write(f"- **Entrance Queue:** `{pipeline.engine.waiting_at_entrance}`")

        st.markdown("---")
        st.markdown("#### 🎬 Scenario Management")
        active_scen = st.session_state.get("active_scenario", "Normal Workday")
        st.info(f"Active Scenario:\n**{active_scen}**")
        if st.button("Change Scenario...", use_container_width=True):
            scenario_selector_dialog()

        st.markdown("---")
        st.markdown("#### ⚙️ Maintenance & Diagnostics")
        c_rst, c_hlp = st.columns(2)
        with c_rst:
            if st.button("Reset...", use_container_width=True, help="Reinitialize simulation"):
                confirm_reset_dialog()
        with c_hlp:
            if st.button("Guide...", use_container_width=True, help="Operator Guide"):
                bems_help_dialog()

    st.markdown("---")
    st.caption("Cadence: 5 min | Horizon: 15 min ahead | Rooms: 5")


# -----------------------------------------------------------------------------
# 3. GLOBAL BEMS SIMULATION TOOLBAR & FEEDBACK
# -----------------------------------------------------------------------------
render_simulation_toolbar(pipeline)
render_action_feedback()


# -----------------------------------------------------------------------------
# 4. OPERATIONAL WORKSPACES (TABS)
# -----------------------------------------------------------------------------
tab_overview, tab_flow, tab_sensors, tab_energy, tab_eval, tab_export = st.tabs([
    "🏢 Building Overview",
    "🚶 People Flow Control",
    "🌡️ Environmental Telemetry",
    "⚡ Smart HVAC & Energy",
    "🔮 Predictive Evaluation",
    "💾 Data Export",
])

with tab_overview:
    render_overview_view(pipeline)

with tab_flow:
    render_people_flow_view(pipeline)

with tab_sensors:
    render_sensors_view(pipeline)

with tab_energy:
    render_energy_view(pipeline)

with tab_eval:
    render_prediction_view(pipeline)

with tab_export:
    render_exports_view(pipeline)


# -----------------------------------------------------------------------------
# 5. CONTINUOUS SIMULATION PLAYBACK LOOP
# -----------------------------------------------------------------------------
if st.session_state.get("is_auto_playing", False) and pipeline is not None:
    # Small interval delay to allow user to view state and click pause
    delay = float(st.session_state.get("playback_speed", 1.0))
    time.sleep(delay)
    step_dur = int(st.session_state.get("step_duration_minutes", 5))
    advance_pipeline_step(step_minutes=step_dur)
    st.rerun()
