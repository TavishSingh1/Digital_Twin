"""
BEMS UI Components & Theme Styling
==================================
Industrial IoT Building Energy Management System (BEMS) components:
- Professional control-room CSS theme
- Global Simulation Control Toolbar (Play/Pause, Step, Duration, Clock, Status)
- Native confirmation dialogs (Reset, Scenario Loading, Help Guide)
- Synchronized Room Summary Cards
- Action feedback banners
"""

import time
from typing import Optional, Any
import streamlit as st
import pandas as pd

from simulation.config import EXPECTED_ROOM_IDS
from dashboard.state import (
    get_pipeline,
    get_selected_room,
    set_selected_room,
    advance_pipeline_step,
    reset_simulation,
    clear_action_feedback,
)
from dashboard.scenarios import SCENARIO_PRESETS, get_scenario_metadata, apply_scenario


def inject_bems_theme():
    """Injects professional industrial IoT control-room CSS styles."""
    st.markdown(
        """
        <style>
        /* Modern Industrial IoT Control-Room Aesthetic */
        .bems-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 16px;
            background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid #334155;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .bems-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #F8FAFC;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .bems-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .badge-running { background: #064E3B; color: #34D399; border: 1px solid #059669; }
        .badge-paused { background: #78350F; color: #FBBF24; border: 1px solid #D97706; }
        .badge-idle { background: #1E293B; color: #94A3B8; border: 1px solid #475569; }
        .badge-sim { background: #1E1B4B; color: #818CF8; border: 1px solid #4338CA; }
        .badge-pred { background: #082F49; color: #38BDF8; border: 1px solid #0284C7; }
        .badge-eco { background: #064E3B; color: #34D399; border: 1px solid #10B981; }
        .badge-demand { background: #7F1D1D; color: #F87171; border: 1px solid #EF4444; }

        /* Toolbar Container */
        .bems-toolbar {
            background: #0F172A;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 14px;
        }

        /* Room Summary Card */
        .room-card-box {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            transition: all 0.2s ease-in-out;
        }
        .room-card-selected {
            background: #0F2A38 !important;
            border: 2px solid #00F0FF !important;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.25);
        }
        .room-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 2px;
        }
        .room-card-sub {
            font-size: 0.75rem;
            color: #94A3B8;
            margin-bottom: 6px;
        }
        .room-card-stat {
            font-size: 0.8rem;
            color: #CBD5E1;
            margin: 2px 0;
            display: flex;
            justify-content: space-between;
        }

        /* Control Panel Container */
        .panel-container {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #E5E7EB;
            margin-bottom: 8px;
            border-bottom: 1px solid #374151;
            padding-bottom: 4px;
        }

        /* Explainability Callout */
        .explain-box {
            background: #182234;
            border-left: 4px solid #38BDF8;
            padding: 10px 14px;
            border-radius: 0 6px 6px 0;
            margin: 8px 0;
        }

        /* Clean metrics formatting */
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            color: #94A3B8 !important;
        }

        /* Responsive Layout: Stack columns gracefully on narrower viewports */
        @media (max-width: 1100px) {
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("⚠️ Confirm Simulation Reset")
def confirm_reset_dialog():
    """Native Streamlit dialog for safe simulation reset."""
    st.markdown(
        "**Resetting will reinitialize the discrete simulation environment:**\n"
        "- All occupants will be reset to initial positions.\n"
        "- Accumulated environmental sensor telemetry will be cleared.\n"
        "- Cumulative kWh energy meters and baseline comparative logs will be reset.\n"
        "- STGNN 60-minute historical lookback buffer will be re-initialized."
    )
    st.markdown("---")
    c1, c2 = st.columns(2)
    pop = c1.slider("Initial Population", min_value=0, max_value=120, value=st.session_state.get("num_occupants", 40), step=5)
    seed = c2.number_input("RNG Seed", min_value=1, max_value=999999, value=st.session_state.get("seed", 42), step=1)

    c_btn1, c_btn2 = st.columns(2)
    if c_btn1.button("Confirm Reset", type="primary", use_container_width=True):
        reset_simulation(seed=int(seed), num_occupants=int(pop))
        st.session_state.is_auto_playing = False
        st.toast(f"Simulation reset with seed={seed}, population={pop}")
        st.rerun()
    if c_btn2.button("Cancel", use_container_width=True):
        st.rerun()


@st.dialog("🎬 Load Simulation Scenario")
def scenario_selector_dialog():
    """Native dialog for selecting and previewing operational demonstration scenarios."""
    st.markdown("Select a documented operational scenario to evaluate building dynamics and HVAC response:")

    scenario_options = {k: v["name"] for k, v in SCENARIO_PRESETS.items()}
    chosen_key = st.selectbox(
        "Available Scenarios",
        options=list(scenario_options.keys()),
        format_func=lambda k: scenario_options[k],
        index=0,
    )

    meta = get_scenario_metadata(chosen_key)

    st.markdown(f"### 📋 {meta['name']}")
    st.caption(meta["description"])

    p_c1, p_c2, p_c3 = st.columns(3)
    p_c1.metric("Occupants", f"{meta['occupants']} people")
    p_c2.metric("Target Setpoint", f"{meta['target_setpoint']} °C")
    p_c3.metric("Outdoor Temp", f"{meta['outdoor_mean_temp']} °C")

    st.info(f"**Expected Power Demand:** `{meta['expected_power_kw']}`\n\n**Expected CO₂ Range:** `{meta['expected_co2_ppm']}`")
    st.markdown(f"**Engineering Impact:** {meta['impact_summary']}")

    st.markdown("---")
    btn1, btn2 = st.columns(2)
    if btn1.button("Apply Scenario", type="primary", use_container_width=True):
        new_pipeline = apply_scenario(chosen_key)
        st.session_state.pipeline = new_pipeline
        st.session_state.active_scenario = meta["name"]
        st.session_state.num_occupants = meta["occupants"]
        st.session_state.seed = meta["seed"]
        st.session_state.is_auto_playing = False
        st.session_state.action_feedback = ("success", f"Loaded '{meta['name']}' scenario successfully ({meta['occupants']} occupants).")
        st.rerun()
    if btn2.button("Cancel", use_container_width=True):
        st.rerun()


@st.dialog("ℹ️ Building Energy Management System Guide")
def bems_help_dialog():
    """Displays operator reference and system architecture guide."""
    st.markdown("### 🏢 5-Room Building Energy Management System (BEMS)")
    st.markdown(
        "This Digital Twin integrates a **five-room physical building simulation**, multi-physics virtual environmental "
        "sensors, real-time smart HVAC & lighting energy management, and a **Spatio-Temporal Graph Neural Network (STGNN)** "
        "predicting occupancy 15 minutes into the future."
    )
    st.markdown("---")
    st.markdown("#### 🧭 Operational Workspaces:")
    st.markdown(
        "1. **Building Overview**: Visual 2D floor plan, interactive room selection, occupancy distribution, and quick room cards.\n"
        "2. **People Flow Control**: Gate entry injection, real-time topological BFS door transfers, and exits.\n"
        "3. **Sensor Telemetry**: 7-channel multi-physics telemetry (Immediate, Dynamic, Outdoor) with history windows.\n"
        "4. **Smart HVAC & Energy**: Real-time power breakdown, thermal sensible heat gains (100W/occ), COP 3.2, daylight harvesting, and baseline comparative savings.\n"
        "5. **Predictive Evaluation**: STGNN 15-minute forecasting benchmarked against Persistence.\n"
        "6. **Data Export**: Validated audit CSV/JSON telemetry downloads."
    )
    st.markdown("---")
    st.markdown("#### ⚡ Simulation Controls:")
    st.markdown(
        "- **Play / Pause**: Start or stop continuous automatic stepping.\n"
        "- **Step**: Advance by discrete simulation intervals (+5m, +15m, +30m, +60m).\n"
        "- **Scenarios**: Test predefined operational loads (Peak surge, Empty building, Heatwave, Energy saving).\n"
        "- **Room Selection**: Click any room card or selector to open the **Selected Room Control Centre**."
    )


def render_action_feedback():
    """Renders temporary user feedback banner if present."""
    feedback = st.session_state.get("action_feedback")
    if feedback:
        f_type, f_msg = feedback
        col_msg, col_dismiss = st.columns([0.92, 0.08])
        with col_msg:
            if f_type == "success":
                st.success(f_msg)
            elif f_type == "warning":
                st.warning(f_msg)
            elif f_type == "error":
                st.error(f_msg)
            else:
                st.info(f_msg)
        with col_dismiss:
            if st.button("✕", help="Dismiss message", key="btn_dismiss_feedback"):
                clear_action_feedback()
                st.rerun()


def render_simulation_toolbar(pipeline: Optional[Any]):
    """Renders the top persistent simulation control toolbar."""
    if not pipeline:
        st.warning("⚠️ Simulation pipeline not initialized.")
        return

    curr_dt = pipeline.engine.current_time
    is_playing = st.session_state.get("is_auto_playing", False)
    step_count = pipeline.engine.step_count

    # Determine status badge
    if is_playing:
        status_html = "<span class='bems-badge badge-running'>▶ RUNNING (Auto-Step)</span>"
    elif step_count > 0:
        status_html = "<span class='bems-badge badge-paused'>⏸ PAUSED</span>"
    else:
        status_html = "<span class='bems-badge badge-idle'>⏹ IDLE</span>"

    active_scen = st.session_state.get("active_scenario", "Normal Workday")

    # Header row
    col_head_left, col_head_right = st.columns([2.5, 1.5])
    with col_head_left:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:6px;'>"
            f"<span style='font-size:1.25rem; font-weight:700;'>🏢 BEMS Command Centre</span>"
            f"{status_html}"
            f"<span class='bems-badge badge-sim'>Physics Engine: Active</span>"
            f"<span class='bems-badge badge-pred'>STGNN: 15-Min Forecast</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_head_right:
        c_scen, c_help = st.columns([1.5, 1])
        with c_scen:
            if st.button(f"🎬 Scenario: {active_scen}", use_container_width=True, help="Load demonstration presets"):
                scenario_selector_dialog()
        with c_help:
            if st.button("ℹ️ Help / Guide", use_container_width=True, help="Operator Guide"):
                bems_help_dialog()

    # Toolbar Controls Container
    with st.container():
        st.markdown("<div class='bems-toolbar'>", unsafe_allow_html=True)
        tb1, tb2, tb3, tb4, tb5, tb6, tb7 = st.columns([1.1, 1.1, 1.4, 1.4, 1.8, 1.0, 1.0])

        with tb1:
            if is_playing:
                if st.button("⏸ Pause", use_container_width=True, type="secondary"):
                    st.session_state.is_auto_playing = False
                    st.rerun()
            else:
                if st.button("▶ Play", use_container_width=True, type="primary"):
                    st.session_state.is_auto_playing = True
                    st.rerun()

        with tb2:
            step_dur = st.session_state.get("step_duration_minutes", 5)
            step_label = f"+{step_dur}m" if step_dur > 0 else "+5m"
            if st.button(f"⏭ Step ({step_label})", use_container_width=True, help=f"Advance simulation by {step_dur} minutes"):
                advance_pipeline_step(step_minutes=step_dur)
                st.rerun()

        with tb3:
            chosen_dur = st.selectbox(
                "Step Size",
                options=[5, 15, 30, 60],
                index=[5, 15, 30, 60].index(st.session_state.get("step_duration_minutes", 5)) if st.session_state.get("step_duration_minutes", 5) in [5, 15, 30, 60] else 0,
                format_func=lambda x: f"{x} min ({x // 5} tick{'s' if x // 5 > 1 else ''})" + (" [Horizon]" if x == 15 else (" [Lookback]" if x == 60 else "")),
                key="sb_step_duration",
                label_visibility="collapsed",
            )
            st.session_state.step_duration_minutes = chosen_dur

        with tb4:
            speed_val = st.selectbox(
                "Auto Speed",
                options=[0.5, 1.0, 2.0, 3.0],
                index=1,
                format_func=lambda x: f"{x}s per step",
                key="sb_auto_speed",
                label_visibility="collapsed",
            )
            st.session_state.playback_speed = speed_val

        with tb5:
            ts_str = curr_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            st.markdown(
                f"<div style='font-size:0.85rem; padding-top:4px;'>"
                f"🕒 <b>{ts_str}</b><br>"
                f"<span style='color:#94A3B8; font-size:0.75rem;'>Step <b>{step_count}</b> | Active: <b>{pipeline.engine.building.total_occupants}</b> occ</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with tb6:
            if st.button("+15m Horizon", use_container_width=True, help="Advance 3 steps = 1 forecast horizon"):
                advance_pipeline_step(step_minutes=15)
                st.rerun()

        with tb7:
            if st.button("🔄 Reset", use_container_width=True, help="Reset simulation state with confirmation"):
                confirm_reset_dialog()

        st.markdown("</div>", unsafe_allow_html=True)


def render_room_summary_cards(pipeline: Any, selected_room: str):
    """
    Renders 5 synchronized room summary cards. Clicking a card selects that room.
    """
    latest_energy = pipeline.engine.energy_manager.get_latest_states() if hasattr(pipeline.engine, "energy_manager") else {}
    sensor_df = pipeline.engine.get_sensor_readings_df()

    st.markdown("#### 🏢 Facility Rooms Overview & Selector")
    st.caption("Click any room card below to focus the floor plan and open the **Selected Room Control Centre**:")

    cols = st.columns(5)
    for idx, r_id in enumerate(EXPECTED_ROOM_IDS):
        room = pipeline.engine.building.get_room(r_id)
        is_sel = (r_id == selected_room)

        # Get energy state
        e_state = latest_energy.get(r_id)
        hvac_m = getattr(e_state, "hvac_mode", "AUTO") if e_state else "AUTO"
        p_w = getattr(e_state, "total_power_w", 0.0) if e_state else 0.0
        cum_kwh = getattr(e_state, "cumulative_total_kwh", 0.0) if e_state else 0.0

        # Get latest sensor readings
        t_in = 24.5
        co2 = 450.0
        if not sensor_df.empty:
            rdf = sensor_df[sensor_df["room_id"] == r_id]
            if not rdf.empty:
                t_in = float(rdf.iloc[-1].get("indoor_temperature", 24.5))
                co2 = float(rdf.iloc[-1].get("indoor_co2", 450.0))

        count = room.occupant_count
        cap = room.capacity
        util_pct = (count / cap * 100.0) if cap > 0 else 0.0

        with cols[idx]:
            card_class = "room-card-box room-card-selected" if is_sel else "room-card-box"
            sel_badge = "<span style='color:#00F0FF; font-size:0.75rem; font-weight:700;'>★ SELECTED</span>" if is_sel else ""

            # Card HTML
            st.markdown(
                f"<div class='{card_class}'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span class='room-card-title'>{r_id}</span>"
                f"{sel_badge}"
                f"</div>"
                f"<div class='room-card-sub'>{room.name}</div>"
                f"<div class='room-card-stat'><span>Occupants:</span><b>{count} / {cap} ({util_pct:.0f}%)</b></div>"
                f"<div class='room-card-stat'><span>Temp / CO₂:</span><b>{t_in:.1f}°C | {co2:.0f} ppm</b></div>"
                f"<div class='room-card-stat'><span>HVAC Mode:</span><b>{hvac_m}</b></div>"
                f"<div class='room-card-stat'><span>Power / Energy:</span><b>{p_w:.0f}W | {cum_kwh:.2f}kWh</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            btn_label = f"✓ Focused" if is_sel else f"Focus {r_id}"
            if st.button(btn_label, key=f"btn_select_room_{r_id}", use_container_width=True, type="primary" if is_sel else "secondary"):
                set_selected_room(r_id)
                st.rerun()

