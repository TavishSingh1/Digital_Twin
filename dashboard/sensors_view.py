"""
Environmental Sensor Telemetry Workspace View
=============================================
Comprehensive multi-physics telemetry monitoring and diagnostics:
- Live cross-room sensor matrix table with units and lighting state
- Physical signal classification (Immediate, Dynamic, Outdoor)
- Historical time-series plots with configurable time windows
- Summary statistics (Current, Min, Max, Average) for selected telemetry window
- Physical model parameters and diagnostic state inspector
"""

from typing import Any, Optional
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np

from simulation.config import EXPECTED_ROOM_IDS
from stgnn_model import COMMON_SENSOR_FEATURES
from dashboard.state import get_selected_room, set_selected_room
from dashboard.charts import create_sensor_charts, FEATURE_UNITS


def render_sensors_view(pipeline: Any):
    """Renders the Environmental Telemetry workspace."""
    if not pipeline:
        st.warning("⚠️ Simulation pipeline is not initialized.")
        return

    st.subheader("🌡️ Virtual Environmental Sensor Engine & Telemetry")
    st.caption("Governed by thermodynamics, conservation of mass, psychrometrics, and acoustic decibel power laws.")

    # 1. Signal Classification Architecture Badges
    sc_c1, sc_c2, sc_c3 = st.columns(3)
    with sc_c1:
        st.markdown(
            "<div style='background:#1E293B; border-left:4px solid #F59E0B; padding:10px; border-radius:4px;'>"
            "<b>⚡ Immediate Response</b><br>"
            "<span style='font-size:0.8rem; color:#CBD5E1;'>"
            "• <b>Illuminance (lux)</b>: Workplane daylight + policy<br>"
            "• <b>Sound (dBA)</b>: Logarithmic acoustic power summation"
            "</span></div>",
            unsafe_allow_html=True,
        )
    with sc_c2:
        st.markdown(
            "<div style='background:#1E293B; border-left:4px solid #10B981; padding:10px; border-radius:4px;'>"
            "<b>⏳ Dynamic Integration (5-min ticks)</b><br>"
            "<span style='font-size:0.8rem; color:#CBD5E1;'>"
            "• <b>Indoor Temp (°C)</b>: Thermal mass, HVAC & sensible heat<br>"
            "• <b>Indoor RH (%)</b>: Psychrometric vapor balance<br>"
            "• <b>Indoor CO₂ (ppm)</b>: Metabolic mass balance & decay"
            "</span></div>",
            unsafe_allow_html=True,
        )
    with sc_c3:
        st.markdown(
            "<div style='background:#1E293B; border-left:4px solid #38BDF8; padding:10px; border-radius:4px;'>"
            "<b>⛅ Outdoor Meteorological Drivers</b><br>"
            "<span style='font-size:0.8rem; color:#CBD5E1;'>"
            "• <b>Dry-Bulb Temp (°C)</b>: Diurnal ambient weather cycle<br>"
            "• <b>Outdoor RH (%)</b>: Anti-correlated atmospheric humidity<br>"
            "*(Independent of indoor human occupancy)*"
            "</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    sensor_df = pipeline.engine.get_sensor_readings_df()

    # 2. Live Cross-Room Snapshot Table
    st.markdown("#### 🏢 Live Cross-Room Telemetry Snapshot")
    if not sensor_df.empty:
        cross_room_data = []
        for r_id in EXPECTED_ROOM_IDS:
            room = pipeline.engine.building.get_room(r_id)
            meta = pipeline.engine.sensor_manager.get_sensor_metadata(r_id)
            rdf = sensor_df[sensor_df["room_id"] == r_id]
            if not rdf.empty:
                last_row = rdf.iloc[-1]
                light_badge = "💡 ON" if meta.get("is_light_on") else "🌑 OFF"
                cross_room_data.append({
                    "Room": r_id,
                    "Name": room.name,
                    "Occupants": f"{room.occupant_count} / {room.capacity}",
                    "Lighting": light_badge,
                    "Illuminance (lux)": f"{last_row['illuminance']:.1f}",
                    "Sound (dBA)": f"{last_row['sound_pressure_level']:.1f}",
                    "Indoor Temp (°C)": f"{last_row['indoor_temperature']:.2f}",
                    "Indoor RH (%)": f"{last_row['indoor_relative_humidity']:.1f}",
                    "Indoor CO₂ (ppm)": f"{last_row['indoor_co2']:.1f}",
                    "Dry Bulb Temp (°C)": f"{last_row['dry_bulb_temperature']:.2f}",
                    "Outdoor RH (%)": f"{last_row['outdoor_relative_humidity']:.1f}",
                })
        st.dataframe(pd.DataFrame(cross_room_data), use_container_width=True, hide_index=True)
    else:
        st.info("No sensor records logged yet. Advance the simulation to collect telemetry.")

    st.markdown("---")

    # 3. Interactive Historical Telemetry & Statistics
    st.markdown("#### 📈 Historical Telemetry Analysis")

    curr_sel = get_selected_room()
    c_rm, c_win, c_feat = st.columns([1, 1, 2])

    with c_rm:
        def_idx = EXPECTED_ROOM_IDS.index(curr_sel) if curr_sel in EXPECTED_ROOM_IDS else 0
        chosen_room = st.selectbox(
            "Select Room to Inspect",
            options=EXPECTED_ROOM_IDS,
            index=def_idx,
            key="sb_sensor_hist_room",
        )
        if chosen_room != curr_sel:
            set_selected_room(chosen_room)

    with c_win:
        chosen_window = st.selectbox(
            "Time Window",
            options=["Last 1 hour (12 steps)", "Last 3 hours (36 steps)", "Last 6 hours (72 steps)", "All Recorded History"],
            index=3,
            key="sb_sensor_window",
        )

    with c_feat:
        default_feats = ["indoor_temperature", "dry_bulb_temperature", "indoor_co2", "illuminance", "sound_pressure_level"]
        selected_feats = st.multiselect(
            "Telemetry Series to Plot",
            options=COMMON_SENSOR_FEATURES,
            default=default_feats,
            key="ms_sensor_feats",
        )

    if not sensor_df.empty:
        rdf = sensor_df[sensor_df["room_id"] == chosen_room].copy()
        if not rdf.empty:
            # Apply time window filtering
            if "1 hour" in chosen_window:
                rdf_filtered = rdf.tail(12)
            elif "3 hours" in chosen_window:
                rdf_filtered = rdf.tail(36)
            elif "6 hours" in chosen_window:
                rdf_filtered = rdf.tail(72)
            else:
                rdf_filtered = rdf

            # Render Window Statistics Table
            st.markdown(f"##### 📊 Window Statistics: {chosen_room} ({chosen_window})")
            stat_rows = []
            for f in selected_feats:
                if f in rdf_filtered.columns:
                    s = rdf_filtered[f]
                    u = FEATURE_UNITS.get(f, "")
                    stat_rows.append({
                        "Feature": f.replace("_", " ").title(),
                        "Unit": u,
                        "Current": f"{s.iloc[-1]:.2f} {u}",
                        "Minimum": f"{s.min():.2f} {u}",
                        "Maximum": f"{s.max():.2f} {u}",
                        "Average": f"{s.mean():.2f} {u}",
                    })
            if stat_rows:
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

            # Render Chart
            fig = create_sensor_charts(
                sensor_df=rdf_filtered,
                room_id=chosen_room,
                selected_features=selected_feats,
                dark_mode=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Diagnostic Expander
            meta = pipeline.engine.sensor_manager.get_sensor_metadata(chosen_room)
            with st.expander(f"🔬 Physical Parameters & Calibration Constants ({chosen_room})", expanded=False):
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.markdown("**💡 Lighting Parameters**")
                    st.write(f"- Lamp State: `{'ON' if meta.get('is_light_on') else 'OFF'}`")
                    st.write(f"- Daylight Lux: `{meta.get('daylight_lux', 0.0):.1f} lx`")
                    st.write(f"- Artificial Lux: `{meta.get('artificial_lux', 0.0):.1f} lx`")
                    st.write(f"- Vacancy Timer: `{meta.get('vacancy_timer_minutes', 0.0):.1f} / 10.0 min`")
                with d2:
                    st.markdown("**🔊 Acoustic Model**")
                    st.write(f"- Noise Floor: `{meta.get('sound_base_dba', 50.0):.1f} dBA`")
                    st.write(f"- Per-Person Contribution: `{meta.get('sound_occ_dba', 54.0):.1f} dBA`")
                with d3:
                    st.markdown("**💨 Ventilation & Envelope**")
                    st.write(f"- Room Volume: `{meta.get('volume_m3', 300.0):.1f} m³`")
                    st.write(f"- Air Exchange: `{meta.get('co2_ach', 0.8):.2f} ACH`")
                    st.write(f"- Thermal Capacitance: `{meta.get('thermal_capacitance_kj_k', 3600.0):.0f} kJ/K`")
        else:
            st.warning(f"No telemetry found for {chosen_room}.")

