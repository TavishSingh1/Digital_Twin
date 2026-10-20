"""
Data Export & Audit Logs Workspace View
=======================================
Structured data export center providing comprehensive audit datasets:
- Ground-Truth Occupancy History
- Virtual Environmental Sensor Telemetry
- STGNN Forecast Records
- Room-Wise Energy Consumption Telemetry
- Benchmark Evaluation Metrics (JSON)

Displays rich metadata (row count, time span, columns, classification) before download.
"""

from typing import Any
import json
import streamlit as st
import pandas as pd


def render_exports_view(pipeline: Any):
    """Renders the Data Export workspace."""
    if not pipeline:
        st.warning("⚠️ Simulation pipeline is not initialized.")
        return

    st.subheader("💾 Facility Telemetry Data Export & Audit Center")
    st.caption("Download synchronized ground-truth observations, virtual sensor telemetry, neural forecasts, and energy logs.")

    gt_df = pipeline.engine.recorder.to_dataframe()
    sensor_df = pipeline.engine.get_sensor_readings_df()
    pred_df = pipeline.get_predictions_df()
    energy_df = pipeline.engine.get_energy_readings_df()
    metrics_dict = pipeline.compute_evaluation_metrics()

    # Time spans
    gt_span = f"{gt_df['timestamp'].min()[:19]} → {gt_df['timestamp'].max()[:19]}" if not gt_df.empty else "N/A"
    sensor_span = f"{sensor_df['timestamp'].min()[:19]} → {sensor_df['timestamp'].max()[:19]}" if not sensor_df.empty else "N/A"
    pred_span = f"{pred_df['forecast_made_at'].min()[:19]} → {pred_df['target_timestamp'].max()[:19]}" if not pred_df.empty else "N/A"
    energy_span = f"{energy_df['timestamp'].min()[:19]} → {energy_df['timestamp'].max()[:19]}" if not energy_df.empty else "N/A"

    streams = [
        {
            "name": "Ground Truth Occupancy",
            "filename": "ground_truth_occupancy.csv",
            "desc": "Authoritative room-by-room discrete occupant counts recorded at 5-minute intervals.",
            "type": "Physical Ground Truth",
            "badge": "badge-sim",
            "df": gt_df,
            "data_str": gt_df.to_csv(index=False),
            "mime": "text/csv",
            "rows": len(gt_df),
            "span": gt_span,
            "cols": list(gt_df.columns),
        },
        {
            "name": "Virtual Sensor Telemetry",
            "filename": "virtual_sensor_readings.csv",
            "desc": "Continuous 7-channel physical sensor readings (temperature, RH, CO₂, illuminance, sound).",
            "type": "First-Principles Physics",
            "badge": "badge-sim",
            "df": sensor_df,
            "data_str": sensor_df.to_csv(index=False),
            "mime": "text/csv",
            "rows": len(sensor_df),
            "span": sensor_span,
            "cols": list(sensor_df.columns),
        },
        {
            "name": "STGNN Neural Predictions",
            "filename": "occupancy_predictions.csv",
            "desc": "Spatio-Temporal Graph Neural Network 15-minute forecasts, persistence baseline, and targets.",
            "type": "Neural Network Inference",
            "badge": "badge-pred",
            "df": pred_df,
            "data_str": pred_df.to_csv(index=False),
            "mime": "text/csv",
            "rows": len(pred_df),
            "span": pred_span,
            "cols": list(pred_df.columns),
        },
        {
            "name": "Room Energy Telemetry",
            "filename": "room_energy_consumption.csv",
            "desc": "Room-by-room instantaneous powers (W), cumulative energy (kWh), and HVAC operating modes.",
            "type": "Simulated Electrical Telemetry",
            "badge": "badge-eco",
            "df": energy_df,
            "data_str": energy_df.to_csv(index=False),
            "mime": "text/csv",
            "rows": len(energy_df),
            "span": energy_span,
            "cols": list(energy_df.columns),
        },
    ]

    st.markdown("#### 📦 Available Export Streams")

    cols = st.columns(2)
    for idx, s in enumerate(streams):
        col = cols[idx % 2]
        with col:
            st.markdown(
                f"<div style='background:#1E293B; border:1px solid #334155; border-radius:8px; padding:12px; margin-bottom:12px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='font-size:1.05rem; font-weight:700; color:#F8FAFC;'>📄 {s['name']}</span>"
                f"<span class='bems-badge {s['badge']}'>{s['type']}</span>"
                f"</div>"
                f"<div style='font-size:0.8rem; color:#94A3B8; margin:4px 0 8px 0;'>{s['desc']}</div>"
                f"<div style='font-size:0.8rem; color:#CBD5E1;'>"
                f"• Rows: <b>{s['rows']}</b> | Time Span: <b>{s['span']}</b><br>"
                f"• Schema: <code>{', '.join(s['cols'][:6])}{'...' if len(s['cols']) > 6 else ''}</code>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                label=f"⬇️ Download {s['filename']}",
                data=s["data_str"],
                file_name=s["filename"],
                mime=s["mime"],
                key=f"dl_btn_{idx}",
                use_container_width=True,
                disabled=(s["rows"] == 0),
            )

    # 5th Stream: Metrics JSON
    st.markdown("---")
    m_c1, m_c2 = st.columns([2, 1])
    with m_c1:
        st.markdown(
            f"<div style='background:#1E293B; border:1px solid #334155; border-radius:8px; padding:12px;'>"
            f"<span style='font-size:1.05rem; font-weight:700; color:#F8FAFC;'>📊 Benchmark Metrics Summary (JSON)</span><br>"
            f"<span style='font-size:0.8rem; color:#94A3B8;'>Aggregated quantitative evaluation metrics (STGNN MAE/RMSE vs Persistence Baseline).</span><br>"
            f"<span style='font-size:0.8rem; color:#CBD5E1;'>Total Evaluated Samples: <b>{metrics_dict.get('total_evaluated_samples', 0)}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m_c2:
        st.download_button(
            label="⬇️ Download evaluation_metrics.json",
            data=json.dumps(metrics_dict, indent=2),
            file_name="evaluation_metrics.json",
            mime="application/json",
            key="dl_btn_metrics",
            use_container_width=True,
        )

    st.markdown("---")

    # Interactive Table Previews
    st.markdown("#### 🔍 Active Simulation Telemetry Previews")
    preview_choice = st.radio(
        "Select Table to Preview:",
        ["Ground Truth Table", "Virtual Sensor Table", "Predictions Table", "Room Energy Table"],
        horizontal=True,
        key="rb_export_preview",
    )

    if preview_choice == "Ground Truth Table":
        st.dataframe(gt_df.tail(25), use_container_width=True)
    elif preview_choice == "Virtual Sensor Table":
        st.dataframe(sensor_df.tail(25), use_container_width=True)
    elif preview_choice == "Predictions Table":
        st.dataframe(pred_df.tail(25), use_container_width=True)
    elif preview_choice == "Room Energy Table":
        st.dataframe(energy_df.tail(25), use_container_width=True)

