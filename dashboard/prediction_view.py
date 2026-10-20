"""
Predictive Evaluation Workspace View
====================================
Spatio-Temporal Graph Neural Network (STGNN) 15-minute occupancy forecasting workspace:
- Non-ML executive summary explaining the graph-neural spatial forecasting mechanism
- Live comparison between ground-truth occupancy and 15-minute forecasts
- Quantitative benchmark metrics (Integer MAE, RMSE, Persistence Baseline)
- Residual prediction error plots and comparative bar charts
- Domain generalization notice explaining real-world trained weights vs physics simulation
"""

from typing import Any, Optional
import streamlit as st
import pandas as pd

from simulation.config import EXPECTED_ROOM_IDS
from dashboard.charts import (
    create_metrics_comparison_chart,
    create_error_chart,
)


def render_prediction_view(pipeline: Any):
    """Renders the Predictive Evaluation workspace."""
    if not pipeline:
        st.warning("⚠️ Simulation pipeline is not initialized.")
        return

    st.subheader("🔮 STGNN Predictive Occupancy Forecasting & Benchmarks")
    st.caption(
        "Joint Improved Spatio-Temporal Graph Neural Network forecasting discrete human occupancy "
        "15 minutes ahead (3 steps) from a 60-minute rolling historical window (12 steps × 13 features)."
    )

    # 1. Non-ML Executive Summary Box
    st.markdown(
        "<div style='background:#1E293B; border-left:4px solid #38BDF8; padding:12px; border-radius:4px; margin-bottom:14px;'>"
        "<span style='font-size:0.95rem; font-weight:700; color:#F8FAFC;'>💡 How STGNN Forecasting Operates:</span><br>"
        "<span style='font-size:0.85rem; color:#CBD5E1;'>"
        "The model analyzes inter-room doorway topology (spatial graph) and multi-variate environmental changes "
        "(CO₂, sound, temperature, lighting) over the preceding 60 minutes to estimate where building occupants will be "
        "15 minutes into the future. Forecasts represent statistical estimations; ground-truth physics always governs "
        "actual facility operations."
        "</span></div>",
        unsafe_allow_html=True,
    )

    # 2. Live Current vs 15-Min Forecast Comparison Cards
    st.markdown("#### ⏱️ Real-Time 15-Minute Horizon Comparison")

    pred_records = pipeline.predictions[-5:] if pipeline.predictions else []
    latest_status = pred_records[0].status if pred_records else "INITIALIZING"

    if latest_status == "WARMUP":
        buf_len = len(pipeline.rolling_buffer)
        st.info(f"⏳ **Inference Pipeline in Warm-Up**: Accumulating 60-minute lookback buffer ({buf_len}/12 steps). Forecasting initiates once 12 steps (60 min) are recorded.")
    elif latest_status == "SUCCESS":
        target_ts = pred_records[0].target_timestamp if pred_records else "N/A"
        st.caption(f"Forecasting logical target: `{target_ts}` (Forecast made at: `{pred_records[0].forecast_made_at}`)")

        pred_map = {rec.room_id: rec for rec in pred_records}
        p_cols = st.columns(5)
        for idx, r_id in enumerate(EXPECTED_ROOM_IDS):
            room = pipeline.engine.building.get_room(r_id)
            rec = pred_map.get(r_id)
            with p_cols[idx]:
                st.markdown(f"**{r_id}**")
                st.caption(room.name)
                curr_occ = room.occupant_count
                st.metric("Current Actual", f"{curr_occ} occ")

                if rec and rec.status == "SUCCESS":
                    pred_val = rec.predicted_occupants
                    delta_occ = pred_val - curr_occ
                    delta_str = f"{delta_occ:+d} forecast" if delta_occ != 0 else "0 change"
                    st.metric(
                        "15-Min Forecast",
                        f"{pred_val} occ",
                        delta=delta_str,
                        help=f"Continuous model output: {rec.predicted_continuous:.2f} | Persistence: {rec.persistence_occupants}",
                    )
                else:
                    st.metric("15-Min Forecast", "Pending")

    st.markdown("---")

    # 3. Quantitative Predictive Evaluation Metrics
    st.markdown("#### 📊 Quantitative Benchmark Parity")

    metrics = pipeline.compute_evaluation_metrics()
    total_eval = metrics.get("total_evaluated_samples", 0)

    if total_eval > 0:
        b_met = metrics["building_metrics"]
        r_met = metrics["room_metrics"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evaluated Predictions", f"{total_eval} samples", help="Predictions compared after clock reached target timestamp")
        m2.metric(
            "STGNN Integer MAE",
            f"{b_met['stgnn_integer_mae']:.3f} occ",
            delta=f"{b_met['stgnn_integer_mae'] - b_met['persistence_mae']:+.3f} vs Persistence",
            delta_color="inverse",
            help="Mean Absolute Error of rounded integer predictions",
        )
        m3.metric("STGNN Integer RMSE", f"{b_met['stgnn_integer_rmse']:.3f} occ", help="Root Mean Square Error of predictions")
        m4.metric("Persistence Baseline MAE", f"{b_met['persistence_mae']:.3f} occ", help="Naive baseline predicting no change over 15 min")

        st.markdown("##### Per-Room Performance Breakdown")
        table_rows = []
        for r_id in EXPECTED_ROOM_IDS:
            rm = r_met.get(r_id, {})
            table_rows.append({
                "Room ID": r_id,
                "Room Name": pipeline.engine.building.get_room(r_id).name,
                "Evaluated Samples": rm.get("samples", 0),
                "STGNN Integer MAE": f"{rm.get('stgnn_int_mae', 0):.3f}",
                "Persistence MAE": f"{rm.get('persistence_mae', 0):.3f}",
                "STGNN Integer RMSE": f"{rm.get('stgnn_int_rmse', 0):.3f}",
                "Persistence RMSE": f"{rm.get('persistence_rmse', 0):.3f}",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 4. Comparative Visualizations
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(create_metrics_comparison_chart(metrics, dark_mode=True), use_container_width=True)
        with col_c2:
            eval_room = st.selectbox(
                "Filter Residual Error Plot",
                options=["All Rooms (Aggregate)"] + EXPECTED_ROOM_IDS,
                index=0,
                key="sb_pred_eval_room",
            )
            err_room = None if eval_room == "All Rooms (Aggregate)" else eval_room
            st.plotly_chart(create_error_chart(pipeline.get_predictions_df(), room_id=err_room, dark_mode=True), use_container_width=True)

        # Domain Generalization Notice
        st.warning(
            "**Domain Generalization & Model Assumption Notice:**\n\n"
            "1. **Mathematical Implementation Parity**: Model architecture, custom GraphConv weights, and scalers reproduce "
            "notebook evaluations with bitwise accuracy (MAE ≤ 1.78e-7).\n"
            "2. **Synthetic vs Empirical Domain Gap**: The neural weights were fitted on empirical historical campus data. "
            "In contrast, this digital twin implements continuous first-principles physics (thermodynamics, conservation of mass, acoustics). "
            "Because human building occupancy exhibits strong short-term inertia over 15-minute intervals, the Persistence Baseline "
            "serves as a highly competitive benchmark."
        )
    else:
        st.info(
            f"ℹ️ Currently evaluated samples: 0. The simulation requires at least 15 steps "
            f"(12 warm-up lookback steps + 3 prediction horizon steps) to evaluate its first forecast. "
            f"Current step: {pipeline.engine.step_count}."
        )

