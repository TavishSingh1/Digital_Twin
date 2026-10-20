"""
Dashboard Plotly Charts & Visualization Utilities
=================================================
Provides modular, interactive time-series plots for occupancy tracking,
environmental sensor telemetry, predictive error analysis, and benchmark metrics.
"""

from typing import Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation.config import EXPECTED_ROOM_IDS
from stgnn_model import COMMON_SENSOR_FEATURES


FEATURE_UNITS = {
    "illuminance": "lux",
    "sound_pressure_level": "dBA",
    "indoor_temperature": "°C",
    "dry_bulb_temperature": "°C",
    "indoor_relative_humidity": "%",
    "outdoor_relative_humidity": "%",
    "indoor_co2": "ppm",
}

FEATURE_COLORS = {
    "illuminance": "#F39C12",
    "sound_pressure_level": "#E67E22",
    "indoor_temperature": "#E74C3C",
    "dry_bulb_temperature": "#C0392B",
    "indoor_relative_humidity": "#3498DB",
    "outdoor_relative_humidity": "#2980B9",
    "indoor_co2": "#2ECC71",
}


def create_occupancy_chart(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    room_id: Optional[str] = None,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders Ground-Truth Occupancy vs 15-Minute STGNN Forecast vs Persistence Baseline.
    """
    fig = go.Figure()

    if gt_df.empty:
        fig.add_annotation(text="No ground-truth occupancy recorded yet", showarrow=False)
        return fig

    # Filter by room or aggregate
    if room_id and room_id in EXPECTED_ROOM_IDS:
        room_gt = gt_df[gt_df["room_id"] == room_id].copy()
        room_pred = pred_df[pred_df["room_id"] == room_id].copy() if not pred_df.empty else pd.DataFrame()
        title = f"Occupancy Timeline — {room_id}"
    else:
        # Building total
        room_gt = gt_df.groupby("timestamp")["actual_occupants"].sum().reset_index()
        title = "Building-Wide Total Occupancy Timeline"
        if not pred_df.empty:
            room_pred = pred_df.groupby("target_timestamp")[["predicted_occupants", "persistence_occupants"]].sum().reset_index()
            room_pred = room_pred.rename(columns={"target_timestamp": "timestamp"})
        else:
            room_pred = pd.DataFrame()

    # 1. Ground truth trace
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(room_gt["timestamp"]),
            y=room_gt["actual_occupants"],
            mode="lines+markers",
            name="Ground Truth (Actual)",
            line=dict(color="#2ECC71", width=2.5),
            marker=dict(size=5),
        )
    )

    # 2. Predicted occupancy (aligned at target timestamp)
    if not room_pred.empty and "predicted_occupants" in room_pred.columns:
        valid_pred = room_pred[room_pred["predicted_occupants"].notna()].copy()
        target_col = "target_timestamp" if "target_timestamp" in valid_pred.columns else "timestamp"
        if not valid_pred.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(valid_pred[target_col]),
                    y=valid_pred["predicted_occupants"],
                    mode="lines+markers",
                    name="STGNN Forecast (+15 min ahead)",
                    line=dict(color="#3498DB", width=2, dash="dash"),
                    marker=dict(symbol="x", size=6),
                )
            )

    # 3. Persistence baseline
    if not room_pred.empty and "persistence_occupants" in room_pred.columns:
        valid_pers = room_pred[room_pred["persistence_occupants"].notna()].copy()
        target_col = "target_timestamp" if "target_timestamp" in valid_pers.columns else "timestamp"
        if not valid_pers.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(valid_pers[target_col]),
                    y=valid_pers["persistence_occupants"],
                    mode="lines",
                    name="Persistence Baseline",
                    line=dict(color="#95A5A6", width=1.5, dash="dot"),
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Timestamp (UTC)", showgrid=True),
        yaxis=dict(title="Occupant Count", showgrid=True, rangemode="tozero"),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_sensor_charts(
    sensor_df: pd.DataFrame,
    room_id: str,
    selected_features: list[str],
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders multi-feature environmental time-series charts for a selected room.
    """
    fig = go.Figure()

    if sensor_df.empty or not selected_features:
        fig.add_annotation(text="No telemetry recorded or no features selected", showarrow=False)
        return fig

    rdf = sensor_df[sensor_df["room_id"] == room_id].copy()
    if rdf.empty:
        fig.add_annotation(text=f"No readings for {room_id}", showarrow=False)
        return fig

    ts = pd.to_datetime(rdf["timestamp"])

    # Determine if secondary y-axis is needed (e.g. CO2/lux on large scale vs temp/RH)
    needs_secondary = any(f in ["indoor_co2", "illuminance"] for f in selected_features) and any(
        f in ["indoor_temperature", "dry_bulb_temperature", "sound_pressure_level"] for f in selected_features
    )

    if needs_secondary:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    for feat in selected_features:
        if feat not in rdf.columns:
            continue
        unit = FEATURE_UNITS.get(feat, "")
        color = FEATURE_COLORS.get(feat, "#3498DB")
        is_large_scale = feat in ["indoor_co2", "illuminance"]

        trace = go.Scatter(
            x=ts,
            y=rdf[feat],
            mode="lines+markers",
            name=f"{feat} [{unit}]",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        )

        if needs_secondary:
            fig.add_trace(trace, secondary_y=is_large_scale)
        else:
            fig.add_trace(trace)

    fig.update_layout(
        title=dict(text=f"Environmental Telemetry — {room_id}", font=dict(size=14)),
        xaxis=dict(title="Timestamp (UTC)", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if needs_secondary:
        fig.update_yaxes(title_text="Temp (°C) / Sound (dBA) / RH (%)", secondary_y=False)
        fig.update_yaxes(title_text="CO₂ (ppm) / Illuminance (lux)", secondary_y=True)
    else:
        fig.update_yaxes(title_text="Sensor Value", showgrid=True)

    return fig


def create_error_chart(
    pred_df: pd.DataFrame,
    room_id: Optional[str] = None,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders prediction residual errors over time: (Predicted - Actual) and (Persistence - Actual).
    """
    fig = go.Figure()

    if pred_df.empty:
        fig.add_annotation(text="No prediction records available", showarrow=False)
        return fig

    eval_df = pred_df[
        (pred_df["status"] == "SUCCESS") &
        (pred_df["actual_occupants_at_target"].notna()) &
        (pred_df["predicted_occupants"].notna())
    ].copy()

    if room_id and room_id in EXPECTED_ROOM_IDS:
        eval_df = eval_df[eval_df["room_id"] == room_id].copy()
        title = f"Forecast Residual Errors Over Time — {room_id}"
    else:
        title = "Forecast Residual Errors Over Time (All Rooms)"

    if eval_df.empty:
        fig.add_annotation(text="No evaluated forecasts with ground-truth observations yet", showarrow=False)
        return fig

    ts = pd.to_datetime(eval_df["target_timestamp"])
    stgnn_error = eval_df["predicted_occupants"] - eval_df["actual_occupants_at_target"]
    pers_error = eval_df["persistence_occupants"] - eval_df["actual_occupants_at_target"]

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=stgnn_error,
            mode="lines+markers",
            name="STGNN Error (Pred - Actual)",
            line=dict(color="#E74C3C", width=1.8),
            marker=dict(size=5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=pers_error,
            mode="lines+markers",
            name="Persistence Error (Pers - Actual)",
            line=dict(color="#F39C12", width=1.5, dash="dash"),
            marker=dict(symbol="diamond", size=4),
        )
    )

    # Zero reference line
    fig.add_hline(y=0, line_dash="solid", line_color="#7F8C8D", line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Target Timestamp (UTC)", showgrid=True),
        yaxis=dict(title="Occupancy Error (Occupants)", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _normalize_prediction_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names between live pipeline prediction DF and historical eval DF."""
    if df.empty:
        return pd.DataFrame()
    norm = df.copy()
    col_mapping = {
        "predicted_occupants": "predicted_integer",
        "persistence_occupants": "persistence",
        "actual_occupants_at_target": "actual",
        "trend_occupants": "recent_trend",
    }
    for old_c, new_c in col_mapping.items():
        if old_c in norm.columns and new_c not in norm.columns:
            norm[new_c] = norm[old_c]

    if "recent_trend" not in norm.columns and "persistence" in norm.columns:
        norm["recent_trend"] = norm["persistence"]

    if "actual" in norm.columns:
        norm = norm[norm["actual"].notna()]
    if "status" in norm.columns:
        norm = norm[norm["status"] == "SUCCESS"]

    return norm


def create_actual_vs_predicted_chart(
    eval_df: pd.DataFrame,
    room_id: Optional[str] = None,
    show_persistence: bool = True,
    show_trend: bool = True,
    show_continuous: bool = True,
    time_window_samples: Optional[int] = None,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders actual ground truth vs 15-minute STGNN forecasts and benchmark baselines.
    All forecasts are strictly aligned and plotted at their future target timestamp (t + 15m).
    """
    fig = go.Figure()
    norm_df = _normalize_prediction_eval_df(eval_df)
    if norm_df.empty:
        fig.add_annotation(text="No evaluated forecasts with ground-truth observations available", showarrow=False)
        fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white", height=380)
        return fig

    if room_id and room_id in EXPECTED_ROOM_IDS:
        rdf = norm_df[norm_df["room_id"] == room_id].copy()
        title = f"Ground Truth vs 15-Minute STGNN Forecast — Room {room_id}"
    else:
        agg_cols = {"actual": "sum", "predicted_integer": "sum"}
        if "predicted_continuous" in norm_df.columns:
            agg_cols["predicted_continuous"] = "sum"
        if "persistence" in norm_df.columns:
            agg_cols["persistence"] = "sum"
        if "recent_trend" in norm_df.columns:
            agg_cols["recent_trend"] = "sum"
        rdf = norm_df.groupby("target_timestamp", as_index=False).agg(agg_cols)
        title = "Ground Truth vs 15-Minute STGNN Forecast — Building Total"

    rdf = rdf.sort_values("target_timestamp")
    if time_window_samples and len(rdf) > time_window_samples:
        rdf = rdf.tail(time_window_samples)

    ts = pd.to_datetime(rdf["target_timestamp"])

    # 1. Ground Truth
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=rdf["actual"],
            mode="lines+markers",
            name="Ground Truth (Actual at t+15m)",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=6, symbol="circle"),
            hovertemplate="Time: %{x}<br>Actual: %{y} occ<extra></extra>",
        )
    )

    # 2. STGNN Discrete Integer Forecast
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=rdf["predicted_integer"],
            mode="lines+markers",
            name="STGNN Forecast (Integer)",
            line=dict(color="#38BDF8", width=2.2),
            marker=dict(size=6, symbol="cross"),
            hovertemplate="Time: %{x}<br>STGNN Forecast: %{y} occ<extra></extra>",
        )
    )

    # 3. STGNN Raw Continuous Forecast
    if show_continuous and "predicted_continuous" in rdf.columns:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=rdf["predicted_continuous"].round(2),
                mode="lines",
                name="STGNN Raw (Continuous)",
                line=dict(color="#06B6D4", width=1.5, dash="dot"),
                hovertemplate="Time: %{x}<br>STGNN Continuous: %{y:.2f} occ<extra></extra>",
            )
        )

    # 4. Persistence Baseline
    if show_persistence and "persistence" in rdf.columns:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=rdf["persistence"],
                mode="lines",
                name="Persistence Baseline (yt)",
                line=dict(color="#F59E0B", width=1.5, dash="dash"),
                hovertemplate="Time: %{x}<br>Persistence: %{y} occ<extra></extra>",
            )
        )

    # 5. Recent-Trend Baseline
    if show_trend and "recent_trend" in rdf.columns:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=rdf["recent_trend"],
                mode="lines",
                name="Recent-Trend Baseline (yt + Δ15m)",
                line=dict(color="#A855F7", width=1.5, dash="dashdot"),
                hovertemplate="Time: %{x}<br>Trend Baseline: %{y} occ<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Forecast Target Time (UTC, t + 15 min)", showgrid=True),
        yaxis=dict(title="Occupant Count", showgrid=True, rangemode="tozero"),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def create_error_distribution_chart(
    eval_df: pd.DataFrame,
    room_id: Optional[str] = None,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders prediction residual error histogram comparing STGNN against Persistence.
    """
    fig = go.Figure()
    norm_df = _normalize_prediction_eval_df(eval_df)
    if norm_df.empty:
        fig.add_annotation(text="No evaluated prediction samples available", showarrow=False)
        fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white", height=320)
        return fig

    if room_id and room_id in EXPECTED_ROOM_IDS:
        rdf = norm_df[norm_df["room_id"] == room_id].copy()
        title = f"Prediction Error Distribution — Room {room_id}"
    else:
        rdf = norm_df.copy()
        title = "Prediction Error Distribution (All Rooms)"

    stgnn_err = rdf["predicted_integer"] - rdf["actual"]
    pers_err = rdf["persistence"] - rdf["actual"]

    fig.add_trace(
        go.Histogram(
            x=stgnn_err,
            name="STGNN Error (Pred - Actual)",
            marker_color="#38BDF8",
            opacity=0.75,
            xbins=dict(start=-10.5, end=10.5, size=1.0),
        )
    )

    fig.add_trace(
        go.Histogram(
            x=pers_err,
            name="Persistence Error (Pers - Actual)",
            marker_color="#F59E0B",
            opacity=0.6,
            xbins=dict(start=-10.5, end=10.5, size=1.0),
        )
    )

    fig.add_vline(x=0, line_dash="solid", line_color="#E2E8F0", line_width=1.5)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Forecast Error (Occupants: 0 = Exact Prediction)", showgrid=True, dtick=1),
        yaxis=dict(title="Sample Frequency", showgrid=True),
        barmode="overlay",
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_responsiveness_delta_chart(
    eval_df: pd.DataFrame,
    room_id: Optional[str] = None,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders histogram of STGNN raw continuous forecast deltas (yt+3,cont - yt).
    """
    fig = go.Figure()
    if eval_df.empty:
        fig.add_annotation(text="No prediction samples available for delta analysis", showarrow=False)
        fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white", height=320)
        return fig

    rdf = eval_df.copy()
    if room_id and room_id in EXPECTED_ROOM_IDS:
        rdf = rdf[rdf["room_id"] == room_id]
        title = f"STGNN Raw Continuous Forecast Delta Distribution — Room {room_id}"
    else:
        title = "STGNN Raw Continuous Forecast Delta Distribution (All Rooms)"

    if "forecast_delta_raw" in rdf.columns:
        deltas = rdf["forecast_delta_raw"].dropna().to_numpy(dtype=float)
    elif "predicted_continuous" in rdf.columns and "persistence_occupants" in rdf.columns:
        deltas = (rdf["predicted_continuous"] - rdf["persistence_occupants"]).dropna().to_numpy(dtype=float)
    elif "predicted_continuous" in rdf.columns and "persistence" in rdf.columns:
        deltas = (rdf["predicted_continuous"] - rdf["persistence"]).dropna().to_numpy(dtype=float)
    else:
        fig.add_annotation(text="Continuous delta telemetry not available", showarrow=False)
        fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white", height=320)
        return fig

    if len(deltas) == 0:
        fig.add_annotation(text="No valid delta values computed", showarrow=False)
        fig.update_layout(template="plotly_dark" if dark_mode else "plotly_white", height=320)
        return fig

    fig.add_trace(
        go.Histogram(
            x=deltas,
            nbinsx=40,
            marker_color="#06B6D4",
            name="Continuous Delta (yt+3,cont - yt)",
            opacity=0.8,
        )
    )

    fig.add_vline(x=0, line_dash="dash", line_color="#EF4444", line_width=1.5, annotation_text="Zero Delta")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Continuous Delta (Occupants: yt+3,cont - yt)", showgrid=True),
        yaxis=dict(title="Frequency", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
        showlegend=False,
    )
    return fig


def create_metrics_comparison_chart(metrics: dict[str, Any], include_trend: bool = False, dark_mode: bool = True) -> go.Figure:
    """
    Renders bar chart comparing STGNN Integer MAE vs Persistence MAE (and optionally Recent-Trend MAE) across all rooms.
    """
    fig = go.Figure()
    room_metrics = metrics.get("room_metrics", {})
    if not room_metrics:
        fig.add_annotation(text="No evaluated samples available for metrics comparison", showarrow=False)
        return fig

    rooms = list(room_metrics.keys())
    stgnn_maes = [room_metrics[r].get("stgnn_int_mae", 0.0) for r in rooms]
    pers_maes = [room_metrics[r].get("persistence_mae", 0.0) for r in rooms]

    fig.add_trace(
        go.Bar(
            x=rooms,
            y=stgnn_maes,
            name="STGNN Integer MAE",
            marker_color="#38BDF8",
        )
    )

    fig.add_trace(
        go.Bar(
            x=rooms,
            y=pers_maes,
            name="Persistence MAE",
            marker_color="#F59E0B",
        )
    )

    if include_trend and any("trend_mae" in room_metrics[r] for r in rooms):
        trend_maes = [room_metrics[r].get("trend_mae", 0.0) for r in rooms]
        fig.add_trace(
            go.Bar(
                x=rooms,
                y=trend_maes,
                name="Recent-Trend MAE",
                marker_color="#A855F7",
            )
        )

    fig.update_layout(
        title=dict(text="Mean Absolute Error (MAE) Benchmark Comparison by Room", font=dict(size=14)),
        xaxis=dict(title="Room"),
        yaxis=dict(title="MAE (Occupants, Lower is Better)", showgrid=True),
        barmode="group",
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# -----------------------------------------------------------------------------
# PHASE 7: SMART HVAC & ENERGY VISUALIZATION CHARTS
# -----------------------------------------------------------------------------

def create_room_power_breakdown_chart(energy_df: pd.DataFrame, dark_mode: bool = True) -> go.Figure:
    """
    Renders stacked bar chart of instantaneous electrical power (HVAC, Lighting, Auxiliary) by room.
    """
    fig = go.Figure()
    if energy_df.empty:
        fig.add_annotation(text="No energy telemetry recorded yet", showarrow=False)
        return fig

    # Get latest reading for each room
    latest_ts = energy_df["timestamp"].max()
    latest_df = energy_df[energy_df["timestamp"] == latest_ts].copy()

    rooms = latest_df["room_id"].tolist()
    hvac_w = latest_df["hvac_power_w"].tolist()
    light_w = latest_df["lighting_power_w"].tolist()
    aux_w = latest_df["auxiliary_power_w"].tolist()

    fig.add_trace(go.Bar(name="HVAC Power (W)", x=rooms, y=hvac_w, marker_color="#3498DB"))
    fig.add_trace(go.Bar(name="Lighting Power (W)", x=rooms, y=light_w, marker_color="#F1C40F"))
    fig.add_trace(go.Bar(name="Auxiliary Power (W)", x=rooms, y=aux_w, marker_color="#95A5A6"))

    fig.update_layout(
        title=dict(text="Instantaneous Room Power Breakdown (W)", font=dict(size=14)),
        xaxis=dict(title="Room"),
        yaxis=dict(title="Electrical Power (Watts)", showgrid=True),
        barmode="stack",
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_cumulative_energy_trend_chart(energy_df: pd.DataFrame, dark_mode: bool = True) -> go.Figure:
    """
    Renders multi-room cumulative energy consumption (kWh) over simulation time.
    """
    fig = go.Figure()
    if energy_df.empty:
        fig.add_annotation(text="No energy history recorded yet", showarrow=False)
        return fig

    room_colors = {
        "Room 1": "#E74C3C",
        "Room 2": "#3498DB",
        "Room 3": "#2ECC71",
        "Room 4": "#9B59B6",
        "Room 5": "#F39C12",
    }

    for r_id in sorted(energy_df["room_id"].unique()):
        rdf = energy_df[energy_df["room_id"] == r_id].copy()
        if not rdf.empty:
            color = room_colors.get(r_id, "#1ABC9C")
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(rdf["timestamp"]),
                    y=rdf["cumulative_total_kwh"],
                    mode="lines",
                    name=r_id,
                    line=dict(color=color, width=2),
                )
            )

    fig.update_layout(
        title=dict(text="Cumulative Energy Consumption by Room (kWh)", font=dict(size=14)),
        xaxis=dict(title="Simulation Timestamp (UTC)", showgrid=True),
        yaxis=dict(title="Cumulative Energy (kWh)", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_occupancy_vs_power_chart(energy_df: pd.DataFrame, dark_mode: bool = True) -> go.Figure:
    """
    Renders scatter plot of room occupancy vs total electrical power to illustrate control modulation.
    """
    fig = go.Figure()
    if energy_df.empty:
        fig.add_annotation(text="No energy data available", showarrow=False)
        return fig

    fig.add_trace(
        go.Scatter(
            x=energy_df["occupant_count"],
            y=energy_df["total_power_w"],
            mode="markers",
            marker=dict(
                size=8,
                color=energy_df["cooling_demand_w"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Cooling Demand (W)", len=0.7),
            ),
            text=energy_df["room_id"] + " (" + energy_df["hvac_mode"] + ")",
            hoverinfo="text+x+y",
            name="Room Observations",
        )
    )

    fig.update_layout(
        title=dict(text="Room Occupancy vs Instantaneous Electrical Power (W)", font=dict(size=14)),
        xaxis=dict(title="Occupant Count", showgrid=True),
        yaxis=dict(title="Total Power (Watts)", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=340,
    )
    return fig


def create_energy_savings_comparison_chart(comparison: dict[str, Any], dark_mode: bool = True) -> go.Figure:
    """
    Renders bar chart comparing Smart Occupancy-Responsive Control vs Baseline Policy.
    """
    fig = go.Figure()
    smart_kwh = comparison.get("smart_total_kwh", 0.0)
    base_kwh = comparison.get("baseline_total_kwh", 0.0)
    savings_pct = comparison.get("savings_percent", 0.0)

    fig.add_trace(
        go.Bar(
            x=["Baseline Policy (Fixed)", "Smart Occupancy-Responsive"],
            y=[base_kwh, smart_kwh],
            marker_color=["#E74C3C", "#2ECC71"],
            text=[f"{base_kwh:.2f} kWh", f"{smart_kwh:.2f} kWh (-{savings_pct:.1f}%)"],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=f"Total Building Energy: Smart Control vs Baseline (Savings: {savings_pct:.1f}%)", font=dict(size=14)),
        yaxis=dict(title="Total Energy (kWh)", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
    )
    return fig


# -----------------------------------------------------------------------------
# PHASE 9: ADVANCED HVAC DIGITAL TWIN ANALYTICS CHARTS
# -----------------------------------------------------------------------------

def create_temperature_vs_setpoint_chart(
    energy_df: pd.DataFrame,
    room_id: Optional[str] = None,
    sensor_df: Optional[pd.DataFrame] = None,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders time-series of indoor temperature vs target setpoint with comfort deadband.
    """
    fig = go.Figure()
    if energy_df is None or energy_df.empty:
        fig.add_annotation(text="Insufficient telemetry data for temperature vs setpoint chart", showarrow=False)
        return fig

    edf = energy_df.copy()
    target_room = room_id or "Room 1"

    if target_room and target_room != "All Rooms (Total Building)":
        merged = edf[edf["room_id"] == target_room].sort_values("timestamp")
        chart_title = f"🌡️ Temperature vs Comfort Setpoint Tracking ({target_room})"
    else:
        merged = edf.groupby("timestamp").agg({
            "indoor_temperature": "mean",
            "hvac_target_setpoint": "mean",
        }).reset_index().sort_values("timestamp")
        chart_title = "🌡️ Facility Mean Indoor Temperature vs Average Comfort Setpoint"

    if merged.empty:
        fig.add_annotation(text=f"No telemetry available for {target_room}", showarrow=False)
        return fig

    ts = pd.to_datetime(merged["timestamp"])
    t_in = merged["indoor_temperature"]
    t_sp = merged["hvac_target_setpoint"]

    # Target Setpoint Line
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=t_sp,
            mode="lines",
            name="Target Setpoint (°C)",
            line=dict(color="#F59E0B", width=2.5, dash="dash"),
        )
    )

    # Upper and Lower Deadband Shading (+- 0.5 C)
    upper_db = t_sp + 0.5
    lower_db = t_sp - 0.5

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=upper_db,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=lower_db,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(245, 158, 11, 0.15)",
            name="Comfort Deadband (±0.5°C)",
        )
    )

    # Actual Indoor Temperature Line
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=t_in,
            mode="lines+markers",
            name=f"{target_room} Indoor Temp (°C)",
            line=dict(color="#00F0FF", width=3),
            marker=dict(size=5, color="#00F0FF"),
        )
    )

    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(size=14),
        ),
        xaxis=dict(title="Timestamp (UTC)", showgrid=True),
        yaxis=dict(title="Temperature (°C)", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_hvac_power_and_energy_chart(
    energy_df: pd.DataFrame,
    room_id: Optional[str] = None,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders dual-axis chart: HVAC electrical power (W) on primary axis,
    cumulative energy (kWh) on secondary axis.
    """
    from plotly.subplots import make_subplots

    if energy_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No energy history recorded yet", showarrow=False)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if room_id and room_id != "All Rooms (Total Building)":
        df_target = energy_df[energy_df["room_id"] == room_id].sort_values("timestamp")
        title_text = f"⚡ HVAC Power Demand & Cumulative Energy ({room_id})"
        ts = pd.to_datetime(df_target["timestamp"])
        power_vals = df_target["hvac_power_w"]
        energy_vals = df_target["cumulative_hvac_kwh"]
    else:
        # Aggregated building-wide
        df_grp = energy_df.groupby("timestamp").agg({
            "hvac_power_w": "sum",
            "cumulative_hvac_kwh": "sum",
        }).reset_index().sort_values("timestamp")
        title_text = "⚡ Aggregate Facility HVAC Power Demand & Cumulative Energy"
        ts = pd.to_datetime(df_grp["timestamp"])
        power_vals = df_grp["hvac_power_w"]
        energy_vals = df_grp["cumulative_hvac_kwh"]

    # Primary: Instantaneous Power
    fig.add_trace(
        go.Bar(
            x=ts,
            y=power_vals,
            name="HVAC Electrical Power (W)",
            marker_color="#0284C7",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # Secondary: Cumulative Energy
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=energy_vals,
            mode="lines+markers",
            name="Cumulative HVAC Energy (kWh)",
            line=dict(color="#10B981", width=3),
            marker=dict(size=4, color="#10B981"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis=dict(title="Timestamp (UTC)", showgrid=True),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Electrical Power (Watts)", showgrid=True, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Energy (kWh)", showgrid=False, secondary_y=True)
    return fig


def create_policy_experiment_chart(
    experiment_results: dict[str, Any],
    dark_mode: bool = True,
) -> go.Figure:
    """
    Renders comparative dual-axis bar chart of Energy Consumption vs Comfort Deviation (MAE).
    """
    from plotly.subplots import make_subplots

    summary_df = experiment_results.get("summary_table", pd.DataFrame())
    if summary_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No experiment data available", showarrow=False)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    policies = summary_df["Policy Code"].tolist()
    kwh = summary_df["Total HVAC Energy (kWh)"].tolist()
    mae = summary_df["Mean Temp Error (°C)"].tolist()

    # Bar: Total Energy (kWh)
    fig.add_trace(
        go.Bar(
            x=policies,
            y=kwh,
            name="Total HVAC Energy (kWh)",
            marker_color=["#0284C7", "#10B981", "#64748B"],
            text=[f"{v:.2f} kWh" for v in kwh],
            textposition="auto",
        ),
        secondary_y=False,
    )

    # Scatter: Mean Temperature Deviation (MAE in °C)
    fig.add_trace(
        go.Scatter(
            x=policies,
            y=mae,
            mode="lines+markers+text",
            name="Mean Temp Deviation (°C)",
            line=dict(color="#EF4444", width=3),
            marker=dict(size=12, color="#EF4444", symbol="diamond"),
            text=[f"{v:.2f} °C" for v in mae],
            textposition="top center",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text="🔬 Policy Trade-Off: Cumulative Energy (kWh) vs Thermal Comfort Deviation (°C)",
            font=dict(size=14),
        ),
        xaxis=dict(title="HVAC Control Policy"),
        template="plotly_dark" if dark_mode else "plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Total HVAC Energy (kWh)", showgrid=True, secondary_y=False)
    fig.update_yaxes(title_text="Mean Temp Deviation (°C)", showgrid=False, secondary_y=True)
    return fig


