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


def create_metrics_comparison_chart(metrics: dict[str, Any], dark_mode: bool = True) -> go.Figure:
    """
    Renders bar chart comparing STGNN Integer MAE vs Persistence MAE across all rooms.
    """
    fig = go.Figure()
    room_metrics = metrics.get("room_metrics", {})
    if not room_metrics:
        fig.add_annotation(text="No evaluated samples available for metrics comparison", showarrow=False)
        return fig

    rooms = list(room_metrics.keys())
    stgnn_maes = [room_metrics[r]["stgnn_int_mae"] for r in rooms]
    pers_maes = [room_metrics[r]["persistence_mae"] for r in rooms]

    fig.add_trace(
        go.Bar(
            x=rooms,
            y=stgnn_maes,
            name="STGNN Integer MAE",
            marker_color="#3498DB",
        )
    )

    fig.add_trace(
        go.Bar(
            x=rooms,
            y=pers_maes,
            name="Persistence Baseline MAE",
            marker_color="#F39C12",
        )
    )

    fig.update_layout(
        title=dict(text="Mean Absolute Error (MAE) Comparison by Room", font=dict(size=14)),
        xaxis=dict(title="Room"),
        yaxis=dict(title="MAE (Occupants)", showgrid=True),
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


