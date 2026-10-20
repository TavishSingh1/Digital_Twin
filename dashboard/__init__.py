"""
Digital Twin Dashboard Package
==============================
Modular visualization, interactive dashboard components, and BEMS operational workspaces.
"""

from dashboard.floorplan import (
    create_floorplan_figure,
    get_utilization_color,
    get_utilization_border_color,
    validate_building_geometry,
    OVERLAY_OPTIONS,
    get_overlay_legend_info,
)
from dashboard.charts import (
    create_occupancy_chart,
    create_sensor_charts,
    create_error_chart,
    create_metrics_comparison_chart,
    create_room_power_breakdown_chart,
    create_cumulative_energy_trend_chart,
    create_occupancy_vs_power_chart,
    create_energy_savings_comparison_chart,
    FEATURE_UNITS,
)
from dashboard.scenarios import SCENARIO_PRESETS, apply_scenario

__all__ = [
    "create_floorplan_figure",
    "get_utilization_color",
    "get_utilization_border_color",
    "validate_building_geometry",
    "OVERLAY_OPTIONS",
    "get_overlay_legend_info",
    "create_occupancy_chart",
    "create_sensor_charts",
    "create_error_chart",
    "create_metrics_comparison_chart",
    "create_room_power_breakdown_chart",
    "create_cumulative_energy_trend_chart",
    "create_occupancy_vs_power_chart",
    "create_energy_savings_comparison_chart",
    "FEATURE_UNITS",
    "SCENARIO_PRESETS",
    "apply_scenario",
]
