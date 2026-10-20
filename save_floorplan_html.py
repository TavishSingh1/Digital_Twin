from simulation.engine import SimulationEngine
from dashboard.svg_floorplan import generate_svg_floorplan_markup

eng = SimulationEngine(num_occupants=35, seed=42)
eng.step()
svg = generate_svg_floorplan_markup(
    building=eng.building,
    occupants=eng.occupants,
    selected_room_id='Room 1',
    active_overlay='Occupancy',
)
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ background: #0B1120; margin: 0; padding: 20px; }}
</style>
</head>
<body>
<div style="width: 800px; height: 600px;">
{svg}
</div>
</body>
</html>"""

with open('screenshots/floorplan.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Saved screenshots/floorplan.html successfully!')

