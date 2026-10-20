"""
Live 60 FPS Digital Twin Canvas & Interactive Vector Visualizer
===============================================================
Provides client-side, 60 FPS continuous visual animation for the 5-room facility:
1. Genuine continuous movement interpolation along doorway-constrained waypoints using requestAnimationFrame.
2. In-browser visual playback controls (Play/Pause, Replay, 0.5x-2.0x Speed, Scrubber slider).
3. 1:1 isometric building aspect ratio (26m x 20m) with high-contrast industrial BEMS styling.
4. Deterministic stationary placement, doorway openings, exterior exit, and entrance queue.
5. Multi-physics room overlay coloring, status alerts, and interactive hover tooltips.
6. Strict simulation decoupling: zero server mutations, zero clock advancement, zero energy accumulation.
"""

import json
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components

from simulation.building import Building
from simulation.occupant import Occupant
from dashboard.animation import AnimationManager
from dashboard.floorplan import get_overlay_color_and_text


def generate_live_digital_twin_html(
    building: Building,
    occupants: dict[str, Occupant],
    animation_manager: Optional[AnimationManager] = None,
    anim_mgr: Optional[AnimationManager] = None,
    selected_room_id: Optional[str] = None,
    active_overlay: str = "Occupancy",
    energy_states: Optional[dict[str, Any]] = None,
    sensor_readings: Optional[dict[str, Any]] = None,
    entrance_queue_count: int = 0,
    height: int = 500,
) -> str:
    """
    Constructs a self-contained, high-performance HTML5/Canvas visualization component
    with continuous client-side 60 FPS interpolation along doorway waypoints.
    """
    am = animation_manager or anim_mgr or AnimationManager()
    animation_manager = am
    rooms_data = []
    for r_id, rm in building.rooms.items():
        fill_col, border_col, metric_str, badge_str, tip_str = get_overlay_color_and_text(
            room_id=r_id,
            room=rm,
            overlay=active_overlay,
            energy_states=energy_states,
            sensor_readings=sensor_readings,
        )
        is_sel = (selected_room_id is not None) and (
            r_id == selected_room_id or r_id.replace(" ", "_") == selected_room_id.replace(" ", "_")
        )
        s_dict = (sensor_readings.get(r_id) or {}) if sensor_readings else {}
        e_st = (energy_states.get(r_id) or None) if energy_states else None

        rooms_data.append({
            "id": r_id,
            "name": rm.name,
            "x": rm.x,
            "y": rm.y,
            "width": rm.width,
            "height": rm.height,
            "capacity": rm.capacity,
            "occupant_count": rm.occupant_count,
            "fill": fill_col,
            "border": border_col,
            "isSelected": is_sel,
            "metric": metric_str,
            "badge": badge_str,
            "temp": round(s_dict.get("indoor_temperature", 24.5), 1),
            "co2": round(s_dict.get("indoor_co2", 450.0)),
            "lux": round(s_dict.get("illuminance", 300.0)),
            "power": round(getattr(e_st, "total_power_w", 0.0)),
            "hvacMode": getattr(e_st, "hvac_mode", "AUTO") if e_st else "AUTO",
        })

    # 2. Package door data
    doors_data = []
    for (r1, r2), pos in building.door_positions.items():
        # Avoid duplicate bidirectional keys
        if r1 < r2:
            doors_data.append({
                "roomA": r1,
                "roomB": r2,
                "x": pos[0],
                "y": pos[1],
            })

    # 3. Package stationary occupant data
    stationary_positions = animation_manager.get_occupant_positions(building, occupants)
    occupants_data = []
    for pid, occ in occupants.items():
        if occ.is_inside_building and occ.current_room_id:
            pos = stationary_positions.get(pid, occ.position)
            occupants_data.append({
                "id": pid,
                "type": occ.occupant_type,
                "room": occ.current_room_id,
                "activity": occ.current_activity,
                "state": occ.movement_state,
                "x": pos[0],
                "y": pos[1],
                "isSelected": (pid == animation_manager.selected_occupant_id),
            })

    # 4. Package active movement event or active replay event
    active_ev = animation_manager.get_active_replay_event()
    is_replay = True if active_ev else False
    if not active_ev and animation_manager.events:
        active_ev = animation_manager.events[-1]

    event_payload = None
    if active_ev and len(active_ev.waypoints) >= 2:
        event_payload = {
            "id": active_ev.event_id,
            "type": active_ev.event_type,
            "timestamp": active_ev.timestamp,
            "personIds": active_ev.person_ids,
            "sourceRoom": active_ev.source_room or "Outside",
            "destRoom": active_ev.destination_room or "Exit",
            "waypoints": active_ev.waypoints,
            "isReplay": is_replay,
        }

    # 5. Package visual exits and queue
    queue_count = entrance_queue_count
    if queue_count == 0 and occupants:
        queue_count = sum(1 for o in occupants.values() if o.movement_state == "waiting_at_entrance")

    exit_records_data = []
    for rec in animation_manager.exit_records[-15:]:
        exit_records_data.append({
            "id": rec["occupant_id"],
            "fromRoom": rec["from_room"],
            "timestamp": rec["timestamp"],
            "x": rec["position"][0],
            "y": rec["position"][1],
        })

    # Selected occupant route waypoints
    selected_occ_route = []
    if animation_manager.selected_occupant_id and animation_manager.selected_occupant_id in occupants:
        sel_occ = occupants[animation_manager.selected_occupant_id]
        if sel_occ.is_inside_building and sel_occ.intended_destination:
            selected_occ_route = animation_manager.get_selected_route_waypoints(building, sel_occ)

    config_json = json.dumps({
        "rooms": rooms_data,
        "doors": doors_data,
        "occupants": occupants_data,
        "activeEvent": event_payload,
        "selectedRoute": selected_occ_route,
        "exitRecords": exit_records_data,
        "queueCount": queue_count,
        "selectedRoom": selected_room_id,
        "selectedOccupant": animation_manager.selected_occupant_id,
        "showOccupants": animation_manager.show_occupants,
        "showPaths": animation_manager.show_paths,
        "speed": animation_manager.animation_speed,
        "isPaused": animation_manager.is_paused,
        "overlay": active_overlay,
    })

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background: #0F172A; color: #ECEFF1; overflow: hidden; user-select: none; }}
  #dt-container {{ position: relative; width: 100%; height: 490px; background: #0B1120; border-radius: 8px; border: 1px solid #1E293B; }}
  #canvas {{ display: block; width: 100%; height: 100%; cursor: crosshair; }}
  
  /* Floating HUD Controls */
  .hud-bar {{
    position: absolute;
    bottom: 12px;
    left: 14px;
    right: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(15, 23, 42, 0.88);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(51, 65, 85, 0.7);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    z-index: 10;
  }}
  .hud-group {{ display: flex; align-items: center; gap: 8px; }}
  .hud-btn {{
    background: #1E293B;
    border: 1px solid #334155;
    color: #F8FAFC;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .hud-btn:hover {{ background: #334155; border-color: #00F0FF; color: #00F0FF; }}
  .hud-btn.active {{ background: #0284C7; border-color: #38BDF8; color: #FFFFFF; }}
  .hud-badge {{
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .badge-live {{ background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #34D399; }}
  .badge-replay {{ background: rgba(245, 158, 11, 0.25); border: 1px solid #F59E0B; color: #FBBF24; animation: pulse 1.5s infinite; }}
  
  /* Range Slider */
  .hud-slider {{
    -webkit-appearance: none;
    width: 140px;
    height: 4px;
    border-radius: 2px;
    background: #334155;
    outline: none;
  }}
  .hud-slider::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #00F0FF;
    cursor: pointer;
  }}

  /* Hover Tooltip Card */
  #tooltip {{
    position: absolute;
    display: none;
    pointer-events: none;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #38BDF8;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 11px;
    line-height: 1.4;
    color: #F8FAFC;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    z-index: 100;
  }}

  @keyframes pulse {{
    0% {{ opacity: 0.7; }}
    50% {{ opacity: 1.0; }}
    100% {{ opacity: 0.7; }}
  }}
</style>
</head>
<body>
<div id="dt-container">
  <canvas id="canvas"></canvas>
  
  <div class="hud-bar">
    <div class="hud-group">
      <span id="live-badge" class="hud-badge badge-live">● Live Digital Twin (60 FPS)</span>
      <button id="btn-play-pause" class="hud-btn">⏸️ Pause</button>
      <button id="btn-replay" class="hud-btn">🔁 Replay Motion</button>
      <span style="color:#64748B;">|</span>
      <span style="color:#94A3B8; font-size:11px;">Speed:</span>
      <button class="hud-btn spd-btn" data-spd="0.5">0.5x</button>
      <button class="hud-btn spd-btn active" data-spd="1.0">1.0x</button>
      <button class="hud-btn spd-btn" data-spd="1.5">1.5x</button>
      <button class="hud-btn spd-btn" data-spd="2.0">2.0x</button>
    </div>
    
    <div class="hud-group">
      <span style="color:#94A3B8; font-size:11px;">Visual Transit:</span>
      <input id="scrub-slider" type="range" class="hud-slider" min="0" max="100" value="100">
      <span id="scrub-val" style="color:#38BDF8; font-family:monospace; min-width:38px;">100%</span>
    </div>
  </div>

  <div id="tooltip"></div>
</div>

<script>
const config = {config_json};
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const btnPlayPause = document.getElementById("btn-play-pause");
const btnReplay = document.getElementById("btn-replay");
const scrubSlider = document.getElementById("scrub-slider");
const scrubVal = document.getElementById("scrub-val");
const liveBadge = document.getElementById("live-badge");
const spdBtns = document.querySelectorAll(".spd-btn");

let width, height;
let scale = 24;
let offsetX = 45;
let offsetY = 25;
let animProgress = config.activeEvent ? 0.0 : 1.0;
let isPlaying = !config.isPaused;
let animSpeed = config.speed || 1.0;
let lastTimestamp = 0;
let activeEvent = config.activeEvent;
let hoverTarget = null;

if (activeEvent && activeEvent.isReplay) {{
  liveBadge.className = "hud-badge badge-replay";
  liveBadge.textContent = "⏪ Replay: " + activeEvent.id;
}}

// Resize canvas handling devicePixelRatio for high-DPI crisp rendering
function resize() {{
  const rect = canvas.getBoundingClientRect();
  const container = document.getElementById("dt-container");
  const cRect = container ? container.getBoundingClientRect() : null;

  width = rect.width || (cRect ? cRect.width : 0) || window.innerWidth || 800;
  height = rect.height || (cRect ? cRect.height : 0) || 480;

  if (width < 200) width = 800;
  if (height < 200) height = 480;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);

  // 1:1 isometric aspect ratio for 26m x 20m building envelope
  const availW = Math.max(100, width - 80);
  const availH = Math.max(100, height - 70);
  scale = Math.min(availW / 26.0, availH / 20.0);
  if (scale <= 0) scale = 20.0;
  offsetX = (width - 26.0 * scale) / 2.0;
  offsetY = (height - 20.0 * scale) / 2.0 - 10;
}}
window.addEventListener("resize", () => {{
  resize();
  scheduleFrame();
}});
resize();

// Safe numerical clamping to prevent coordinate corruption or NaN crashes
function clamp(val, min, max, fallback) {{
  if (typeof val !== "number" || isNaN(val) || !isFinite(val)) return fallback;
  return Math.max(min, Math.min(max, val));
}}

// Coordinate transformation: Simulation (x, y) meters -> Canvas (cx, cy) pixels
function toCanvas(x, y) {{
  const safeX = clamp(x, -2.0, 28.0, 0.0);
  const safeY = clamp(y, -3.0, 23.0, 0.0);
  return {{
    cx: offsetX + safeX * scale,
    cy: offsetY + (20.0 - safeY) * scale,
  }};
}}

// Linear interpolation along multi-segment waypoints with safe bounds checking
function interpolateWaypoints(wps, p) {{
  if (!Array.isArray(wps) || wps.length === 0) return {{ x: 7.0, y: 0.0 }};
  if (wps.length === 1) {{
    return {{
      x: clamp(wps[0][0], -2.0, 28.0, 7.0),
      y: clamp(wps[0][1], -3.0, 23.0, 0.0)
    }};
  }}
  p = clamp(p, 0.0, 1.0, 1.0);
  if (p <= 0.0) return {{ x: clamp(wps[0][0], -2.0, 28.0, 7.0), y: clamp(wps[0][1], -3.0, 23.0, 0.0) }};
  if (p >= 1.0) {{
    const last = wps[wps.length - 1];
    return {{ x: clamp(last[0], -2.0, 28.0, 7.0), y: clamp(last[1], -3.0, 23.0, 0.0) }};
  }}

  let segLens = [];
  let totalLen = 0;
  for (let i = 0; i < wps.length - 1; i++) {{
    let x1 = clamp(wps[i][0], -2.0, 28.0, 0.0);
    let y1 = clamp(wps[i][1], -3.0, 23.0, 0.0);
    let x2 = clamp(wps[i+1][0], -2.0, 28.0, 0.0);
    let y2 = clamp(wps[i+1][1], -3.0, 23.0, 0.0);
    let d = Math.hypot(x2 - x1, y2 - y1);
    segLens.push(d);
    totalLen += d;
  }}
  if (totalLen <= 0.0001) {{
    return {{ x: clamp(wps[0][0], -2.0, 28.0, 7.0), y: clamp(wps[0][1], -3.0, 23.0, 0.0) }};
  }}

  let targetDist = p * totalLen;
  let acc = 0;
  for (let i = 0; i < segLens.length; i++) {{
    if (acc + segLens[i] >= targetDist || i === segLens.length - 1) {{
      let segP = (targetDist - acc) / Math.max(0.0001, segLens[i]);
      segP = Math.max(0, Math.min(1, segP));
      return {{
        x: wps[i][0] + segP * (wps[i+1][0] - wps[i][0]),
        y: wps[i][1] + segP * (wps[i+1][1] - wps[i][1]),
      }};
    }}
    acc += segLens[i];
  }}
  return {{ x: wps[wps.length - 1][0], y: wps[wps.length - 1][1] }};
}}

// Main Render Loop (60 FPS)
function draw(timestamp) {{
  if (!lastTimestamp) lastTimestamp = timestamp;
  const dt = (timestamp - lastTimestamp) / 1000.0;
  lastTimestamp = timestamp;

  if (isPlaying && animProgress < 1.0) {{
    // Base travel duration: ~2.2 seconds normalized by speed
    animProgress = Math.min(1.0, animProgress + (dt * animSpeed) / 2.2);
    scrubSlider.value = Math.round(animProgress * 100);
    scrubVal.textContent = Math.round(animProgress * 100) + "%";
  }}

  // 1. Clear Canvas & Background Grid
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0B1120";
  ctx.fillRect(0, 0, width, height);

  // Grid lines
  ctx.strokeStyle = "#1E293B";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= 26; x += 2) {{
    let p1 = toCanvas(x, 0);
    let p2 = toCanvas(x, 20);
    ctx.moveTo(p1.cx, p1.cy);
    ctx.lineTo(p2.cx, p2.cy);
  }}
  for (let y = 0; y <= 20; y += 2) {{
    let p1 = toCanvas(0, y);
    let p2 = toCanvas(26, y);
    ctx.moveTo(p1.cx, p1.cy);
    ctx.lineTo(p2.cx, p2.cy);
  }}
  ctx.stroke();

  // 2. Draw Rooms (Simplified, uncluttered annotations)
  config.rooms.forEach(rm => {{
    const tl = toCanvas(rm.x, rm.y + rm.height);
    const rw = rm.width * scale;
    const rh = rm.height * scale;

    ctx.fillStyle = rm.fill;
    ctx.fillRect(tl.cx, tl.cy, rw, rh);

    if (rm.isSelected) {{
      ctx.strokeStyle = "#00F0FF";
      ctx.lineWidth = 4;
      ctx.strokeRect(tl.cx, tl.cy, rw, rh);

      ctx.save();
      ctx.strokeStyle = "rgba(0, 240, 255, 0.35)";
      ctx.lineWidth = 8;
      ctx.strokeRect(tl.cx - 2, tl.cy - 2, rw + 4, rh + 4);
      ctx.restore();
    }} else {{
      ctx.strokeStyle = rm.border;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(tl.cx, tl.cy, rw, rh);
    }}

    // Room Label & Occupant Count (Simplified)
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(rm.id + (rm.isSelected ? " ★" : ""), tl.cx + rw / 2, tl.cy + rh / 2 - 10);

    ctx.fillStyle = "#94A3B8";
    ctx.font = "10px sans-serif";
    ctx.fillText(rm.name, tl.cx + rw / 2, tl.cy + rh / 2 + 5);

    ctx.fillStyle = "#38BDF8";
    ctx.font = "bold 11px monospace";
    ctx.fillText("👥 " + rm.occupant_count + " / " + rm.capacity, tl.cx + rw / 2, tl.cy + rh / 2 + 22);
  }});

  // 3. Draw Interior Doors
  config.doors.forEach(d => {{
    const p = toCanvas(d.x, d.y);
    ctx.fillStyle = "#F1C40F";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 1.5;
    ctx.fillRect(p.cx - 6, p.cy - 6, 12, 12);
    ctx.strokeRect(p.cx - 6, p.cy - 6, 12, 12);
  }});

  // 4. Draw Exterior Main Exit at Room 1 (7.0, 0.0)
  const exitPos = toCanvas(7.0, 0.0);
  ctx.fillStyle = "#EF4444";
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(exitPos.cx, exitPos.cy + 10);
  ctx.lineTo(exitPos.cx - 8, exitPos.cy - 4);
  ctx.lineTo(exitPos.cx + 8, exitPos.cy - 4);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#EF4444";
  ctx.font = "bold 9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("MAIN EXIT", exitPos.cx, exitPos.cy + 22);

  // 5. Draw Entrance Queue Indicator at (7.0, -1.6)
  if (config.queueCount > 0) {{
    const qPos = toCanvas(7.0, -1.6);
    ctx.fillStyle = "#E67E22";
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(qPos.cx, qPos.cy, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#E67E22";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("QUEUE: " + config.queueCount, qPos.cx, qPos.cy + 16);
  }}

  // 6. Draw Visual Exits (outside building)
  config.exitRecords.forEach((rec, i) => {{
    const offX = ((i % 5) - 2) * 0.4;
    const offY = Math.floor(i / 5) * -0.4;
    const p = toCanvas(rec.x + offX, rec.y + offY);
    ctx.strokeStyle = "#E74C3C";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p.cx - 4, p.cy - 4);
    ctx.lineTo(p.cx + 4, p.cy + 4);
    ctx.moveTo(p.cx + 4, p.cy - 4);
    ctx.lineTo(p.cx - 4, p.cy + 4);
    ctx.stroke();
  }});

  // 7. Draw Movement Trajectory Paths
  if (config.showPaths && activeEvent && activeEvent.waypoints.length >= 2) {{
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = activeEvent.type === "admission" ? "#2ECC71" : (activeEvent.type === "exit" ? "#E74C3C" : "#3498DB");
    ctx.beginPath();
    activeEvent.waypoints.forEach((wp, idx) => {{
      const p = toCanvas(wp[0], wp[1]);
      if (idx === 0) ctx.moveTo(p.cx, p.cy);
      else ctx.lineTo(p.cx, p.cy);
    }});
    ctx.stroke();

    // Intermediate door waypoint dots
    ctx.fillStyle = ctx.strokeStyle;
    activeEvent.waypoints.forEach(wp => {{
      const p = toCanvas(wp[0], wp[1]);
      ctx.beginPath();
      ctx.arc(p.cx, p.cy, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }});
    ctx.restore();
  }}

  // 8. Draw Selected Occupant Route (if active)
  if (config.selectedRoute && config.selectedRoute.length >= 2) {{
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#F1C40F";
    ctx.beginPath();
    config.selectedRoute.forEach((wp, idx) => {{
      const p = toCanvas(wp[0], wp[1]);
      if (idx === 0) ctx.moveTo(p.cx, p.cy);
      else ctx.lineTo(p.cx, p.cy);
    }});
    ctx.stroke();
    ctx.restore();
  }}

  // 9. Draw Occupants & Interpolate Moving Markers
  if (config.showOccupants) {{
    const movingPids = (activeEvent && animProgress < 1.0) ? new Set(activeEvent.personIds) : new Set();
    const interpCoords = (activeEvent && animProgress < 1.0) ? interpolateWaypoints(activeEvent.waypoints, animProgress) : null;

    config.occupants.forEach(occ => {{
      let occX = clamp(occ.x, 0.0, 26.0, 7.0);
      let occY = clamp(occ.y, 0.0, 20.0, 3.5);
      let pos;
      let isMoving = movingPids.has(occ.id);
      if (isMoving && interpCoords) {{
        pos = toCanvas(interpCoords.x, interpCoords.y);
      }} else {{
        pos = toCanvas(occX, occY);
      }}

      // Marker colors: regular=blue, visitor=purple, staff=teal
      let fillCol = occ.type === "visitor" ? "#9B59B6" : (occ.type === "staff" ? "#1ABC9C" : "#3498DB");

      // Draw outer halo if selected
      if (occ.isSelected) {{
        ctx.fillStyle = "rgba(241, 196, 15, 0.35)";
        ctx.beginPath();
        ctx.arc(pos.cx, pos.cy, 12, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = "#F1C40F";
        ctx.lineWidth = 2.5;
        ctx.stroke();

        ctx.fillStyle = "#F1C40F";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("★ " + occ.id, pos.cx, pos.cy - 12);
      }}

      ctx.fillStyle = fillCol;
      ctx.strokeStyle = isMoving ? "#00F0FF" : "#FFFFFF";
      ctx.lineWidth = isMoving ? 2 : 1;
      ctx.beginPath();
      ctx.arc(pos.cx, pos.cy, isMoving ? 6.5 : 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }});
  }}

  // Only continue loop if actively animating and not tab-hidden
  if (isPlaying && animProgress < 1.0 && !document.hidden) {{
    scheduleFrame();
  }}
}}

let animFrameId = null;
function scheduleFrame() {{
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = requestAnimationFrame(draw);
}}

// Draw single frame initially
draw(performance.now());

document.addEventListener("visibilitychange", () => {{
  if (document.hidden) {{
    if (animFrameId) cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }} else if (isPlaying && animProgress < 1.0) {{
    lastTimestamp = null;
    scheduleFrame();
  }}
}});

window.addEventListener("beforeunload", () => {{
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = null;
}});

if (window.ResizeObserver) {{
  const ro = new ResizeObserver(() => {{
    resize();
    draw(performance.now());
  }});
  const cEl = document.getElementById("dt-container");
  if (cEl) ro.observe(cEl);
}}

// UI Event Handlers
btnPlayPause.addEventListener("click", () => {{
  isPlaying = !isPlaying;
  btnPlayPause.textContent = isPlaying ? "⏸️ Pause" : "▶️ Play";
  if (isPlaying && animProgress < 1.0) {{
    lastTimestamp = null;
    scheduleFrame();
  }}
}});

btnReplay.addEventListener("click", () => {{
  animProgress = 0.0;
  isPlaying = true;
  btnPlayPause.textContent = "⏸️ Pause";
  lastTimestamp = null;
  scheduleFrame();
}});

spdBtns.forEach(btn => {{
  btn.addEventListener("click", () => {{
    spdBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    animSpeed = parseFloat(btn.dataset.spd);
    scheduleFrame();
  }});
}});

scrubSlider.addEventListener("input", (e) => {{
  isPlaying = false;
  btnPlayPause.textContent = "▶️ Play";
  animProgress = e.target.value / 100.0;
  scrubVal.textContent = e.target.value + "%";
  scheduleFrame();
}});

// Canvas Mouse Hover Tooltip
canvas.addEventListener("mousemove", (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  let found = null;

  // Check occupants first
  for (let occ of config.occupants) {{
    const p = toCanvas(occ.x, occ.y);
    if (Math.hypot(p.cx - mx, p.cy - my) < 9) {{
      found = {{
        title: "Occupant: " + occ.id,
        lines: [
          "Type: <b>" + occ.type.toUpperCase() + "</b>",
          "Current Room: <b>" + occ.room + "</b>",
          "Activity: " + occ.activity,
          "State: " + occ.state,
        ]
      }};
      break;
    }}
  }}

  // Check rooms
  if (!found) {{
    for (let rm of config.rooms) {{
      const tl = toCanvas(rm.x, rm.y + rm.height);
      const rw = rm.width * scale;
      const rh = rm.height * scale;
      if (mx >= tl.cx && mx <= tl.cx + rw && my >= tl.cy && my <= tl.cy + rh) {{
        found = {{
          title: rm.id + ": " + rm.name,
          lines: [
            "Occupancy: <b>" + rm.occupant_count + " / " + rm.capacity + "</b> (" + Math.round(rm.occupant_count / rm.capacity * 100) + "%)",
            "Temp: <b>" + rm.temp + " °C</b> | CO₂: <b>" + rm.co2 + " ppm</b>",
            "Illuminance: <b>" + rm.lux + " lux</b>",
            "HVAC: <b>" + rm.hvacMode + "</b> | Power: <b>" + rm.power + " W</b>",
            "Status: " + rm.badge,
          ]
        }};
        break;
      }}
    }}
  }}

  if (found) {{
    tooltip.style.display = "block";
    tooltip.style.left = (mx + 15) + "px";
    tooltip.style.top = (my - 10) + "px";
    tooltip.innerHTML = "<b style='color:#38BDF8; font-size:12px;'>" + found.title + "</b><br>" + found.lines.join("<br>");
  }} else {{
    tooltip.style.display = "none";
  }}
}});

canvas.addEventListener("mouseleave", () => {{
  tooltip.style.display = "none";
}});
</script>
</body>
</html>
"""
    return html


def render_live_floorplan_component(
    building: Building,
    occupants: dict[str, Occupant],
    animation_manager: Optional[AnimationManager] = None,
    anim_mgr: Optional[AnimationManager] = None,
    selected_room_id: Optional[str] = None,
    active_overlay: str = "Occupancy",
    energy_states: Optional[dict[str, Any]] = None,
    sensor_readings: Optional[dict[str, Any]] = None,
    entrance_queue_count: int = 0,
    height: int = 500,
):
    """
    Renders the live, client-side 60 FPS Digital Twin canvas component into Streamlit.
    """
    am = animation_manager or anim_mgr or AnimationManager()
    html_content = generate_live_digital_twin_html(
        building=building,
        occupants=occupants,
        animation_manager=am,
        selected_room_id=selected_room_id,
        active_overlay=active_overlay,
        energy_states=energy_states,
        sensor_readings=sensor_readings,
        entrance_queue_count=entrance_queue_count,
        height=height,
    )
    components.html(html_content, height=height, scrolling=False)
