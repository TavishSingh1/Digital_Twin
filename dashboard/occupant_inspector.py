"""
Occupant Inspector & Spatial Diagnostics Panel
=============================================
Provides deep contextual inspection of individual occupants:
1. Distinguishes authoritative room membership from visual transit progress.
2. Route waypoints, source/destination, behavioral activities, and queue status.
3. Interactive search by ID, room filtering, and movement status filtering.
4. Graceful handling of stale selections when occupants depart.
5. Strict simulation decoupling: zero mutations to logical clock or occupancy state.
"""

from typing import Any, Optional
import streamlit as st
import pandas as pd

from simulation.occupant import Occupant
from simulation.config import EXPECTED_ROOM_IDS
from dashboard.animation import AnimationManager


def render_occupant_inspector(
    pipeline: Any,
    anim_mgr: AnimationManager,
    selected_room: str,
):
    """
    Renders the unified Occupant Inspector & Spatial Diagnostics panel.
    """
    occupants = pipeline.engine.occupants
    entrance_queue = pipeline.engine.entrance_queue

    st.markdown("##### 👤 Occupant Spatial Inspector & Diagnostics")

    # Validate and clean up any stale selection
    selected_id = anim_mgr.validate_selected_occupant(occupants)

    # 1. Filter and Search Controls Row
    f_c1, f_c2, f_c3, f_c4 = st.columns([1.2, 1.2, 1.2, 0.8])
    with f_c1:
        search_query = st.text_input(
            "🔍 Search Occupant ID",
            value="",
            placeholder="e.g. P0001",
            key="tb_occ_search",
            help="Filter occupants by partial or full ID match."
        )

    with f_c2:
        room_options = ["All Rooms"] + EXPECTED_ROOM_IDS + ["Entrance Queue"]
        def_r_idx = room_options.index(selected_room) if selected_room in room_options else 0
        filter_room = st.selectbox(
            "Filter by Room",
            options=room_options,
            index=def_r_idx,
            key="sb_occ_filter_room",
        )

    with f_c3:
        status_options = ["All Statuses", "Stationary", "Moving", "In Queue"]
        filter_status = st.selectbox(
            "Movement Status",
            options=status_options,
            index=0,
            key="sb_occ_filter_status",
        )

    with f_c4:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Clear Selection", key="btn_clear_occ_sel", use_container_width=True):
            anim_mgr.selected_occupant_id = None
            if "sb_selected_occupant" in st.session_state:
                del st.session_state["sb_selected_occupant"]
            if "sb_inspector_pick_occupant" in st.session_state:
                del st.session_state["sb_inspector_pick_occupant"]
            st.rerun()

    # 2. Filter Candidate Occupants
    candidates = []
    # Add inside building occupants
    for pid, occ in sorted(occupants.items()):
        if occ.is_inside_building and occ.current_room_id:
            # Check room filter
            if filter_room != "All Rooms" and filter_room != "Entrance Queue" and occ.current_room_id != filter_room:
                continue
            if filter_room == "Entrance Queue":
                continue

            # Check status filter
            st_label = "Moving" if occ.movement_state == "moving" else "Stationary"
            if filter_status == "Moving" and st_label != "Moving":
                continue
            if filter_status == "Stationary" and st_label != "Stationary":
                continue
            if filter_status == "In Queue":
                continue

            # Check search
            if search_query and search_query.strip().upper() not in pid.upper():
                continue

            candidates.append((pid, occ, False))

    # Add entrance queue occupants
    if filter_room in ["All Rooms", "Entrance Queue"]:
        for q_occ in entrance_queue:
            if filter_status in ["All Statuses", "In Queue"]:
                if not search_query or search_query.strip().upper() in q_occ.occupant_id.upper():
                    candidates.append((q_occ.occupant_id, q_occ, True))

    # 3. Selected Occupant Detail Card (if active)
    if selected_id and selected_id in occupants:
        target_occ = occupants[selected_id]
        in_queue = (target_occ.movement_state == "waiting_at_entrance")

        # Find latest movement event mentioning this occupant
        recent_ev = None
        for ev in reversed(anim_mgr.events):
            if selected_id in ev.person_ids:
                recent_ev = ev
                break

        # Calculate doorway route description
        waypoints = anim_mgr.get_selected_route_waypoints(pipeline.engine.building, target_occ)
        wp_desc = f"{len(waypoints)} Waypoints: " + " → ".join([f"({w[0]:.1f}, {w[1]:.1f})" for w in waypoints[:4]])
        if len(waypoints) > 4:
            wp_desc += f" ... (+{len(waypoints)-4} more)"

        st.markdown(
            f"<div style='background:#1E293B; border:2px solid #F1C40F; border-radius:8px; padding:12px 16px; margin-bottom:12px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
            f"<div>"
            f"<span style='font-size:1.1rem; font-weight:700; color:#F1C40F;'>★ Inspecting Occupant: {selected_id}</span>"
            f"<span style='background:#334155; color:#94A3B8; padding:2px 8px; border-radius:4px; margin-left:8px; font-size:0.75rem; font-weight:600; text-transform:uppercase;'>"
            f"{target_occ.occupant_type}"
            f"</span>"
            f"</div>"
            f"<span style='font-size:0.8rem; color:#38BDF8; font-family:monospace;'>"
            f"{'⏳ Waiting in Entrance Queue' if in_queue else '🟢 Inside Facility'}</span>"
            f"</div>"
            f"<div style='display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size:0.8rem;'>"
            f"<div><span style='color:#94A3B8;'>Authoritative Room:</span><br><b>{target_occ.current_room_id or 'Outside / Queue'}</b></div>"
            f"<div><span style='color:#94A3B8;'>Visual Status:</span><br><b>{target_occ.movement_state.capitalize()}</b></div>"
            f"<div><span style='color:#94A3B8;'>Activity:</span><br><b>{target_occ.current_activity.capitalize()}</b></div>"
            f"<div><span style='color:#94A3B8;'>Intended Target:</span><br><b>{target_occ.intended_destination or 'None (Stationary)'}</b></div>"
            f"</div>"
            f"<div style='margin-top:8px; font-size:0.75rem; color:#CBD5E1; border-top:1px solid #334155; padding-top:6px;'>"
            f"<div><b>Planned Doorway Trajectory:</b> <code>{wp_desc if waypoints else 'Stationary — No active route'}</code></div>"
            f"{f'<div><b>Recent Event:</b> {recent_ev.event_type.upper()} ({recent_ev.source_room} → {recent_ev.destination_room}) at {recent_ev.timestamp}</div>' if recent_ev else ''}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # 4. Candidate Occupants Table & Quick Selector
    if candidates:
        candidate_ids = [c[0] for c in candidates]
        inspector_options = ["None"] + candidate_ids

        # If previous widget value in session_state is invalid, clear it
        if "sb_inspector_pick_occupant" in st.session_state:
            if st.session_state["sb_inspector_pick_occupant"] not in inspector_options:
                del st.session_state["sb_inspector_pick_occupant"]

        def on_inspector_occ_change():
            val = st.session_state.get("sb_inspector_pick_occupant", "None")
            anim_mgr.selected_occupant_id = None if val == "None" else val
            if "sb_selected_occupant" in st.session_state:
                del st.session_state["sb_selected_occupant"]

        cur_sel = anim_mgr.selected_occupant_id or "None"
        sel_idx = inspector_options.index(cur_sel) if cur_sel in inspector_options else 0
        chosen_pid = st.selectbox(
            f"Select Occupant to Inspect ({len(candidates)} matching filters)",
            options=inspector_options,
            index=sel_idx,
            key="sb_inspector_pick_occupant",
            on_change=on_inspector_occ_change,
            help="Select an occupant to view spatial route, activity, and highlight on the floor plan."
        )

        # Compact summary table of filtered occupants
        with st.expander(f"📋 Candidate Occupants List ({len(candidates)} individuals)", expanded=False):
            table_rows = []
            for pid, occ, is_q in candidates[:30]:
                table_rows.append({
                    "ID": pid,
                    "Type": occ.occupant_type.capitalize(),
                    "Room": occ.current_room_id or ("Queue" if is_q else "Outside"),
                    "Activity": occ.current_activity,
                    "State": occ.movement_state.capitalize(),
                    "Target": occ.intended_destination or "—",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
            if len(candidates) > 30:
                st.caption(f"Showing first 30 of {len(candidates)} matching occupants. Refine search or room filters to narrow down.")
    else:
        st.info("No occupants match the selected room, status, or search query.")

