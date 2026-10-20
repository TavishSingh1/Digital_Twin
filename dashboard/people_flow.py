"""
People Flow Control Workspace View
==================================
Intuitive building occupant dispatch and transition workspace:
- Section A: Add people with real-time capacity and entrance queue preview
- Section B: Move people between rooms with topological door validation and capacity checks
- Section C: Exit people from specific rooms or building-wide with preview
- Section D: Diurnal schedule viewer inspecting occupant circadian goals
"""

from typing import Any, Optional
import streamlit as st
import pandas as pd

from simulation.config import EXPECTED_ROOM_IDS
from dashboard.state import (
    get_selected_room,
    set_selected_room,
    add_person_action,
    exit_person_action,
    move_people_action,
)


def render_people_flow_view(pipeline: Any):
    """Renders the comprehensive People Flow Control workspace."""
    if not pipeline:
        st.warning("⚠️ Simulation pipeline is not initialized.")
        return

    st.subheader("🚶 Building People Flow & Dispatch Operations")
    st.caption("Coordinate and command occupant transitions across the five rooms with real-time topological door routing.")

    # 1. Live Facility Occupancy Matrix Bar
    occ_cols = st.columns(6)
    for idx, r_id in enumerate(EXPECTED_ROOM_IDS):
        rm = pipeline.engine.building.get_room(r_id)
        occ_cols[idx].metric(
            label=f"{r_id}",
            value=f"{rm.occupant_count} / {rm.capacity}",
            delta=f"{rm.capacity - rm.occupant_count} free",
        )
    occ_cols[5].metric(
        label="Entrance Queue",
        value=pipeline.engine.waiting_at_entrance,
        help="Individuals waiting at exterior entrance due to Room 1 capacity limit",
    )

    st.markdown("---")

    # 2. Add / Move / Exit in 3 Structured Operational Columns
    col_add, col_move, col_exit = st.columns([1, 1.4, 1])

    # -------------------------------------------------------------------------
    # SECTION A: ADD PEOPLE
    # -------------------------------------------------------------------------
    with col_add:
        st.markdown("#### ➕ Add People")
        st.caption("Inject new entrants through Room 1 (Main Entrance).")

        add_count = st.number_input(
            "Number of Entrants",
            min_value=1,
            max_value=200,
            value=5,
            step=1,
            key="flow_add_count",
            help="Any number of individuals can enter. Excess occupants queue at entrance FIFO.",
        )

        dest_choice = st.selectbox(
            "Target Destination",
            options=["Free Movement (Unassigned)"] + EXPECTED_ROOM_IDS,
            index=0,
            key="flow_add_dest_choice",
        )
        target_dest = None if dest_choice == "Free Movement (Unassigned)" else dest_choice

        # Preview calculation
        r1 = pipeline.engine.building.get_room("Room 1")
        r1_free = max(0, r1.capacity - r1.occupant_count)
        admit_preview = min(add_count, r1_free)
        queue_preview = max(0, add_count - r1_free)

        st.markdown(
            f"<div style='background:#1E293B; border:1px solid #334155; border-radius:6px; padding:10px; margin-bottom:12px; font-size:0.85rem;'>"
            f"<b>Admission Preview:</b><br>"
            f"• Directly admitted to Room 1: <b style='color:#34D399;'>{admit_preview}</b><br>"
            f"• Placed in Entrance Queue: <b style='color:#FBBF24;'>{queue_preview}</b><br>"
            f"• Room 1 resulting occupancy: <b>{r1.occupant_count + admit_preview} / {r1.capacity}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("➕ Inject Entrants", type="primary", use_container_width=True):
            res = add_person_action(count=int(add_count), destination=target_dest)
            st.rerun()

    # -------------------------------------------------------------------------
    # SECTION B: MOVE PEOPLE
    # -------------------------------------------------------------------------
    with col_move:
        st.markdown("#### 🔄 Move Between Rooms")
        st.caption("Transfer occupants across adjoining rooms via connecting doorways.")

        curr_sel = get_selected_room()
        src_default_idx = EXPECTED_ROOM_IDS.index(curr_sel) if curr_sel in EXPECTED_ROOM_IDS else 0

        source_room = st.selectbox(
            "Source Room",
            options=EXPECTED_ROOM_IDS,
            index=src_default_idx,
            key="pf_source_room",
        )
        src_obj = pipeline.engine.building.get_room(source_room)
        src_occupants = sorted(list(src_obj.occupants))

        col_m_occ1, col_m_occ2 = st.columns(2)
        with col_m_occ1:
            st.caption(f"Occupants in {source_room}: **{len(src_occupants)} / {src_obj.capacity}**")
        with col_m_occ2:
            selection_mode = st.radio(
                "Selection Method",
                options=["By Batch Count", "By Specific Occupant ID"],
                horizontal=True,
                key="pf_sel_mode",
            )

        selected_pids = None
        if selection_mode == "By Specific Occupant ID":
            selected_pids = st.multiselect(
                "Select Occupant(s) to Move",
                options=src_occupants,
                default=[],
                key="pf_selected_pids",
            )
            move_num = len(selected_pids)
        else:
            max_can_move = max(1, len(src_occupants))
            move_num = st.number_input(
                "Count to Move",
                min_value=1,
                max_value=max_can_move,
                value=min(1, len(src_occupants)) if len(src_occupants) > 0 else 1,
                step=1,
                disabled=(len(src_occupants) == 0),
                key="pf_batch_count",
            )

        # Destination Room
        adj_rooms = sorted(list(pipeline.engine.building.adjacency.get(source_room, [])))
        default_dst_idx = EXPECTED_ROOM_IDS.index(adj_rooms[0]) if adj_rooms else 1

        destination_room = st.selectbox(
            "Destination Room",
            options=EXPECTED_ROOM_IDS,
            index=default_dst_idx,
            key="pf_destination_room",
        )
        dst_obj = pipeline.engine.building.get_room(destination_room)
        dst_free = dst_obj.capacity - dst_obj.occupant_count

        # Validation logic
        is_same = source_room == destination_room
        is_direct = destination_room in adj_rooms
        has_src = len(src_occupants) > 0
        has_cap = dst_free >= move_num

        valid_transition = True
        status_msg = ""
        if is_same:
            valid_transition = False
            status_msg = "⚠️ Source and destination rooms cannot be the same."
        elif not is_direct:
            path = pipeline.engine.building.find_path(source_room, destination_room)
            inter = f"'{path[1]}'" if (path and len(path) > 2) else "an adjacent room"
            valid_transition = False
            status_msg = f"⛔ No direct doorway between {source_room} and {destination_room}. Intermediate room ({inter}) required."
        elif not has_src:
            valid_transition = False
            status_msg = f"⚠️ Source room '{source_room}' has no occupants to move."
        elif dst_free <= 0:
            valid_transition = False
            status_msg = f"⛔ Destination room '{destination_room}' is full ({dst_obj.occupant_count}/{dst_obj.capacity})."
        elif not has_cap:
            valid_transition = False
            status_msg = f"⛔ Capacity Limit: Destination room only has room for {dst_free} more, but {move_num} requested."
        else:
            valid_transition = True
            status_msg = f"✅ Valid Direct Doorway: {source_room} ↔ {destination_room} (Capacity available: {dst_free})"

        if valid_transition:
            st.success(status_msg)
        else:
            st.error(status_msg)

        if st.button("Move Occupant(s)", type="primary", use_container_width=True, disabled=not valid_transition):
            res = move_people_action(
                source_room=source_room,
                destination_room=destination_room,
                person_ids=selected_pids if selected_pids else None,
                count=int(move_num) if not selected_pids else None,
            )
            st.rerun()

    # -------------------------------------------------------------------------
    # SECTION C: EXIT PEOPLE
    # -------------------------------------------------------------------------
    with col_exit:
        st.markdown("#### ➖ Exit People")
        st.caption("Remove occupants from building through exterior exit.")

        exit_room_choice = st.selectbox(
            "Exit From",
            options=["Building-wide (Any Room)"] + EXPECTED_ROOM_IDS,
            index=0,
            key="flow_exit_room_choice",
        )
        target_exit_room = None if exit_room_choice == "Building-wide (Any Room)" else exit_room_choice

        avail_to_exit = (
            pipeline.engine.building.total_occupants
            if target_exit_room is None
            else pipeline.engine.building.get_room(target_exit_room).occupant_count
        )

        exit_count = st.number_input(
            "Count to Exit",
            min_value=1,
            max_value=max(1, avail_to_exit),
            value=min(1, avail_to_exit) if avail_to_exit > 0 else 1,
            step=1,
            disabled=(avail_to_exit == 0),
            key="flow_exit_count_input",
        )

        st.markdown(
            f"<div style='background:#1E293B; border:1px solid #334155; border-radius:6px; padding:10px; margin-bottom:12px; font-size:0.85rem;'>"
            f"<b>Exit Status:</b><br>"
            f"• Available to exit: <b>{avail_to_exit}</b><br>"
            f"• Building total occupants: <b>{pipeline.engine.building.total_occupants}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("➖ Remove Occupant(s)", use_container_width=True, disabled=(avail_to_exit == 0)):
            res = exit_person_action(count=int(exit_count), room_id=target_exit_room)
            st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION D: DIURNAL SCHEDULE VIEWER
    # -------------------------------------------------------------------------
    st.markdown("#### 📋 Diurnal Schedules & Agent Circadian Profiles")
    st.caption("Inspect pre-configured occupant circadian objectives across morning, lunch, and afternoon work intervals.")

    with st.expander("🔍 Active Occupant Population Roster & Profiles", expanded=False):
        roster_data = []
        for occ_id, occ in sorted(pipeline.engine.occupants.items()):
            sched = occ.schedule
            roster_data.append({
                "Occupant ID": occ_id,
                "Type": occ.occupant_type.capitalize(),
                "Current Room": occ.current_room_id or "Exited / Outside",
                "Home Room": getattr(sched, "primary_room_id", "Room 1"),
                "Activity": occ.current_activity,
                "State": occ.movement_state,
                "Target Destination": occ.intended_destination or "None",
            })
        if roster_data:
            st.dataframe(pd.DataFrame(roster_data), use_container_width=True, height=280)
        else:
            st.info("No occupants currently registered.")

