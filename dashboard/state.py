"""
Streamlit Session State & Simulation Control Manager
====================================================
Maintains persistent simulation and pipeline state across Streamlit UI reruns.
Prevents accidental multiple stepping, handles model loading errors, coordinates
continuous playback, scenario applications, and deterministic reset/replay workflows.
"""

from typing import Optional, Any
import streamlit as st

from simulation.engine import SimulationEngine
from simulation.pipeline import DigitalTwinPipeline
from simulation.config import EXPECTED_ROOM_IDS
from stgnn_model import STGNNInference
from dashboard.animation import AnimationManager


def init_session_state():
    """Initializes persistent pipeline and control state in st.session_state."""
    if "seed" not in st.session_state:
        st.session_state.seed = 42
    if "num_occupants" not in st.session_state:
        st.session_state.num_occupants = 40
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0  # seconds between automatic steps
    if "is_auto_playing" not in st.session_state:
        st.session_state.is_auto_playing = False
    if "step_duration_minutes" not in st.session_state:
        st.session_state.step_duration_minutes = 5
    if "selected_room" not in st.session_state:
        st.session_state.selected_room = "Room 1"
    if "selected_overlay" not in st.session_state:
        target_ov = st.query_params.get("overlay", "Occupancy")
        if target_ov:
            target_ov = target_ov.replace("+", " ")
        st.session_state.selected_overlay = target_ov
    elif "overlay" in st.query_params:
        target_ov = st.query_params.get("overlay")
        if target_ov:
            target_ov = target_ov.replace("+", " ")
        st.session_state.selected_overlay = target_ov
    if "active_scenario" not in st.session_state:
        st.session_state.active_scenario = "Normal Workday"
    if "action_feedback" not in st.session_state:
        st.session_state.action_feedback = None  # Tuple of (type, message)
    if "model_load_error" not in st.session_state:
        st.session_state.model_load_error = None
    if "animation_manager" not in st.session_state:
        st.session_state.animation_manager = AnimationManager()

    if "pipeline" not in st.session_state:
        try:
            # Instantiate pipeline
            engine = SimulationEngine(
                num_occupants=st.session_state.num_occupants,
                seed=st.session_state.seed,
            )
            pipeline = DigitalTwinPipeline(
                engine=engine,
                seed=st.session_state.seed,
            )
            st.session_state.pipeline = pipeline
            st.session_state.model_load_error = None
        except Exception as e:
            st.session_state.model_load_error = str(e)
            st.session_state.pipeline = None


def get_pipeline() -> Optional[DigitalTwinPipeline]:
    """Retrieves the active pipeline instance from session state."""
    init_session_state()
    return st.session_state.get("pipeline")


def get_animation_manager() -> AnimationManager:
    """Retrieves or initializes the AnimationManager instance from session state."""
    init_session_state()
    if "animation_manager" not in st.session_state or st.session_state.animation_manager is None:
        st.session_state.animation_manager = AnimationManager()
    return st.session_state.animation_manager


def get_selected_room() -> str:
    """Returns the currently active room ID (defaults to 'Room 1')."""
    init_session_state()
    try:
        if hasattr(st, "query_params") and "selected_room" in st.query_params:
            q_room = st.query_params["selected_room"]
            if q_room in EXPECTED_ROOM_IDS and q_room != st.session_state.get("selected_room"):
                st.session_state.selected_room = q_room
    except Exception:
        pass
    room = st.session_state.get("selected_room", "Room 1")
    if room not in EXPECTED_ROOM_IDS:
        room = "Room 1"
        st.session_state.selected_room = room
    return room


def set_selected_room(room_id: str):
    """Updates the globally synchronized selected room."""
    if room_id in EXPECTED_ROOM_IDS:
        st.session_state.selected_room = room_id


def get_selected_overlay() -> str:
    """Returns the currently active floor plan visualization overlay."""
    init_session_state()
    return st.session_state.get("selected_overlay", "Occupancy")


def set_selected_overlay(overlay: str):
    """Updates the active floor plan visualization overlay."""
    st.session_state.selected_overlay = overlay


def set_action_feedback(feedback_type: str, message: str):
    """Sets a temporary feedback banner message."""
    st.session_state.action_feedback = (feedback_type, message)


def clear_action_feedback():
    """Clears the temporary feedback message."""
    st.session_state.action_feedback = None


def advance_pipeline_step(step_minutes: int = 5) -> Optional[dict[str, Any]]:
    """
    Advances the pipeline by the configured duration.
    Since the base physical and STGNN cadence is 5 minutes,
    advances by max(1, step_minutes // 5) steps.
    """
    pipeline = get_pipeline()
    if pipeline is not None:
        steps_to_run = max(1, step_minutes // 5)
        last_res = None
        anim_mgr = get_animation_manager()
        for _ in range(steps_to_run):
            # Track previous room locations
            prev_rooms = {
                pid: occ.current_room_id
                for pid, occ in pipeline.engine.occupants.items()
                if occ.is_inside_building
            }
            last_res = pipeline.step()
            curr_rooms = {
                pid: occ.current_room_id
                for pid, occ in pipeline.engine.occupants.items()
                if occ.is_inside_building
            }
            # Record admissions, transfers, and exits
            ts = pipeline.engine.current_time.isoformat()
            transfers: dict[tuple[Optional[str], Optional[str]], list[str]] = {}
            for pid, c_room in curr_rooms.items():
                p_room = prev_rooms.get(pid)
                if p_room and c_room and p_room != c_room:
                    transfers.setdefault((p_room, c_room), []).append(pid)
                elif not p_room and c_room:
                    transfers.setdefault(("Outside", c_room), []).append(pid)
            for pid, p_room in prev_rooms.items():
                if pid not in curr_rooms and p_room:
                    transfers.setdefault((p_room, "Outside"), []).append(pid)

            for (s_r, d_r), pids in transfers.items():
                ev_type = "transfer"
                d_target = d_r
                if s_r == "Outside":
                    ev_type = "admission"
                elif d_r == "Outside":
                    ev_type = "exit"
                    d_target = None
                anim_mgr.record_movement(
                    event_type=ev_type,
                    timestamp=ts,
                    person_ids=pids,
                    source_room=s_r if s_r != "Outside" else None,
                    destination_room=d_target,
                    building=pipeline.engine.building,
                )
        return last_res
    return None


def run_pipeline_steps(num_steps: int) -> Optional[dict[str, Any]]:
    """Advances the pipeline by num_steps."""
    pipeline = get_pipeline()
    if pipeline is not None:
        last_res = None
        for _ in range(num_steps):
            last_res = pipeline.step()
        return last_res
    return None


def add_person_action(count: int = 1, destination: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Interactively adds count occupants through Room 1 (Main Entrance).
    Admitted to Room 1 if capacity allows; queued otherwise.
    """
    pipeline = get_pipeline()
    if pipeline is not None:
        res = pipeline.add_person(count=count, destination=destination)
        if res:
            adm = res.get("admitted_count", 0)
            que = res.get("queued_count", 0)
            target = destination or "Free Movement"
            r1_obj = pipeline.engine.building.get_room("Room 1")
            msg = f"Added {count} occupant(s) towards {target}. {adm} admitted to Room 1 ({r1_obj.occupant_count}/{r1_obj.capacity})."
            if que > 0:
                msg += f" {que} placed in Entrance Queue."
            set_action_feedback("success" if que == 0 else "warning", msg)

            # Record confirmed visual admission event
            admitted_ids = res.get("admitted_ids", [])
            if admitted_ids:
                anim_mgr = get_animation_manager()
                anim_mgr.record_movement(
                    event_type="admission",
                    timestamp=pipeline.engine.current_time.isoformat(),
                    person_ids=admitted_ids,
                    source_room="Outside",
                    destination_room="Room 1",
                    building=pipeline.engine.building,
                )
        return res
    return None


def exit_person_action(count: int = 1, room_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Interactively exits occupants from a specific room or building-wide.
    """
    pipeline = get_pipeline()
    if pipeline is not None:
        res = pipeline.exit_person(count=count, room_id=room_id)
        if res:
            exited_ids = res.get("exited_ids", [])
            ext = res.get("exited_count", len(exited_ids))
            src_str = room_id if room_id else "Building-wide"
            set_action_feedback("info", f"Exited {ext} occupant(s) from {src_str}. Remaining building occupancy: {pipeline.engine.building.total_occupants}.")

            # Record confirmed visual departure event
            if exited_ids:
                anim_mgr = get_animation_manager()
                anim_mgr.record_movement(
                    event_type="exit",
                    timestamp=pipeline.engine.current_time.isoformat(),
                    person_ids=exited_ids,
                    source_room=room_id or "Room 1",
                    destination_room=None,
                    building=pipeline.engine.building,
                )
        return res
    return None


def move_people_action(
    source_room: str,
    destination_room: str,
    person_ids: Optional[list[str]] = None,
    count: Optional[int] = None,
) -> dict[str, Any]:
    """
    Executes an interactive manual movement of occupants between rooms.
    Returns a result dict containing success status, moved count/IDs, or error message.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        return {"success": False, "error": "Simulation pipeline is not initialized."}
    try:
        res = pipeline.move_people(
            source_room=source_room,
            destination_room=destination_room,
            person_ids=person_ids,
            count=count,
        )
        if res.get("success"):
            src_rm = pipeline.engine.building.get_room(source_room)
            dst_rm = pipeline.engine.building.get_room(destination_room)
            msg = (
                f"Moved {res['moved_count']} occupant(s) from {source_room} ({src_rm.occupant_count}/{src_rm.capacity}) "
                f"to {destination_room} ({dst_rm.occupant_count}/{dst_rm.capacity})."
            )
            set_action_feedback("success", msg)

            # Record confirmed visual inter-room transfer
            moved_ids = res.get("moved_ids", [])
            if moved_ids:
                anim_mgr = get_animation_manager()
                anim_mgr.record_movement(
                    event_type="transfer",
                    timestamp=pipeline.engine.current_time.isoformat(),
                    person_ids=moved_ids,
                    source_room=source_room,
                    destination_room=destination_room,
                    building=pipeline.engine.building,
                )
        else:
            set_action_feedback("error", f"Movement Failed: {res.get('error', 'Unknown error')}")
        return res
    except Exception as e:
        err_msg = str(e)
        set_action_feedback("error", f"Movement Failed: {err_msg}")
        return {"success": False, "error": err_msg}


def reset_simulation(
    seed: Optional[int] = None,
    num_occupants: Optional[int] = None,
    preserve_person_counter: bool = True,
):
    """Resets the simulation with specified or default seed and population."""
    if seed is not None:
        st.session_state.seed = seed
    if num_occupants is not None:
        st.session_state.num_occupants = num_occupants

    s = st.session_state.seed
    n = st.session_state.num_occupants

    # Capture current person counter if preserving across resets
    prev_counter = 1
    curr_pipeline = st.session_state.get("pipeline")
    if preserve_person_counter and curr_pipeline and hasattr(curr_pipeline, "engine"):
        prev_counter = curr_pipeline.engine.person_counter

    try:
        engine = SimulationEngine(num_occupants=n, seed=s, person_counter=prev_counter)
        pipeline = DigitalTwinPipeline(engine=engine, seed=s)
        st.session_state.pipeline = pipeline
        st.session_state.is_auto_playing = False
        st.session_state.model_load_error = None
        anim_mgr = get_animation_manager()
        anim_mgr.clear()
        set_action_feedback("info", f"Simulation reset completed with seed={s}, initial population={n}. All telemetry, visual animations, and energy meters cleared.")
    except Exception as e:
        st.session_state.model_load_error = str(e)
        st.session_state.pipeline = None
        set_action_feedback("error", f"Simulation Reset Failed: {e}")
