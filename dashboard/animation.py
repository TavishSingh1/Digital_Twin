"""
Occupant Movement Animation & Trajectory Layer
==============================================
Provides visual interpolation and trajectory planning for building occupants:
1. Deterministic stationary marker positioning derived from room geometry and occupant index.
2. Topological waypoint generation strictly through doorway openings and valid corridor routes.
3. Piecewise linear interpolation along multi-segment movement paths.
4. Support for simultaneous multi-occupant transfers, admissions, and departures.
5. Transient visual exit and entrance queue tracking.
6. Deduplicated movement event registry.
7. Strict simulation decoupling: zero mutations to logical clock, sensors, energy, or room membership.
"""

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple
import plotly.graph_objects as go

from simulation.building import Building
from simulation.occupant import Occupant
from simulation.room import Room


@dataclass
class MovementEvent:
    """Represents a confirmed occupant transition event for visual rendering."""
    event_id: str
    event_type: str  # "transfer", "admission", "exit", "step_transition"
    timestamp: str
    person_ids: list[str]
    source_room: Optional[str]
    destination_room: Optional[str]
    waypoints: list[tuple[float, float]]
    progress: float = 1.0  # 0.0 (source) to 1.0 (destination)
    status: str = "completed"  # "active", "completed"

    def get_interpolated_position(self, progress_override: Optional[float] = None) -> tuple[float, float]:
        """Calculates interpolated (x, y) coordinates along waypoints."""
        p = progress_override if progress_override is not None else self.progress
        return interpolate_along_waypoints(self.waypoints, p)


def get_deterministic_room_grid_position(
    occupant_id: str,
    room: Room,
    index: int,
    total_in_room: int,
) -> tuple[float, float]:
    """
    Computes a stable, deterministic (x, y) position for a stationary occupant.
    Arranges occupants in a structured 2D grid within room boundaries with margin offsets.
    Does not randomize coordinates across reruns.
    """
    if total_in_room <= 1:
        # Single occupant near center
        cx, cy = room.center_position()
        return round(cx, 2), round(cy, 2)

    # Determine grid dimensions based on room aspect ratio and count
    margin = 1.2
    w_avail = max(1.0, room.width - 2 * margin)
    h_avail = max(1.0, room.height - 2 * margin)

    aspect = room.width / max(1.0, room.height)
    cols = max(2, min(8, int(math.ceil(math.sqrt(total_in_room * aspect)))))
    rows = max(1, int(math.ceil(total_in_room / cols)))

    col = index % cols
    row = index // cols

    col_spacing = w_avail / max(1, cols)
    row_spacing = h_avail / max(1, rows)

    x = room.x + margin + (col + 0.5) * col_spacing
    y = room.y + margin + (row + 0.5) * row_spacing

    # Clamp safely inside room boundary
    clamped_x, clamped_y = room.clamp_position(x, y)
    return round(clamped_x, 2), round(clamped_y, 2)


def calculate_route_waypoints(
    building: Building,
    source_room: Optional[str],
    destination_room: Optional[str],
    start_pos: Optional[tuple[float, float]] = None,
    end_pos: Optional[tuple[float, float]] = None,
) -> list[tuple[float, float]]:
    """
    Generates a list of (x, y) waypoints passing strictly through door openings.
    Ensures occupants never pass through solid structural walls.
    """
    waypoints: list[tuple[float, float]] = []

    # Case 1: Entrance Admission (Outside -> Room 1 -> Destination)
    if source_room is None or source_room == "Outside":
        ext_queue_pos = (7.0, -1.6)
        ext_door_pos = (7.0, 0.0)
        waypoints.append(start_pos or ext_queue_pos)
        waypoints.append(ext_door_pos)

        target_r = destination_room or "Room 1"
        if target_r == "Room 1":
            r1 = building.get_room("Room 1")
            waypoints.append(end_pos or r1.center_position())
            return waypoints

        # Route through Room 1 to destination
        path = building.find_path("Room 1", target_r)
        if not path:
            dest_obj = building.get_room(target_r)
            waypoints.append(end_pos or (dest_obj.center_position() if dest_obj else (14.0, 5.0)))
            return waypoints

        # Traverse interior doors from Room 1
        for i in range(len(path) - 1):
            r_from = path[i]
            r_to = path[i + 1]
            door = building.door_positions.get((r_from, r_to))
            if door:
                waypoints.append(door)

        dst = building.get_room(target_r)
        waypoints.append(end_pos or dst.center_position())
        return waypoints

    # Case 2: Departure / Exit (Source -> Doors to Room 1 -> Exterior Door -> Outside)
    if destination_room is None or destination_room == "Outside":
        src = building.get_room(source_room)
        waypoints.append(start_pos or (src.center_position() if src else (14.0, 5.0)))

        if source_room != "Room 1":
            path = building.find_path(source_room, "Room 1")
            if path:
                for i in range(len(path) - 1):
                    r_from = path[i]
                    r_to = path[i + 1]
                    door = building.door_positions.get((r_from, r_to))
                    if door:
                        waypoints.append(door)

        # Main exterior exit at Room 1
        ext_door_pos = (7.0, 0.0)
        ext_outside_pos = (7.0, -1.8)
        waypoints.append(ext_door_pos)
        waypoints.append(end_pos or ext_outside_pos)
        return waypoints

    # Case 3: Inter-Room Transfer (Source -> Connected Doors -> Destination)
    src = building.get_room(source_room)
    dst = building.get_room(destination_room)
    if not src or not dst:
        return [start_pos or (0.0, 0.0), end_pos or (0.0, 0.0)]

    waypoints.append(start_pos or src.center_position())

    # Direct door connection
    if destination_room in building.adjacency.get(source_room, set()):
        door = building.door_positions.get((source_room, destination_room))
        if door:
            waypoints.append(door)
    else:
        # Multi-hop route via intermediate rooms
        path = building.find_path(source_room, destination_room)
        if path:
            for i in range(len(path) - 1):
                r_from = path[i]
                r_to = path[i + 1]
                door = building.door_positions.get((r_from, r_to))
                if door:
                    waypoints.append(door)

    waypoints.append(end_pos or dst.center_position())
    return waypoints


def interpolate_along_waypoints(
    waypoints: list[tuple[float, float]],
    progress: float,
) -> tuple[float, float]:
    """
    Interpolates an (x, y) position along piecewise linear waypoints for progress in [0.0, 1.0].
    """
    if not waypoints:
        return 0.0, 0.0
    if len(waypoints) == 1:
        return waypoints[0]

    clamped_p = max(0.0, min(1.0, float(progress)))
    if clamped_p <= 0.0:
        return waypoints[0]
    if clamped_p >= 1.0:
        return waypoints[-1]

    # Calculate segment lengths
    segment_lengths = []
    total_length = 0.0
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        dist = math.hypot(x2 - x1, y2 - y1)
        segment_lengths.append(dist)
        total_length += dist

    if total_length == 0.0:
        return waypoints[0]

    target_dist = clamped_p * total_length
    accumulated = 0.0

    for i, seg_len in enumerate(segment_lengths):
        if accumulated + seg_len >= target_dist or i == len(segment_lengths) - 1:
            seg_progress = (target_dist - accumulated) / max(1e-6, seg_len)
            seg_progress = max(0.0, min(1.0, seg_progress))
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            interp_x = x1 + seg_progress * (x2 - x1)
            interp_y = y1 + seg_progress * (y2 - y1)
            return round(interp_x, 2), round(interp_y, 2)
        accumulated += seg_len

    return waypoints[-1]


class AnimationManager:
    """
    Maintains transient visual animation state, trajectory history,
    interactive replay controls, and deterministic deduplicated event logging.
    """

    def __init__(self):
        self.events: list[MovementEvent] = []
        self.exit_records: list[dict[str, Any]] = []
        self.seen_event_hashes: set[str] = set()
        self.show_occupants: bool = True
        self.show_paths: bool = True
        self.animation_progress: float = 1.0  # 1.0 = completed positions
        self.animation_speed: float = 1.0
        self.selected_occupant_id: Optional[str] = None
        self.is_paused: bool = False
        self.replay_event_id: Optional[str] = None
        self.replay_progress: float = 0.0
        self.max_events: int = 100
        self.max_exit_records: int = 25

    def record_movement(
        self,
        event_type: str,
        timestamp: str,
        person_ids: list[str],
        source_room: Optional[str],
        destination_room: Optional[str],
        building: Building,
        start_pos: Optional[tuple[float, float]] = None,
        end_pos: Optional[tuple[float, float]] = None,
    ) -> Optional[MovementEvent]:
        """
        Deduplicates and registers a confirmed movement event.
        Calculates doorway waypoints without mutating simulation state.
        """
        if not person_ids:
            return None

        # Build unique deterministic hash
        hash_str = f"{event_type}_{timestamp}_{source_room}_{destination_room}_{'_'.join(sorted(person_ids))}"
        event_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()

        if event_hash in self.seen_event_hashes:
            return None  # Prevent duplicate registration on Streamlit reruns

        self.seen_event_hashes.add(event_hash)

        waypoints = calculate_route_waypoints(
            building=building,
            source_room=source_room,
            destination_room=destination_room,
            start_pos=start_pos,
            end_pos=end_pos,
        )

        event = MovementEvent(
            event_id=event_hash[:8],
            event_type=event_type,
            timestamp=timestamp,
            person_ids=list(person_ids),
            source_room=source_room,
            destination_room=destination_room,
            waypoints=waypoints,
            progress=1.0,
            status="completed",
        )

        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)  # Bounded retention: keep rolling history of last 100 events

        # If exit event, log temporary visual exit marker (outside Room 1)
        if event_type == "exit" or destination_room is None:
            for pid in person_ids:
                self.exit_records.append({
                    "occupant_id": pid,
                    "from_room": source_room,
                    "timestamp": timestamp,
                    "position": (7.0, -1.8),
                })
            if len(self.exit_records) > self.max_exit_records:
                self.exit_records = self.exit_records[-self.max_exit_records:]

        return event

    def set_speed(self, speed: float):
        """Sets visual animation playback speed without mutating simulation clock."""
        if speed in [0.5, 1.0, 1.5, 2.0]:
            self.animation_speed = speed

    def pause(self):
        """Pauses visual playback."""
        self.is_paused = True

    def resume(self):
        """Resumes visual playback."""
        self.is_paused = False

    def toggle_pause(self) -> bool:
        """Toggles visual playback pause state."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def start_replay(self, event_id: str) -> bool:
        """
        Initiates visual replay of a recorded movement event.
        Strictly visual: zero mutations to simulation state or room memberships.
        """
        for ev in self.events:
            if ev.event_id == event_id:
                self.replay_event_id = event_id
                self.replay_progress = 0.0
                return True
        return False

    def stop_replay(self):
        """Terminates visual replay and returns to live view."""
        self.replay_event_id = None
        self.replay_progress = 0.0

    def get_active_replay_event(self) -> Optional[MovementEvent]:
        """Returns the currently active replaying event, if any."""
        if not self.replay_event_id:
            return None
        for ev in self.events:
            if ev.event_id == self.replay_event_id:
                return ev
        return None

    def filter_events(
        self,
        room_id: Optional[str] = None,
        occupant_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[MovementEvent]:
        """Filters recorded movement events by room, occupant, and event type."""
        filtered = list(self.events)
        if room_id and room_id != "All Rooms":
            filtered = [
                ev for ev in filtered
                if ev.source_room == room_id or ev.destination_room == room_id
            ]
        if occupant_id and occupant_id.strip():
            needle = occupant_id.strip().upper()
            filtered = [
                ev for ev in filtered
                if any(needle in pid.upper() for pid in ev.person_ids)
            ]
        if event_type and event_type != "All Types":
            ev_lower = event_type.lower()
            filtered = [ev for ev in filtered if ev.event_type.lower() == ev_lower]
        return filtered

    def validate_selected_occupant(self, occupants: dict[str, Occupant]) -> Optional[str]:
        """
        Ensures selected occupant is still valid and inside the building.
        Handles stale selections gracefully if occupant has exited or been removed.
        """
        if not self.selected_occupant_id:
            return None
        occ = occupants.get(self.selected_occupant_id)
        if not occ or not occ.is_inside_building:
            self.selected_occupant_id = None
            return None
        return self.selected_occupant_id

    def clear(self):
        """Clears all transient animation states, events, exit records, and replay states."""
        self.events.clear()
        self.exit_records.clear()
        self.seen_event_hashes.clear()
        self.selected_occupant_id = None
        self.animation_progress = 1.0
        self.animation_speed = 1.0
        self.is_paused = False
        self.replay_event_id = None
        self.replay_progress = 0.0

    def get_occupant_positions(
        self,
        building: Building,
        occupants: dict[str, Occupant],
        progress: Optional[float] = None,
    ) -> dict[str, tuple[float, float]]:
        """
        Computes current display positions for all occupants.
        Stationary occupants receive stable grid positions.
        Moving occupants in active events or active replay receive interpolated positions along their waypoints.
        """
        positions: dict[str, tuple[float, float]] = {}
        prog = progress if progress is not None else self.animation_progress

        # Group stationary occupants by room
        room_occupants: dict[str, list[str]] = {r: [] for r in building.rooms.keys()}
        for pid, occ in sorted(occupants.items()):
            if occ.is_inside_building and occ.current_room_id in room_occupants:
                room_occupants[occ.current_room_id].append(pid)

        # Place stationary occupants using deterministic grid
        for r_id, pids in room_occupants.items():
            rm = building.get_room(r_id)
            total = len(pids)
            for idx, pid in enumerate(pids):
                positions[pid] = get_deterministic_room_grid_position(pid, rm, idx, total)

        # Check for active replay override first
        replay_ev = self.get_active_replay_event()
        if replay_ev:
            interp_pos = interpolate_along_waypoints(replay_ev.waypoints, self.replay_progress)
            for pid in replay_ev.person_ids:
                positions[pid] = interp_pos
            return positions

        # If an active event is interpolating, override moving occupant positions
        if prog < 1.0 and self.events:
            latest_ev = self.events[-1]
            if latest_ev.status == "active" or prog < 0.99:
                interp_pos = interpolate_along_waypoints(latest_ev.waypoints, prog)
                for pid in latest_ev.person_ids:
                    if pid in occupants:
                        positions[pid] = interp_pos

        return positions

    def get_selected_route_waypoints(
        self,
        building: Building,
        occupant: Occupant,
    ) -> list[tuple[float, float]]:
        """Returns planned doorway waypoints for a specific selected occupant."""
        if not occupant:
            return []
        src = occupant.current_room_id
        dst = occupant.intended_destination
        return calculate_route_waypoints(
            building=building,
            source_room=src,
            destination_room=dst,
            start_pos=occupant.position,
        )

