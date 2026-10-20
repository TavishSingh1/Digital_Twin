"""
Virtual Occupant State & Movement System
========================================
Implements individual occupant state, 2D continuous spatial positioning,
door-based topological transitions, and capacity awareness.
"""

import math
import random
from typing import Optional
from simulation.building import Building
from simulation.schedules import OccupantSchedule


class Occupant:
    def __init__(
        self,
        occupant_id: str,
        occupant_type: str,
        schedule: OccupantSchedule,
    ):
        self.occupant_id = occupant_id
        self.occupant_type = occupant_type
        self.schedule = schedule

        # State model
        self.current_room_id: Optional[str] = None
        self.position: tuple[float, float] = (0.0, 0.0)
        self.current_activity: str = "off_site"
        self.intended_destination: Optional[str] = None
        self.movement_state: str = "exited"  # "stationary", "moving", "waiting_for_capacity", "exited"
        
        # Route: list of room IDs to traverse [next_room, ..., destination]
        self.planned_route: list[str] = []

    @property
    def is_inside_building(self) -> bool:
        """Returns True if the occupant is currently inside any room of the building."""
        return self.current_room_id is not None

    def update_decision(self, current_dt, building: Building):
        """
        Consults behavioral schedule to determine target room and planned route.
        """
        desired_room, activity = self.schedule.get_desired_target(current_dt)
        self.current_activity = activity
        self.intended_destination = desired_room

        # Case 1: Off-site and desires to be off-site -> stay outside
        if desired_room is None and not self.is_inside_building:
            self.movement_state = "exited"
            self.planned_route.clear()
            return

        # Case 2: Desires to enter building from outside
        if not self.is_inside_building and desired_room is not None:
            # Plan route: enters through building exterior exit door
            exit_rooms = building.get_exterior_exit_rooms()
            if not exit_rooms:
                return
            entry_room = exit_rooms[0]
            if entry_room == desired_room:
                self.planned_route = [entry_room]
            else:
                p = building.find_path(entry_room, desired_room)
                self.planned_route = p if p else [entry_room]
            self.movement_state = "moving"
            return

        # Case 3: Inside building and desires to leave (desired_room is None)
        if self.is_inside_building and desired_room is None:
            route_res = building.find_path_to_exterior(self.current_room_id)
            if route_res is not None:
                path, _ = route_res
                # Path includes [current_room, ..., exit_room]
                self.planned_route = path[1:]  # Next rooms to traverse before leaving
                self.movement_state = "moving"
            else:
                self.movement_state = "stationary"
            return

        # Case 4: Inside building and already in desired room -> stationary activity
        if self.is_inside_building and self.current_room_id == desired_room:
            self.movement_state = "stationary"
            self.planned_route.clear()
            return

        # Case 5: Inside building, desires to move to a different room
        if self.is_inside_building and desired_room is not None and self.current_room_id != desired_room:
            path = building.find_path(self.current_room_id, desired_room)
            if path is not None and len(path) > 1:
                self.planned_route = path[1:]  # Steps to take
                self.movement_state = "moving"
            else:
                # Disconnected room; cannot reach through walls
                self.movement_state = "stationary"
                self.planned_route.clear()

    def step_movement(
        self,
        building: Building,
        rng: random.Random,
    ) -> tuple[int, int, int, int]:
        """
        Executes one 5-minute timestep of movement:
        Returns:
            (arrivals, departures, incoming_transitions, outgoing_transitions)
            for metrics tracking.
        """
        arr, dep, inc, out = 0, 0, 0, 0

        # Case 1: Outside building attempting to enter
        if not self.is_inside_building:
            if self.intended_destination is not None and self.planned_route:
                entry_room_id = self.planned_route[0]
                entry_room = building.get_room(entry_room_id)

                if entry_room.can_accept():
                    entry_room.add_occupant(self.occupant_id)
                    self.current_room_id = entry_room_id
                    # Place near door position or center
                    door_pos = entry_room.exterior_door_position or entry_room.center_position()
                    self.position = entry_room.clamp_position(door_pos[0], door_pos[1])
                    self.planned_route.pop(0)
                    arr += 1
                    inc += 1
                    self.movement_state = "stationary" if not self.planned_route else "moving"
                else:
                    self.movement_state = "waiting_for_capacity"
            return arr, dep, inc, out

        curr_room = building.get_room(self.current_room_id)

        # Case 2: Inside building, leaving exterior
        if self.intended_destination is None:
            if not self.planned_route:
                # Currently in exit room -> exit building now
                if curr_room.has_exterior_exit:
                    curr_room.remove_occupant(self.occupant_id)
                    self.current_room_id = None
                    self.position = (0.0, 0.0)
                    self.movement_state = "exited"
                    dep += 1
                    out += 1
                return arr, dep, inc, out

        # Case 3: Transitioning to next room along route
        if self.planned_route:
            next_room_id = self.planned_route[0]
            next_room = building.get_room(next_room_id)

            if next_room.can_accept():
                # Perform valid transition through door
                curr_room.remove_occupant(self.occupant_id)
                next_room.add_occupant(self.occupant_id)
                
                # Get door position
                door_pos = building.door_positions.get(
                    (self.current_room_id, next_room_id),
                    next_room.center_position()
                )
                self.current_room_id = next_room_id
                self.position = next_room.clamp_position(door_pos[0], door_pos[1])
                self.planned_route.pop(0)

                out += 1
                inc += 1
                self.movement_state = "stationary" if not self.planned_route else "moving"
            else:
                # Room at capacity; wait in current room
                self.movement_state = "waiting_for_capacity"
                return arr, dep, inc, out

        # Case 4: Stationary inside room -> stochastic micro-wander within boundaries
        if self.is_inside_building and self.movement_state in ("stationary", "waiting_for_capacity"):
            px, py = self.position
            # Small random drift inside room (e.g. pacing, adjusting chair)
            dx = rng.uniform(-0.8, 0.8)
            dy = rng.uniform(-0.8, 0.8)
            self.position = curr_room.clamp_position(px + dx, py + dy)

        return arr, dep, inc, out

    def set_manual_destination(self, destination: str):
        """
        Manually assigns this occupant to a room, updating routing and schedule overrides.
        """
        self.current_room_id = destination
        self.intended_destination = destination
        self.planned_route.clear()
        self.movement_state = "stationary"
        if hasattr(self.schedule, "set_manual_override"):
            self.schedule.set_manual_override(destination)
        elif hasattr(self.schedule, "manual_destination_override"):
            self.schedule.manual_destination_override = destination
        elif hasattr(self.schedule, "destination"):
            self.schedule.destination = destination

    def reset(self):
        """Resets occupant to default off-site state."""
        self.current_room_id = None
        self.position = (0.0, 0.0)
        self.current_activity = "off_site"
        self.intended_destination = None
        self.movement_state = "exited"
        self.planned_route.clear()
        if hasattr(self.schedule, "clear_manual_override"):
            self.schedule.clear_manual_override()
        elif hasattr(self.schedule, "manual_destination_override"):
            self.schedule.manual_destination_override = None

    def __repr__(self) -> str:
        loc = self.current_room_id or "Outside"
        return f"<Occupant '{self.occupant_id}' ({self.occupant_type}) in {loc} [{self.movement_state}]>"

