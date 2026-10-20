"""
Virtual Building & Topological Navigation
=========================================
Manages the five virtual rooms, spatial layout, door connections, route
planning (BFS pathfinding), and building-wide population conservation.
"""

from collections import deque
from typing import Optional
from simulation.config import BuildingConfig, EXPECTED_ROOM_IDS, ConfigurationError
from simulation.room import Room, Door


class Building:
    def __init__(self, config: BuildingConfig):
        self.config = config
        self.building_name = config.building_name
        self.rooms: dict[str, Room] = {}
        self.adjacency: dict[str, set[str]] = {rid: set() for rid in EXPECTED_ROOM_IDS}
        self.door_positions: dict[tuple[str, str], tuple[float, float]] = {}

        self._initialize_rooms()
        self._initialize_connections()

    def _initialize_rooms(self):
        """Initializes Room entities according to configuration."""
        for rc in self.config.rooms:
            room = Room(
                room_id=rc.id,
                name=rc.name,
                width=rc.width,
                height=rc.height,
                x=rc.x,
                y=rc.y,
                capacity=rc.capacity,
                has_exterior_exit=rc.has_exterior_exit,
                exterior_door_position=rc.exterior_door_position,
            )
            self.rooms[rc.id] = room

    def _initialize_connections(self):
        """Constructs bidirectional door connections between rooms."""
        for conn in self.config.connections:
            ra = conn.room_a
            rb = conn.room_b
            pos = conn.door_position
            width = conn.door_width

            # Add to adjacency graph
            self.adjacency[ra].add(rb)
            self.adjacency[rb].add(ra)

            # Store door coordinate mapping
            self.door_positions[(ra, rb)] = pos
            self.door_positions[(rb, ra)] = pos

            # Add doors to rooms
            self.rooms[ra].doors.append(Door(target_room_id=rb, position=pos, width=width))
            self.rooms[rb].doors.append(Door(target_room_id=ra, position=pos, width=width))

    def get_room(self, room_id: str) -> Room:
        """Retrieves room by ID or raises KeyError."""
        if room_id in self.rooms:
            return self.rooms[room_id]
        normalized = room_id.replace("_", " ")
        if normalized in self.rooms:
            return self.rooms[normalized]
        raise KeyError(f"Room '{room_id}' does not exist in building.")

    @property
    def total_occupants(self) -> int:
        """Total number of occupants currently inside all rooms of the building."""
        return sum(room.occupant_count for room in self.rooms.values())

    @property
    def total_capacity(self) -> int:
        """Total maximum capacity across all five rooms."""
        return sum(room.capacity for room in self.rooms.values())

    def get_exterior_exit_rooms(self) -> list[str]:
        """Returns IDs of all rooms that possess an exterior exit."""
        return [r.room_id for r in self.rooms.values() if r.has_exterior_exit]

    def find_path(self, start_room_id: str, goal_room_id: str) -> Optional[list[str]]:
        """
        Computes the shortest topological room sequence between start and goal
        using Breadth-First Search (BFS) over configured door connections.
        Returns a list of room IDs [start, ..., goal], or None if unreachable.
        """
        if start_room_id == goal_room_id:
            return [start_room_id]
        if start_room_id not in self.rooms or goal_room_id not in self.rooms:
            return None

        queue = deque([[start_room_id]])
        visited = {start_room_id}

        while queue:
            path = queue.popleft()
            curr = path[-1]

            if curr == goal_room_id:
                return path

            for neighbor in sorted(self.adjacency[curr]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return None  # No valid route exists through doors

    def find_path_to_exterior(self, start_room_id: str) -> Optional[tuple[list[str], tuple[float, float]]]:
        """
        Finds the shortest route from a room to an exterior exit.
        Returns (room_path, exterior_door_position) or None.
        """
        exit_rooms = self.get_exterior_exit_rooms()
        if not exit_rooms:
            return None

        best_path = None
        best_exit_room = None

        for er in exit_rooms:
            p = self.find_path(start_room_id, er)
            if p is not None:
                if best_path is None or len(p) < len(best_path):
                    best_path = p
                    best_exit_room = er

        if best_path and best_exit_room:
            door_pos = self.rooms[best_exit_room].exterior_door_position
            if door_pos is None:
                door_pos = self.rooms[best_exit_room].center_position()
            return best_path, door_pos

        return None

    def validate_invariants(self, all_active_occupant_ids: set[str]):
        """
        Validates core physical and demographic invariants:
        1. Non-negative occupancy in every room.
        2. No room exceeds its maximum configured capacity.
        3. Every occupant in the building belongs to exactly one room.
        4. Total occupant count matches sum across rooms.
        """
        accounted_occupants = set()
        for room_id in EXPECTED_ROOM_IDS:
            room = self.rooms[room_id]
            if room.occupant_count < 0:
                raise ValueError(f"Room '{room_id}' has negative occupancy: {room.occupant_count}")
            if room.occupant_count > room.capacity:
                raise ValueError(
                    f"Room '{room_id}' exceeded capacity: {room.occupant_count} > {room.capacity}"
                )

            intersection = accounted_occupants & room.occupants
            if intersection:
                raise ValueError(
                    f"Occupants {intersection} exist in multiple rooms simultaneously!"
                )
            accounted_occupants.update(room.occupants)

        # Ensure no active occupants inside building are missing from room sets
        if accounted_occupants != all_active_occupant_ids:
            diff = all_active_occupant_ids.symmetric_difference(accounted_occupants)
            raise ValueError(
                f"Discrepancy in occupant accounting between building and active list: {diff}"
            )

    def reset(self):
        """Resets all rooms in the building to zero occupants."""
        for room in self.rooms.values():
            room.reset()

