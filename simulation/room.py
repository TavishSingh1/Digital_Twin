"""
Room Entity & Capacity Management
=================================
Represents a physical room with spatial bounds, door locations, capacity limits,
and occupant membership tracking.
"""

from dataclasses import dataclass, field
from typing import Optional


class RoomCapacityExceededError(ValueError):
    """Raised when an attempt is made to exceed a room's maximum capacity."""
    pass


class OccupantNotFoundError(KeyError):
    """Raised when removing an occupant that is not recorded in the room."""
    pass


@dataclass
class Door:
    """Represents a doorway connecting this room to another room or exterior."""
    target_room_id: Optional[str]  # None indicates exterior exit
    position: tuple[float, float]
    width: float = 1.0


class Room:
    def __init__(
        self,
        room_id: str,
        name: str,
        width: float,
        height: float,
        x: float,
        y: float,
        capacity: int,
        has_exterior_exit: bool = False,
        exterior_door_position: Optional[tuple[float, float]] = None,
    ):
        self.room_id = room_id
        self.name = name
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.capacity = capacity
        self.has_exterior_exit = has_exterior_exit
        self.exterior_door_position = exterior_door_position

        # Membership and topological doors
        self._occupants: set[str] = set()
        self.doors: list[Door] = []
        
        # Historical ground truth records for this room: list of (timestamp_str, count)
        self.occupancy_history: list[tuple[str, int]] = []

        if self.has_exterior_exit and self.exterior_door_position is not None:
            self.doors.append(Door(
                target_room_id=None,
                position=self.exterior_door_position,
                width=1.2,
            ))

    @property
    def occupant_count(self) -> int:
        """Ground-truth occupant count. Always non-negative integer <= capacity."""
        return len(self._occupants)

    @property
    def occupants(self) -> set[str]:
        """Read-only view of occupants currently inside this room."""
        return set(self._occupants)

    @property
    def is_full(self) -> bool:
        """Returns True if the room has reached its maximum capacity."""
        return len(self._occupants) >= self.capacity

    def can_accept(self) -> bool:
        """Returns True if at least one more occupant can enter without exceeding capacity."""
        return len(self._occupants) < self.capacity

    def add_occupant(self, occupant_id: str):
        """Adds an occupant to the room, strictly enforcing room capacity."""
        if occupant_id in self._occupants:
            return  # Already present
        if len(self._occupants) >= self.capacity:
            raise RoomCapacityExceededError(
                f"Cannot add occupant '{occupant_id}' to room '{self.room_id}': "
                f"Capacity limit of {self.capacity} reached."
            )
        self._occupants.add(occupant_id)

    def remove_occupant(self, occupant_id: str):
        """Removes an occupant from the room."""
        if occupant_id not in self._occupants:
            raise OccupantNotFoundError(
                f"Occupant '{occupant_id}' not found in room '{self.room_id}'"
            )
        self._occupants.remove(occupant_id)

    def contains_point(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Checks if a point (px, py) is strictly inside the room boundaries."""
        return (
            (self.x + margin) <= px <= (self.x + self.width - margin)
            and (self.y + margin) <= py <= (self.y + self.height - margin)
        )

    def clamp_position(self, px: float, py: float, margin: float = 0.2) -> tuple[float, float]:
        """Clamps a 2D position so the occupant remains strictly inside room walls."""
        clamped_x = max(self.x + margin, min(px, self.x + self.width - margin))
        clamped_y = max(self.y + margin, min(py, self.y + self.height - margin))
        return (clamped_x, clamped_y)

    def center_position(self) -> tuple[float, float]:
        """Returns the geometric center of the room."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def record_history(self, timestamp: str):
        """Appends the current ground-truth occupancy count to history."""
        self.occupancy_history.append((timestamp, self.occupant_count))

    def reset(self):
        """Resets occupants and history to clean initial state."""
        self._occupants.clear()
        self.occupancy_history.clear()

    def __repr__(self) -> str:
        return f"<Room '{self.room_id}' ({self.name}): {self.occupant_count}/{self.capacity} occupants>"

