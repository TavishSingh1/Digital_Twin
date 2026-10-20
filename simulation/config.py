"""
Simulation Configuration Loader & Validator
==========================================
Loads and strictly validates the building layout, capacities, connections,
and parameters against model specifications.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

EXPECTED_ROOM_IDS = ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"]


class ConfigurationError(ValueError):
    """Raised when simulation or building configuration violates constraints."""
    pass


class InvalidRoomError(ConfigurationError):
    """Raised when an unrecognized or invalid room ID is encountered."""
    pass


@dataclass
class RoomConfig:
    id: str
    name: str
    width: float
    height: float
    x: float
    y: float
    capacity: int
    has_exterior_exit: bool = False
    exterior_door_position: Optional[tuple[float, float]] = None

    def validate(self):
        if self.id not in EXPECTED_ROOM_IDS:
            raise InvalidRoomError(
                f"Invalid room ID '{self.id}'. Expected one of {EXPECTED_ROOM_IDS}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(
                f"Room '{self.id}' dimensions must be positive, got width={self.width}, height={self.height}"
            )
        if self.capacity <= 0:
            raise ConfigurationError(
                f"Room '{self.id}' capacity must be at least 1, got {self.capacity}"
            )
        if self.has_exterior_exit and self.exterior_door_position is not None:
            dx, dy = self.exterior_door_position
            if not (self.x <= dx <= self.x + self.width and self.y <= dy <= self.y + self.height):
                raise ConfigurationError(
                    f"Room '{self.id}' exterior door at ({dx}, {dy}) is outside room bounds "
                    f"[{self.x}, {self.x+self.width}] x [{self.y}, {self.y+self.height}]"
                )


@dataclass
class ConnectionConfig:
    room_a: str
    room_b: str
    door_position: tuple[float, float]
    door_width: float = 1.0

    def validate(self, valid_room_ids: set[str]):
        if self.room_a not in valid_room_ids:
            raise InvalidRoomError(f"Connection references unknown room_a '{self.room_a}'")
        if self.room_b not in valid_room_ids:
            raise InvalidRoomError(f"Connection references unknown room_b '{self.room_b}'")
        if self.room_a == self.room_b:
            raise ConfigurationError(f"Connection cannot link room '{self.room_a}' to itself")
        if self.door_width <= 0:
            raise ConfigurationError(f"Door width must be positive, got {self.door_width}")


@dataclass
class BuildingConfig:
    building_name: str
    cadence_minutes: int
    default_start_time: str
    default_seed: int
    rooms: list[RoomConfig] = field(default_factory=list)
    connections: list[ConnectionConfig] = field(default_factory=list)
    simulation_assumptions: dict[str, str] = field(default_factory=dict)

    def validate(self):
        if len(self.rooms) != len(EXPECTED_ROOM_IDS):
            raise ConfigurationError(
                f"Building must contain exactly {len(EXPECTED_ROOM_IDS)} rooms, found {len(self.rooms)}"
            )
        
        configured_ids = [r.id for r in self.rooms]
        if configured_ids != EXPECTED_ROOM_IDS:
            raise ConfigurationError(
                f"Room ordering must strictly match model specification {EXPECTED_ROOM_IDS}, got {configured_ids}"
            )

        for room in self.rooms:
            room.validate()

        valid_ids = set(configured_ids)
        for conn in self.connections:
            conn.validate(valid_ids)

        if not any(r.has_exterior_exit for r in self.rooms):
            raise ConfigurationError(
                "At least one room must have an exterior exit for occupants to enter/leave the building"
            )

        if self.cadence_minutes != 5:
            raise ConfigurationError(
                f"Cadence must be exactly 5 minutes to match STGNN time series model, got {self.cadence_minutes}"
            )


def load_building_config(filepath: str = "config/building.json") -> BuildingConfig:
    """Loads and validates building configuration from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    rooms = []
    for r in data.get("rooms", []):
        ext_pos = tuple(r["exterior_door_position"]) if "exterior_door_position" in r else None
        rooms.append(RoomConfig(
            id=r["id"],
            name=r.get("name", r["id"]),
            width=float(r["width"]),
            height=float(r["height"]),
            x=float(r["x"]),
            y=float(r["y"]),
            capacity=int(r["capacity"]),
            has_exterior_exit=bool(r.get("has_exterior_exit", False)),
            exterior_door_position=ext_pos,
        ))

    connections = []
    for c in data.get("connections", []):
        connections.append(ConnectionConfig(
            room_a=c["room_a"],
            room_b=c["room_b"],
            door_position=tuple(c["door_position"]),
            door_width=float(c.get("door_width", 1.0)),
        ))

    config = BuildingConfig(
        building_name=data.get("building_name", "Virtual Building"),
        cadence_minutes=int(data.get("cadence_minutes", 5)),
        default_start_time=data.get("default_start_time", "2021-09-06T08:00:00Z"),
        default_seed=int(data.get("default_seed", 42)),
        rooms=rooms,
        connections=connections,
        simulation_assumptions=data.get("simulation_assumptions", {}),
    )

    config.validate()
    return config

