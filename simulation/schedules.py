"""
Occupant Behavioral Schedules & Activity Decisions
===================================================
Provides deterministic daily schedules for Regular, Visitor, and Staff occupants.
Schedules specify arrival/departure times, preferred rooms, and activity transitions.

NOTE: These schedules are heuristic simulation assumptions designed for
reproducible digital twin demonstration; they are not calibrated against
real-world indoor trajectory observations.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
import random
from typing import Optional


@dataclass
class ScheduleItem:
    """Represents a scheduled activity and location at a specific time of day."""
    start_time: time
    end_time: time
    target_room_id: Optional[str]  # None indicates exterior (outside building)
    activity: str


class OccupantSchedule:
    """Encapsulates an individual occupant's daily timeline."""
    def __init__(
        self,
        occupant_id: str,
        occupant_type: str,
        arrival_time: time,
        departure_time: time,
        items: list[ScheduleItem],
    ):
        self.occupant_id = occupant_id
        self.occupant_type = occupant_type
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.items = sorted(items, key=lambda x: x.start_time)
        self.is_exited = False
        self.manual_destination_override: Optional[str] = None

    def set_manual_override(self, destination: str):
        """Overrides scheduled destination with a manual operator room assignment."""
        self.manual_destination_override = destination

    def clear_manual_override(self):
        """Clears manual operator room assignment, returning to scheduled routine."""
        self.manual_destination_override = None

    def get_desired_target(self, current_dt: datetime) -> tuple[Optional[str], str]:
        """
        Determines the desired room and activity for the given simulation timestamp.
        Returns:
            (desired_room_id, activity_name)
            desired_room_id is None if the occupant should be outside the building.
        """
        if self.is_exited:
            return None, "off_site"

        # Explicit manual operator assignment takes precedence over default timeline
        if self.manual_destination_override is not None:
            return self.manual_destination_override, "manual_assignment"

        curr_time = current_dt.time()

        # Before arrival or after departure -> outside
        if curr_time < self.arrival_time or curr_time >= self.departure_time:
            return None, "off_site"

        # Check explicit scheduled items
        for item in self.items:
            if item.start_time <= curr_time < item.end_time:
                return item.target_room_id, item.activity

        # Default fallback during work hours
        return self.items[0].target_room_id if self.items else None, "idle"


def generate_schedules_for_population(
    num_occupants: int,
    seed: int = 42,
    ratio_regular: float = 0.75,
    ratio_visitor: float = 0.15,
    ratio_staff: float = 0.10,
) -> list[OccupantSchedule]:
    """
    Deterministically generates behavioral schedules for a population using
    an isolated random number generator seeded with `seed`.
    """
    rng = random.Random(seed)
    schedules = []

    num_reg = int(num_occupants * ratio_regular)
    num_vis = int(num_occupants * ratio_visitor)
    num_staff = num_occupants - (num_reg + num_vis)

    occ_counter = 1

    # 1. Regular Occupants (Desks in Room 2 or 4, meetings in Room 1, lunch in Room 5)
    for _ in range(num_reg):
        occ_id = f"occ_reg_{occ_counter:03d}"
        occ_counter += 1

        primary_room = rng.choice(["Room 2", "Room 4"])
        arr_hour = rng.randint(8, 9)
        arr_minute = rng.choice([0, 15, 30, 45])
        dep_hour = rng.randint(17, 18)
        dep_minute = rng.choice([0, 15, 30])

        arr_t = time(arr_hour, arr_minute)
        dep_t = time(dep_hour, dep_minute)

        items = []
        # Morning work
        lunch_start_hour = rng.choice([12, 13])
        lunch_t = time(lunch_start_hour, 0)
        lunch_end_t = time(lunch_start_hour, 45)

        items.append(ScheduleItem(
            start_time=arr_t,
            end_time=lunch_t,
            target_room_id=primary_room,
            activity="working",
        ))

        # Lunch in Cafeteria (Room 5)
        items.append(ScheduleItem(
            start_time=lunch_t,
            end_time=lunch_end_t,
            target_room_id="Room 5",
            activity="lunch",
        ))

        # Afternoon meeting in Conference (Room 1) or Suite (Room 3)
        has_afternoon_meeting = rng.random() < 0.60
        if has_afternoon_meeting:
            meet_start = time(14, 0)
            meet_end = time(15, 0)
            meeting_room = rng.choice(["Room 1", "Room 3"])

            items.append(ScheduleItem(
                start_time=lunch_end_t,
                end_time=meet_start,
                target_room_id=primary_room,
                activity="working",
            ))
            items.append(ScheduleItem(
                start_time=meet_start,
                end_time=meet_end,
                target_room_id=meeting_room,
                activity="meeting",
            ))
            items.append(ScheduleItem(
                start_time=meet_end,
                end_time=dep_t,
                target_room_id=primary_room,
                activity="working",
            ))
        else:
            items.append(ScheduleItem(
                start_time=lunch_end_t,
                end_time=dep_t,
                target_room_id=primary_room,
                activity="working",
            ))

        schedules.append(OccupantSchedule(
            occupant_id=occ_id,
            occupant_type="regular",
            arrival_time=arr_t,
            departure_time=dep_t,
            items=items,
        ))

    # 2. Visitors (Short visits for meetings in Room 1 or Executive Suite Room 3)
    for _ in range(num_vis):
        occ_id = f"occ_vis_{occ_counter:03d}"
        occ_counter += 1

        meeting_room = rng.choice(["Room 1", "Room 3"])
        visit_hour = rng.choice([10, 11, 14, 15])
        visit_minute = rng.choice([0, 30])
        duration_minutes = rng.choice([45, 60, 90])

        arr_t = time(visit_hour, visit_minute)
        end_minute = (visit_minute + duration_minutes) % 60
        end_hour = visit_hour + (visit_minute + duration_minutes) // 60
        dep_t = time(end_hour, end_minute)

        items = [
            ScheduleItem(
                start_time=arr_t,
                end_time=dep_t,
                target_room_id=meeting_room,
                activity="meeting",
            )
        ]

        schedules.append(OccupantSchedule(
            occupant_id=occ_id,
            occupant_type="visitor",
            arrival_time=arr_t,
            departure_time=dep_t,
            items=items,
        ))

    # 3. Facilities & Service Staff (Move between rooms throughout the day)
    for _ in range(num_staff):
        occ_id = f"occ_staff_{occ_counter:03d}"
        occ_counter += 1

        arr_t = time(7, 30)
        dep_t = time(16, 30)

        # Staff rotation across all rooms
        items = [
            ScheduleItem(time(7, 30), time(9, 30), "Room 1", "inspection"),
            ScheduleItem(time(9, 30), time(11, 30), "Room 2", "maintenance"),
            ScheduleItem(time(11, 30), time(12, 30), "Room 5", "break"),
            ScheduleItem(time(12, 30), time(14, 30), "Room 4", "maintenance"),
            ScheduleItem(time(14, 30), time(16, 30), "Room 3", "cleaning"),
        ]

        schedules.append(OccupantSchedule(
            occupant_id=occ_id,
            occupant_type="staff",
            arrival_time=arr_t,
            departure_time=dep_t,
            items=items,
        ))

    return schedules


class InteractiveSchedule:
    """
    Flexible behavioral schedule for interactively added occupants.
    Can be configured with an immediate destination or dynamic presence in the building.
    """
    def __init__(
        self,
        occupant_id: str,
        occupant_type: str = "visitor",
        destination: Optional[str] = None,
    ):
        self.occupant_id = occupant_id
        self.occupant_type = occupant_type
        self.destination = destination
        self.arrival_time = time(0, 0)
        self.departure_time = time(23, 59)
        self.items = []
        self.is_exited = False

    def set_manual_override(self, destination: str):
        """Overrides destination with a manual operator room assignment."""
        self.destination = destination

    def clear_manual_override(self):
        """Clears destination override."""
        self.destination = None

    def get_desired_target(self, current_dt: datetime) -> tuple[Optional[str], str]:
        if self.is_exited:
            return None, "off_site"
        if self.destination:
            return self.destination, "visiting"
        return "Room 1", "visiting"


