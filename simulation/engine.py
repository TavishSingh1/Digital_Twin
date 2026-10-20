"""
Simulation Engine & Logical Clock
=================================
Drives the 5-minute discrete-time building simulation, coordinates occupant
decision making, movement, capacity enforcement, and ground-truth logging.
"""

from datetime import datetime, timedelta, timezone
import random
import time as wall_time
from typing import Optional, Any
import pandas as pd

from simulation.config import BuildingConfig, load_building_config, EXPECTED_ROOM_IDS
from simulation.building import Building
from simulation.occupant import Occupant
from simulation.schedules import generate_schedules_for_population, OccupantSchedule, InteractiveSchedule
from simulation.ground_truth import GroundTruthRecorder
from simulation.sensors import VirtualSensorManager, load_sensor_config
from simulation.energy import EnergyManager, load_energy_config


class SimulationEngine:
    def __init__(
        self,
        config: Optional[BuildingConfig] = None,
        num_occupants: int = 50,
        seed: int = 42,
        start_time_str: Optional[str] = None,
        sensor_config: Optional[dict[str, Any]] = None,
        energy_config: Optional[dict[str, Any]] = None,
        person_counter: int = 1,
    ):
        self.config = config or load_building_config()
        self.seed = seed
        self.num_occupants = num_occupants

        # Isolated RNG for simulation reproducibility
        self.rng = random.Random(self.seed)

        # Initialize physical building
        self.building = Building(self.config)

        # Simulation logical clock
        start_iso = start_time_str or self.config.default_start_time
        # Normalize ISO string to datetime
        self.initial_time = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        self.current_time = self.initial_time
        self.step_delta = timedelta(minutes=self.config.cadence_minutes)
        self.step_count = 0

        # State tracking
        self.is_paused = False
        self.occupants: dict[str, Occupant] = {}
        self.recorder = GroundTruthRecorder()

        # Interactive entry and lifecycle tracking
        self.entrance_queue: list[Occupant] = []
        self.total_arrivals: int = 0
        self.total_exits: int = 0
        self.person_counter: int = person_counter

        # Initialize virtual environmental sensors
        self.sensor_config = sensor_config or load_sensor_config()
        self.sensor_manager = VirtualSensorManager(config=self.sensor_config, seed=self.seed)

        # Initialize smart HVAC and energy management (Phase 7)
        self.energy_config = energy_config or load_energy_config()
        self.energy_manager = EnergyManager(config=self.energy_config, seed=self.seed)

        # Initialize population and schedules
        self._initialize_population()

        # Record initial t=0 ground-truth state before stepping
        self._record_current_state({r: {"arrivals": 0, "departures": 0, "incoming": 0, "outgoing": 0} for r in EXPECTED_ROOM_IDS})
        # Record initial t=0 sensor readings
        self.sensor_manager.step(self.current_time, self.building)
        # Record initial t=0 energy telemetry (elapsed 0.0 seconds)
        self.energy_manager.step(self.current_time, self.building, self.sensor_manager, elapsed_seconds=0.0)

    @property
    def current_occupants(self) -> int:
        """Returns the number of people currently inside the building."""
        return self.building.total_occupants

    @property
    def waiting_at_entrance(self) -> int:
        """Returns the number of occupants currently waiting in the entrance queue."""
        return len(self.entrance_queue)



    def _initialize_population(self):
        """Generates deterministic schedules and instantiates virtual occupants."""
        schedules = generate_schedules_for_population(
            num_occupants=self.num_occupants,
            seed=self.seed,
        )
        self.occupants.clear()
        for sched in schedules:
            occ = Occupant(
                occupant_id=sched.occupant_id,
                occupant_type=sched.occupant_type,
                schedule=sched,
            )
            self.occupants[sched.occupant_id] = occ

    def _record_current_state(self, room_events: dict[str, dict[str, int]]):
        """Logs ground-truth occupancy state for all rooms."""
        ts_str = self.current_time.isoformat()
        self.recorder.record_step(ts_str, self.building, room_events)

    def step(self) -> dict[str, Any]:
        """
        Advances the simulation by exactly one logical step (5 minutes).
        Execution order:
        1. Determine scheduled goals and transitions.
        2. Progress occupant movement and room transitions.
        3. Reconcile room occupant memberships.
        4. Validate room capacities and population conservation invariants.
        5. Advance the logical clock by +5 minutes.
        6. Record ground-truth occupancy.
        """
        room_events = {
            r: {"arrivals": 0, "departures": 0, "incoming": 0, "outgoing": 0}
            for r in EXPECTED_ROOM_IDS
        }

        # 0. Process entrance queue: admit waiting occupants into Room 1 if capacity allows
        r1 = self.building.get_room("Room 1")
        while self.entrance_queue and r1.can_accept():
            queued_occ = self.entrance_queue.pop(0)
            r1.add_occupant(queued_occ.occupant_id)
            queued_occ.current_room_id = "Room 1"
            door_pos = r1.exterior_door_position or (7.0, 0.0)
            queued_occ.position = r1.clamp_position(door_pos[0], door_pos[1])
            queued_occ.movement_state = "stationary"
            self.total_arrivals += 1
            room_events["Room 1"]["arrivals"] += 1

        # 1. Update behavioral decisions for all occupants
        for occ in self.occupants.values():
            if occ.movement_state != "waiting_at_entrance":
                occ.update_decision(self.current_time, self.building)

        # 2. Execute movement steps
        # To avoid bias from occupant order, shuffle occupant execution order deterministically
        occ_keys = list(self.occupants.keys())
        self.rng.shuffle(occ_keys)

        for occ_id in occ_keys:
            occ = self.occupants[occ_id]
            if occ.movement_state == "waiting_at_entrance":
                continue
            prev_room = occ.current_room_id

            arr, dep, inc, out = occ.step_movement(self.building, self.rng)
            curr_room = occ.current_room_id

            # Attribute transition metrics to appropriate rooms
            if arr > 0 and curr_room:
                room_events[curr_room]["arrivals"] += arr
                self.total_arrivals += arr
            if dep > 0 and prev_room:
                room_events[prev_room]["departures"] += dep
                self.total_exits += dep
            if out > 0 and prev_room and prev_room != curr_room:
                room_events[prev_room]["outgoing"] += 1
            if inc > 0 and curr_room and prev_room != curr_room and arr == 0:
                room_events[curr_room]["incoming"] += 1

        # 3. Validate building-wide invariants
        active_ids = {o.occupant_id for o in self.occupants.values() if o.is_inside_building}
        self.building.validate_invariants(active_ids)

        # 4. Advance logical simulation clock by 5 minutes
        self.current_time += self.step_delta
        self.step_count += 1

        # 5. Record ground truth
        self._record_current_state(room_events)

        # 6. Advance virtual environmental sensors for all 5 rooms (coupled with HVAC state)
        current_hvac = self.energy_manager.get_latest_states()
        sensor_readings = self.sensor_manager.step(self.current_time, self.building, hvac_states=current_hvac)

        # 7. Advance smart HVAC and room-wise energy management (Phase 7)
        energy_states = self.energy_manager.step(
            self.current_time,
            self.building,
            self.sensor_manager,
            elapsed_seconds=self.step_delta.total_seconds(),
        )

        return {
            "step": self.step_count,
            "timestamp": self.current_time.isoformat(),
            "total_occupants": self.building.total_occupants,
            "waiting_at_entrance": len(self.entrance_queue),
            "total_arrivals": self.total_arrivals,
            "total_exits": self.total_exits,
            "room_counts": {r: self.building.get_room(r).occupant_count for r in EXPECTED_ROOM_IDS},
            "sensor_readings": sensor_readings,
            "energy_states": energy_states,
        }

    def add_person(
        self,
        count: int = 1,
        destination: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Interactively adds count persons to the simulation through Room 1 (Main Entrance).
        - If Room 1 has available capacity: admitted immediately into Room 1.
        - If Room 1 is full: queued in entrance_queue outside Room 1.
        Updates immediate telemetry (sound, lights) and ground truth without advancing logical clock.
        """
        if isinstance(count, str):
            destination = count
            count = 1
        if count < 1:
            raise ValueError(f"Number of people to add must be at least 1, got {count}")

        admitted_ids = []
        queued_ids = []
        r1 = self.building.get_room("Room 1")

        for _ in range(count):
            pid = f"P{self.person_counter:04d}"
            self.person_counter += 1

            sched = InteractiveSchedule(
                occupant_id=pid,
                occupant_type="visitor",
                destination=destination,
            )
            occ = Occupant(
                occupant_id=pid,
                occupant_type="visitor",
                schedule=sched,
            )
            self.occupants[pid] = occ

            if r1.can_accept():
                r1.add_occupant(pid)
                occ.current_room_id = "Room 1"
                door_pos = r1.exterior_door_position or (7.0, 0.0)
                occ.position = r1.clamp_position(door_pos[0], door_pos[1])
                occ.movement_state = "stationary"
                self.total_arrivals += 1
                admitted_ids.append(pid)
            else:
                occ.current_room_id = None
                occ.movement_state = "waiting_at_entrance"
                occ.position = (7.0, -1.5)  # Outside entrance door
                self.entrance_queue.append(occ)
                queued_ids.append(pid)

        # Refresh immediate sensor readings (sound levels and lights respond immediately)
        self.sensor_manager.refresh_immediate_sensors(self.current_time, self.building)
        self.energy_manager.refresh_immediate(self.current_time, self.building, self.sensor_manager)

        # Update latest ground truth record for current timestamp
        room_events = {r: {"arrivals": 0, "departures": 0, "incoming": 0, "outgoing": 0} for r in EXPECTED_ROOM_IDS}
        room_events["Room 1"]["arrivals"] = len(admitted_ids)
        self._record_current_state(room_events)

        return {
            "requested": count,
            "admitted_count": len(admitted_ids),
            "queued_count": len(queued_ids),
            "admitted_ids": admitted_ids,
            "queued_ids": queued_ids,
            "total_building_occupants": self.building.total_occupants,
            "waiting_at_entrance": len(self.entrance_queue),
            "total_arrivals": self.total_arrivals,
            "total_exits": self.total_exits,
        }

    def exit_person(
        self,
        count: int = 1,
        room_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Interactively exits count occupants from the building.
        Picks occupants from room_id (or any room if room_id is None).
        Can also take an occupant ID string directly as the first argument.
        """
        if isinstance(count, str):
            if count in self.occupants:
                target_occ = self.occupants[count]
                curr_r = target_occ.current_room_id
                if curr_r:
                    room = self.building.get_room(curr_r)
                    room.remove_occupant(count)
                target_occ.current_room_id = None
                target_occ.movement_state = "exited"
                target_occ.position = (0.0, 0.0)
                target_occ.intended_destination = None
                target_occ.planned_route.clear()
                if hasattr(target_occ, "schedule") and hasattr(target_occ.schedule, "is_exited"):
                    target_occ.schedule.is_exited = True
                self.total_exits += 1
                self.sensor_manager.refresh_immediate_sensors(self.current_time, self.building)
                self.energy_manager.refresh_immediate(self.current_time, self.building, self.sensor_manager)
                room_events = {r: {"arrivals": 0, "departures": 0, "incoming": 0, "outgoing": 0} for r in EXPECTED_ROOM_IDS}
                if curr_r and curr_r in room_events:
                    room_events[curr_r]["departures"] = 1
                self._record_current_state(room_events)
                return {"exited_ids": [count], "total_building_occupants": self.building.total_occupants}
            room_id = count
            count = 1


        if count < 1:
            raise ValueError(f"Number of people to exit must be at least 1, got {count}")

        exited_ids = []
        r1 = self.building.get_room("Room 1")

        target_rooms = [room_id] if room_id else ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"]

        for r_name in target_rooms:
            room = self.building.get_room(r_name)
            while room.occupants and len(exited_ids) < count:
                occ_id = next(iter(room.occupants))
                room.remove_occupant(occ_id)
                occ = self.occupants[occ_id]
                occ.current_room_id = None
                occ.movement_state = "exited"
                occ.position = (0.0, 0.0)
                if hasattr(occ, "schedule") and hasattr(occ.schedule, "is_exited"):
                    occ.schedule.is_exited = True
                self.total_exits += 1
                exited_ids.append(occ_id)
            if len(exited_ids) >= count:
                break

        # Refresh immediate sensor readings and ground truth
        self.sensor_manager.refresh_immediate_sensors(self.current_time, self.building)
        self.energy_manager.refresh_immediate(self.current_time, self.building, self.sensor_manager)
        room_events = {r: {"arrivals": 0, "departures": 0, "incoming": 0, "outgoing": 0} for r in EXPECTED_ROOM_IDS}
        room_events["Room 1"]["departures"] = len(exited_ids)
        self._record_current_state(room_events)

        return {
            "requested": count,
            "exited_count": len(exited_ids),
            "exited_ids": exited_ids,
            "total_building_occupants": self.building.total_occupants,
            "waiting_at_entrance": len(self.entrance_queue),
            "total_arrivals": self.total_arrivals,
            "total_exits": self.total_exits,
        }

    def move_people(
        self,
        source_room: str,
        destination_room: str,
        person_ids: Optional[list[str]] = None,
        count: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Manually moves occupants from source_room to destination_room.
        Validates topology, direct connectivity, occupant presence, and destination capacity.
        Updates spatial positioning, immediate sensor telemetry, and ground truth.
        """
        source_room = source_room.replace("_", " ")
        destination_room = destination_room.replace("_", " ")

        if isinstance(person_ids, int):
            count = person_ids
            person_ids = None

        # 1. Validate room existence
        if source_room not in self.building.rooms:
            raise ValueError(f"Source room '{source_room}' does not exist in building.")
        if destination_room not in self.building.rooms:
            raise ValueError(f"Destination room '{destination_room}' does not exist in building.")

        # 2. Validate distinct rooms
        if source_room == destination_room:
            raise ValueError(f"Source and destination rooms cannot be the same ('{source_room}').")


        # 3. Validate direct topological connectivity via doors
        if destination_room not in self.building.adjacency[source_room]:
            path = self.building.find_path(source_room, destination_room)
            if path and len(path) > 2:
                intermediate = path[1]
                raise ValueError(
                    f"Rooms '{source_room}' and '{destination_room}' are not directly connected. "
                    f"Direct passage requires an adjoining doorway; an intermediate room (such as '{intermediate}') is required."
                )
            else:
                raise ValueError(
                    f"Rooms '{source_room}' and '{destination_room}' are not connected in the building topology."
                )

        src = self.building.get_room(source_room)
        dst = self.building.get_room(destination_room)

        # 4. Resolve and validate target occupants
        if person_ids is not None:
            if not person_ids:
                raise ValueError("No person IDs specified for movement.")

            for pid in person_ids:
                if pid not in self.occupants:
                    raise ValueError(f"Occupant ID '{pid}' does not exist in the simulation.")
                occ = self.occupants[pid]
                if occ.current_room_id != source_room or pid not in src.occupants:
                    loc = occ.current_room_id or "Outside"
                    raise ValueError(
                        f"Occupant '{pid}' is not currently in source room '{source_room}' (currently in: '{loc}')."
                    )
            target_ids = list(person_ids)
        else:
            if count is None or count < 1:
                raise ValueError(f"Movement count must be at least 1, got {count}")
            if src.occupant_count == 0:
                raise ValueError(f"Source room '{source_room}' currently has no occupants to move.")
            if count > src.occupant_count:
                raise ValueError(
                    f"Cannot move {count} people: '{source_room}' only contains {src.occupant_count} occupants."
                )
            target_ids = sorted(list(src.occupants))[:count]

        # 5. Validate destination capacity
        available_cap = dst.capacity - dst.occupant_count
        if available_cap <= 0:
            raise ValueError(
                f"Destination room '{destination_room}' is at full capacity "
                f"({dst.occupant_count}/{dst.capacity} occupants). Cannot move any occupants."
            )
        if len(target_ids) > available_cap:
            raise ValueError(
                f"Cannot move {len(target_ids)} occupants into '{destination_room}': "
                f"only {available_cap} spot(s) available ({dst.occupant_count}/{dst.capacity} capacity)."
            )

        # 6. Perform the atomic movement
        door_pos = self.building.door_positions.get(
            (source_room, destination_room),
            dst.center_position(),
        )

        for pid in target_ids:
            occ = self.occupants[pid]
            src.remove_occupant(pid)
            dst.add_occupant(pid)
            occ.set_manual_destination(destination_room)
            occ.position = dst.clamp_position(door_pos[0], door_pos[1])

        # 7. Refresh immediate sensor telemetry (sound pressure level & lighting)
        self.sensor_manager.refresh_immediate_sensors(self.current_time, self.building)
        self.energy_manager.refresh_immediate(self.current_time, self.building, self.sensor_manager)

        # 8. Record movement transitions in ground truth
        room_events = {r: {"arrivals": 0, "departures": 0, "incoming": 0, "outgoing": 0} for r in EXPECTED_ROOM_IDS}
        room_events[source_room]["outgoing"] = len(target_ids)
        room_events[destination_room]["incoming"] = len(target_ids)
        self._record_current_state(room_events)

        return {
            "success": True,
            "moved_count": len(target_ids),
            "moved_ids": target_ids,
            "source_room": source_room,
            "destination_room": destination_room,
            "source_occupancy": src.occupant_count,
            "destination_occupancy": dst.occupant_count,
            "message": (
                f"Successfully moved {len(target_ids)} occupant(s) ({', '.join(target_ids)}) "
                f"from {source_room} ({src.occupant_count}/{src.capacity}) to "
                f"{destination_room} ({dst.occupant_count}/{dst.capacity})."
            ),
        }

    def run(self, num_steps: int) -> pd.DataFrame:
        """Runs the simulation for a specified number of 5-minute steps."""
        for _ in range(num_steps):
            if self.is_paused:
                break
            self.step()
        return self.recorder.to_dataframe()

    def pause(self):
        """Pauses multi-step simulation execution."""
        self.is_paused = True

    def resume(self):
        """Resumes multi-step simulation execution."""
        self.is_paused = False

    def reset(self, new_seed: Optional[int] = None, keep_session_counter: bool = True):
        """Restores the simulation to its initial state."""
        if new_seed is not None:
            self.seed = new_seed
        self.rng = random.Random(self.seed)
        self.current_time = self.initial_time
        self.step_count = 0
        self.is_paused = False

        self.entrance_queue.clear()
        self.total_arrivals = 0
        self.total_exits = 0
        if not keep_session_counter:
            self.person_counter = 1

        self.building.reset()
        self.recorder.clear()
        self.sensor_manager.reset(self.seed)
        self.energy_manager.reset(self.seed)
        self._initialize_population()
        self._record_current_state({r: {"arrivals": 0, "departures": 0, "incoming": 0, "outgoing": 0} for r in EXPECTED_ROOM_IDS})
        self.sensor_manager.step(self.current_time, self.building)
        self.energy_manager.step(self.current_time, self.building, self.sensor_manager, elapsed_seconds=0.0)

    def export_ground_truth(self, filepath: str = "artifacts/ground_truth_occupancy.csv"):
        """Exports the recorded ground truth to CSV."""
        self.recorder.export_csv(filepath)

    def export_virtual_sensors(self, filepath: str = "artifacts/virtual_sensor_readings.csv"):
        """Exports recorded virtual environmental sensor readings to CSV."""
        self.sensor_manager.export_csv(filepath)

    def get_sensor_readings_df(self) -> pd.DataFrame:
        """Returns recorded virtual environmental sensor readings as a DataFrame."""
        return self.sensor_manager.to_dataframe()

    def set_room_lighting(self, room_id: str, state: Optional[bool]):
        """Sets manual lighting override for a room and refreshes immediate telemetry."""
        self.sensor_manager.set_manual_lighting(room_id, state)
        # Synchronize with energy manager
        light_mode = "FORCE_ON" if state is True else ("FORCE_OFF" if state is False else "AUTO")
        self.energy_manager.set_lighting_override(room_id, light_mode)
        self.sensor_manager.refresh_immediate_sensors(self.current_time, self.building)
        self.energy_manager.refresh_immediate(self.current_time, self.building, self.sensor_manager)

    def set_room_hvac_mode(self, room_id: str, mode: Optional[str]):
        """Sets manual HVAC override ('AUTO', 'ECO', 'FORCE_OFF') for a room."""
        self.energy_manager.set_hvac_mode_override(room_id, mode)
        self.energy_manager.refresh_immediate(self.current_time, self.building, self.sensor_manager)

    def set_room_hvac_setpoint(self, room_id: str, setpoint: float):
        """Sets custom target temperature setpoint for a room."""
        self.energy_manager.set_hvac_setpoint(room_id, setpoint)
        norm_r = room_id.replace("_", " ")
        if norm_r in self.sensor_manager.room_models:
            self.sensor_manager.room_models[norm_r].temp_setpoint = float(setpoint)
        self.energy_manager.refresh_immediate(self.current_time, self.building, self.sensor_manager)

    def get_energy_readings_df(self) -> pd.DataFrame:
        """Returns recorded room-wise energy history as a DataFrame."""
        return self.energy_manager.to_dataframe()

    def export_energy_readings(self, filepath: str = "artifacts/room_energy_consumption.csv"):
        """Exports recorded room-wise energy history to CSV."""
        self.energy_manager.export_csv(filepath)


