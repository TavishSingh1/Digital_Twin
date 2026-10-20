"""
Ground Truth Occupancy Recorder & Exporter
==========================================
Maintains high-fidelity, structured ground-truth occupancy logs at every 5-minute
simulation timestep, and supports exporting to standard CSV formats.
"""

import os
from dataclasses import dataclass, asdict
import pandas as pd
from simulation.building import Building
from simulation.config import EXPECTED_ROOM_IDS


@dataclass
class GroundTruthRecord:
    timestamp: str
    room_id: str
    actual_occupants: int
    room_capacity: int
    occupancy_fraction: float
    arrivals: int
    departures: int
    incoming_transitions: int
    outgoing_transitions: int


class GroundTruthRecorder:
    def __init__(self):
        self.records: list[GroundTruthRecord] = []

    def record_step(
        self,
        timestamp_str: str,
        building: Building,
        room_events: dict[str, dict[str, int]],
    ):
        """
        Records the exact ground truth for each of the 5 rooms for the current timestep.
        room_events mapping: room_id -> {'arrivals': x, 'departures': y, 'incoming': z, 'outgoing': w}
        """
        for room_id in EXPECTED_ROOM_IDS:
            room = building.get_room(room_id)
            count = room.occupant_count
            cap = room.capacity
            frac = round(count / cap, 4) if cap > 0 else 0.0

            events = room_events.get(room_id, {
                "arrivals": 0,
                "departures": 0,
                "incoming": 0,
                "outgoing": 0,
            })

            rec = GroundTruthRecord(
                timestamp=timestamp_str,
                room_id=room_id,
                actual_occupants=count,
                room_capacity=cap,
                occupancy_fraction=frac,
                arrivals=events["arrivals"],
                departures=events["departures"],
                incoming_transitions=events["incoming"],
                outgoing_transitions=events["outgoing"],
            )
            self.records.append(rec)
            room.record_history(timestamp_str)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts accumulated ground-truth records into a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=[
                "timestamp", "room_id", "actual_occupants", "room_capacity",
                "occupancy_fraction", "arrivals", "departures",
                "incoming_transitions", "outgoing_transitions"
            ])
        data = [asdict(r) for r in self.records]
        df = pd.DataFrame(data)
        return df

    def export_csv(self, filepath: str):
        """Exports ground-truth history to a CSV file."""
        df = self.to_dataframe()
        os_dir = os.path.dirname(filepath)
        if os_dir:
            os.makedirs(os_dir, exist_ok=True)
        df.to_csv(filepath, index=False)

    def clear(self):
        """Clears all logged records."""
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)
