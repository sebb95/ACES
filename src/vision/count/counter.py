from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CountConfig:
    """
    Configuration for line-crossing counting.

    Attributes:
        line_position: Position of the counting line in pixels.
            - if axis="y", this means horizontal line at y=line_position
            - if axis="x", this means vertical line at x=line_position
        axis: Axis used for crossing detection:
            - "y" for top/bottom movement
            - "x" for left/right movement
        line_margin: Half-width of neutral band around line
        min_positions: Minimum observations before counting
        max_missing_frames: Remove stale tracks after this many missing frames
        direction: Allowed crossing direction:
            - "positive": increasing coordinate
            - "negative": decreasing coordinate
            - "any": both directions
    """
    line_position: float
    axis: str = "y"
    line_margin: float = 20.0
    min_positions: int = 2
    max_missing_frames: int = 30
    direction: str = "positive"


@dataclass
class TrackState:
    """
    Internal state for one tracked fish.
    """
    track_id: int
    centers: List[List[float]] = field(default_factory=list)
    zone_history: List[str] = field(default_factory=list)
    counted: bool = False #each fish counted once
    last_frame_seen: int = -1
    class_id: Optional[int] = None


class LineCounter:
    """
    Count fish by detecting line crossings from tracker output.

    Expected input per frame:
        tracked_objects = [
            {
                "track_id": int | None,
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "center": [cx, cy],
            },
            ...
        ]

    Strategy:
    - keep center history for each track_id
    - convert each center to a zone relative to the horizontal line:
        "before" / "middle" / "after"
    - count only when a track moves across the full pattern:
        before -> middle -> after   (down)
        after -> middle -> before   (up)
    - count each track_id only once
    """

    def __init__(self, config: CountConfig) -> None:
        self.config = config
        self.tracks: Dict[int, TrackState] = {}
        self.total_count: int = 0
        self.counted_track_ids: List[int] = []

    def update(self, tracked_objects: List[dict], frame_index: int) -> int:
        """
        Update counter with tracker output from a single frame.

        Args:
            tracked_objects: List of tracked objects from FishTracker.update()["tracked_objects"]
            frame_index: Sequential frame number

        Returns:
            Number of newly counted fish in this frame
        """
        new_counts = 0

        for obj in tracked_objects:
            track_id = obj.get("track_id")
            center = obj.get("center")
            class_id = obj.get("class_id")

            if track_id is None:
                continue
            if center is None or len(center) != 2:
                continue

            if self.config.axis == "y":
                position_value = center[1]
            elif self.config.axis == "x":
                position_value = center[0]
            else:
                raise ValueError(f"Unsupported axis: {self.config.axis}")

            zone = self._get_zone(position_value)

            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(track_id=track_id)
                self.tracks[track_id] = state

            state.centers.append(center)
            state.zone_history.append(zone)
            state.last_frame_seen = frame_index
            state.class_id = class_id

            if not state.counted and self._should_count(state):
                state.counted = True
                self.total_count += 1
                self.counted_track_ids.append(track_id)
                new_counts += 1

        self._cleanup_old_tracks(frame_index)

        return new_counts

    def get_total_count(self) -> int:
        return self.total_count

    def get_counted_track_ids(self) -> List[int]:
        return self.counted_track_ids.copy()

    def get_track_states(self) -> Dict[int, TrackState]:
        return self.tracks

    def reset(self) -> None:
        """
        Reset counter state for a new independent sequence.
        """
        self.tracks.clear()
        self.total_count = 0
        self.counted_track_ids.clear()

    def _get_zone(self, position_value: float) -> str:
        """
        Convert x or y position into a zone relative to the counting line.
        """
        line_position = self.config.line_position
        margin = self.config.line_margin

        if position_value < line_position - margin:
            return "before"
        if position_value > line_position + margin:
            return "after"
        return "middle"

    def _compress_zones(self, zones: List[str]) -> List[str]:
        """
        Remove consecutive duplicates.

        Example:
            ["before", "before", "middle", "middle", "after"]
        becomes:
            ["before", "middle", "after"]
        """
        if not zones:
            return []

        compressed = [zones[0]]
        for zone in zones[1:]:
            if zone != compressed[-1]:
                compressed.append(zone)

        return compressed

    def _should_count(self, state: TrackState) -> bool:
        """
        Decide whether this track has completed a valid crossing.
        """
        if len(state.centers) < self.config.min_positions:
            return False

        zones = self._compress_zones(state.zone_history)

        if self.config.direction == "positive":
            return self._contains_pattern(zones, ["before", "middle", "after"])

        if self.config.direction == "negative":
            return self._contains_pattern(zones, ["after", "middle", "before"])

        if self.config.direction == "any":
            return (
                self._contains_pattern(zones, ["before", "middle", "after"])
                or self._contains_pattern(zones, ["after", "middle", "before"])
            )

        raise ValueError(f"Unsupported direction: {self.config.direction}")

    @staticmethod
    def _contains_pattern(sequence: List[str], pattern: List[str]) -> bool:
        """
        Check whether pattern appears contiguously inside sequence.
        """
        if len(sequence) < len(pattern):
            return False

        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                return True

        return False

    def _cleanup_old_tracks(self, current_frame: int) -> None:
        """
        Remove tracks that have disappeared for too long.
        """
        to_remove = []

        for track_id, state in self.tracks.items():
            if current_frame - state.last_frame_seen > self.config.max_missing_frames:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]