from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CountConfig:
    """
    Konfigurasjon for linjebasert telling.

    Attributes:
        line_position: Posisjon til tellelinjen i piksler.
            Ved axis="y" brukes horisontal linje.
            Ved axis="x" brukes vertikal linje.
        axis: Aksen som brukes for kryssingsdeteksjon.
            "y" for bevegelse opp/ned.
            "x" for bevegelse venstre/høyre.
        line_margin: Halv bredde på nøytral sone rundt tellelinjen.
        min_positions: Minimum antall observasjoner før et objekt kan telles.
        max_missing_frames: Antall frames før tapte tracks fjernes.
        direction: Tillatt kryssingsretning:
            "positive", "negative" eller "any".
    """
    line_position: float = 600.0
    axis: str = "x"
    line_margin: float = 80.0
    min_positions: int = 2
    max_missing_frames: int = 30
    direction: str = "positive"


@dataclass
class TrackState:
    """
    Intern tilstand for ett sporet objekt.

    Lagrer posisjonshistorikk, sonehistorikk og informasjon om objektet
    allerede er telt. Brukes for å sikre at hver track_id kun telles én gang.
    """
    track_id: int
    centers: List[List[float]] = field(default_factory=list)
    zone_history: List[str] = field(default_factory=list)
    counted: bool = False #each fish counted once
    last_frame_seen: int = -1
    class_id: Optional[int] = None


class LineCounter:
    """
    Teller fisk ved å analysere linjekryssing basert på tracker-output.

    Klassen mottar tracked objects fra FishTracker og følger hvert objekt
    over tid basert på track_id. For hvert objekt lagres senterpunkter og
    posisjon relativt til en virtuell tellelinje.

    Tellelogikk:
    - hvert objekt plasseres i sonene "before", "middle" eller "after"
    - et objekt telles når det følger et gyldig kryssingsmønster
    - hvert track_id telles kun én gang
    - gamle tracks fjernes automatisk etter et definert antall frames
    """

    def __init__(self, config: CountConfig) -> None:
        self.config = config
        self.tracks: Dict[int, TrackState] = {}
        self.total_count: int = 0
        self.counted_track_ids: List[int] = []

    def update(self, tracked_objects: List[dict], frame_index: int) -> int:
        """
        Oppdaterer telleren med tracked objects fra én frame.

        Args:
            tracked_objects: Liste med objekter fra FishTracker.
            frame_index: Sekvensielt frame-nummer.

        Returns:
            Antall nye fisk som ble telt i denne framen.
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

            #debug printout
            print(
                f"[COUNTER DEBUG] track={track_id} "
                f"pos={position_value:.1f} "
                f"zone={zone} "
                f"class={class_id}"
            )

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
        """
        Returnerer totalt antall fisk telt siden siste reset.
        """
        return self.total_count

    def get_counted_track_ids(self) -> List[int]:
        """
        Returnerer kopi av listen over track_id-er som allerede er telt.
        """
        return self.counted_track_ids.copy()

    def get_track_states(self) -> Dict[int, TrackState]:
        """
        Returnerer intern track-state for alle aktive tracks.
        """
        return self.tracks

    def reset(self) -> None:
        """
        Nullstiller tellerens interne tilstand for en ny uavhengig sekvens.
        """
        self.tracks.clear()
        self.total_count = 0
        self.counted_track_ids.clear()

    def _get_zone(self, position_value: float) -> str:
        """
        Konverterer en x- eller y-posisjon til sone relativt til tellelinjen.

        Returns:
            "before", "middle" eller "after".
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
        Fjerner etterfølgende duplikater fra sonehistorikken.

        Dette gjør at lange perioder i samme sone ikke påvirker
        kryssingsmønsteret.
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
        Vurderer om et track har fullført et gyldig kryssingsmønster
        og dermed skal telles.
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
        Sjekker om et bestemt sonemønster finnes i sekvensen.
        """
        if len(sequence) < len(pattern):
            return False

        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                return True

        return False

    def _cleanup_old_tracks(self, current_frame: int) -> None:
        """
        Fjerner tracks som ikke har vært observert på for mange frames.
        """
        to_remove = []

        for track_id, state in self.tracks.items():
            if current_frame - state.last_frame_seen > self.config.max_missing_frames:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]