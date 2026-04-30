from datetime import datetime
from pathlib import Path

from apps.dashboard import state
from services.session_manager import SessionManager
from services.trip_service import TripService


class SessionService:
    
    """
    Tjenestelag for håndtering av aktive økter (sessions) i systemet.

    Klassen fungerer som et mellomlag mellom runtime-pipelinen og lagring,
    og er ansvarlig for å opprette, oppdatere og avslutte økter.

    Ansvar:
    - opprette ny økt ved start
    - oppdatere tellinger og statistikk fortløpende
    - utføre autosave under kjøring
    - lagre ferdig økt til disk via SessionManager
    - koble økt til aktiv tur via TripService
    """
    def __init__(self) -> None:
        self.manager = SessionManager()
        self.trip_service = TripService()

    def ensure_session_exists(self) -> None:
        if not state.has_active_session():
            self.start_session()

    def start_session(self) -> None:
        now = datetime.now()
        session_id = self._generate_session_id(now)
        trip = self.trip_service.ensure_active_trip()

        session_data = {
            "session_id": session_id,
            "trip_id": trip["trip_id"],
            "trip_name": trip["trip_name"],
            "started_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "duration_seconds": None,
            "species_counts": {},
            "total_count": 0,
            "uncertain_count": 0,
            "review_items_created": 0,
            "corrections": 0,
            "status": "running",
        }

        state.set_active_session(session_data)

    def get_active_session(self) -> dict | None:
        return state.get_active_session()

    def increment_species_count(self, species_name: str, amount: int = 1) -> None:
        session = self.get_active_session()
        if not session:
            return

        counts = session.setdefault("species_counts", {})
        counts[species_name] = counts.get(species_name, 0) + amount

        session["total_count"] = sum(counts.values())

        state.set_active_session(session)
        self.autosave_session()

    def increment_uncertain_count(self, amount: int = 1) -> None:
        session = self.get_active_session()
        if not session:
            return

        session["uncertain_count"] = session.get("uncertain_count", 0) + amount
        session["review_items_created"] = session.get("review_items_created", 0) + amount

        state.set_active_session(session)
        self.autosave_session()

    def increment_corrections(self, amount: int = 1) -> None:
        """
        Kept for compatibility, but live review corrections should no longer
        change session counts in the new strategy.
        """
        session = self.get_active_session()
        if not session:
            return

        session["corrections"] = session.get("corrections", 0) + amount
        state.set_active_session(session)

    def stop_session(self) -> dict | None:
        session = self.get_active_session()
        if not session:
            return None

        ended_at = datetime.now()
        started_at = datetime.fromisoformat(session["started_at"])

        session["ended_at"] = ended_at.isoformat(timespec="seconds")
        session["duration_seconds"] = int((ended_at - started_at).total_seconds())
        session["status"] = "completed"
        session["total_count"] = sum(session.get("species_counts", {}).values())

        session.setdefault("uncertain_count", 0)
        session.setdefault("review_items_created", 0)
        session.setdefault("corrections", 0)

        self.manager.save_session(session)
        state.clear_active_session()

        return session

    def _generate_session_id(self, now: datetime) -> str:
        today_str = now.strftime("%Y-%m-%d")
        sessions_dir = Path("data/history/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)

        existing_numbers = []

        for file in sessions_dir.glob(f"Økt_*_{today_str}.json"):
            parts = file.stem.split("_")
            if len(parts) != 3:
                continue

            try:
                number = int(parts[1])
                existing_numbers.append(number)
            except ValueError:
                continue

        next_number = max(existing_numbers, default=0) + 1

        while True:
            session_id = f"Økt_{next_number:03d}_{today_str}"
            file_path = sessions_dir / f"{session_id}.json"

            if not file_path.exists():
                return session_id

            next_number += 1
        
        
    
    def autosave_session(self) -> None:
        session = self.get_active_session()
        if not session:
            return

        session["status"] = "running_autosaved"
        session["last_autosaved_at"] = datetime.now().isoformat(timespec="seconds")
        session["total_count"] = sum(session.get("species_counts", {}).values())

        self.manager.save_session(session)
        state.set_active_session(session)