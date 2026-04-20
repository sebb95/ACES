#session logic

from datetime import datetime
from pathlib import Path

from apps.dashboard import state
from services.session_manager import SessionManager


class SessionService:
    def __init__(self) -> None:
        self.manager = SessionManager()

    def ensure_session_exists(self) -> None:
        if not state.has_active_session():
            self.start_session()

    def start_session(self) -> None:
        now = datetime.now()
        session_id = self._generate_session_id(now)

        session_data = {
            "session_id": session_id,
            "started_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "species_counts": {},
            "total_count": 0,
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

        counts = session["species_counts"]

        if species_name not in counts:
            counts[species_name] = 0

        counts[species_name] += amount
        session["total_count"] += amount

        state.set_active_session(session)

    def increment_corrections(self, amount: int = 1) -> None:
        session = self.get_active_session()
        if not session:
            return

        session["corrections"] += amount
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
        session["total_count"] = sum(session["species_counts"].values())

        self.manager.save_session(session)
        state.clear_active_session()

        return session

    def _generate_session_id(self, now: datetime) -> str:
        today_str = now.strftime("%Y-%m-%d")
        sessions_dir = Path("data/history/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)

        existing_numbers = []

        for file in sessions_dir.glob(f"okt_*_{today_str}.json"):
            parts = file.stem.split("_")
            if len(parts) != 3:
                continue

            try:
                number = int(parts[1])
                existing_numbers.append(number)
            except ValueError:
                continue

        next_number = max(existing_numbers, default=0) + 1
        return f"okt_{next_number:03d}_{today_str}"