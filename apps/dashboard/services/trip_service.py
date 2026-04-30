from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class TripService:
    """
    Håndterer opprettelse, oppdatering og avslutning av turer.

    En tur representerer en overordnet tidsperiode som kan inneholde
    flere økter. Klassen sørger for at det alltid finnes en aktiv tur,
    og at økter kobles korrekt til denne.

    Ansvar:
    - opprette ny tur dersom ingen er aktiv
    - lagre og oppdatere turdata
    - håndtere navn på tur
    - avslutte tur og arkivere den
    """
    def __init__(self) -> None:
        self.trips_dir = Path("data/history/trips")
        self.active_trip_path = Path("data/history/active_trip.json")

        self.trips_dir.mkdir(parents=True, exist_ok=True)
        self.active_trip_path.parent.mkdir(parents=True, exist_ok=True)

    def get_active_trip(self) -> dict | None:
        if not self.active_trip_path.exists():
            return None

        try:
            with open(self.active_trip_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def ensure_active_trip(self) -> dict:
        active_trip = self.get_active_trip()

        if active_trip:
            return active_trip

        return self.start_new_trip()

    def start_new_trip(self, trip_name: str | None = None) -> dict:
        now = datetime.now()
        trip_id = f"tur_{now.strftime('%Y-%m-%d_%H%M%S')}"

        if trip_name is None:
            trip_name = f"Tur_{now.strftime('%Y-%m-%d')}"

        trip = {
            "trip_id": trip_id,
            "trip_name": trip_name,
            "started_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "status": "active",
        }

        self._save_trip(trip)
        self._save_active_trip(trip)

        return trip

    def rename_active_trip(self, new_name: str) -> dict | None:
        trip = self.get_active_trip()

        if not trip:
            return None

        new_name = new_name.strip()
        if not new_name:
            return trip

        old_name = trip["trip_name"]
        trip["trip_name"] = new_name

        self._save_trip(trip)
        self._save_active_trip(trip)

        # 🔥 Update all sessions belonging to this trip
        sessions_dir = Path("data/history/sessions")

        for file in sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("trip_id") == trip["trip_id"]:
                    data["trip_name"] = new_name

                    with open(file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)

            except Exception:
                continue

        return trip

    def end_active_trip(self) -> dict | None:
        trip = self.get_active_trip()

        if not trip:
            return None

        trip["ended_at"] = datetime.now().isoformat(timespec="seconds")
        trip["status"] = "completed"

        self._save_trip(trip)

        if self.active_trip_path.exists():
            self.active_trip_path.unlink()

        return trip

    def _save_trip(self, trip: dict) -> None:
        path = self.trips_dir / f"{trip['trip_id']}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(trip, f, indent=2, ensure_ascii=False)

    def _save_active_trip(self, trip: dict) -> None:
        with open(self.active_trip_path, "w", encoding="utf-8") as f:
            json.dump(trip, f, indent=2, ensure_ascii=False)