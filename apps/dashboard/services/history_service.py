from __future__ import annotations

from services.history_manager import HistoryManager


class HistoryService:
    """UI-facing adapter for the history page."""

    def __init__(self) -> None:
        self.manager = HistoryManager()

    def get_history_page_data(self) -> dict:
        raw_sessions = self.manager.list_sessions()

        if not raw_sessions:
            return {
                "trip": {
                    "name": "Ingen tur lastet",
                    "start": "-",
                    "end": "-",
                    "total_count": 0,
                    "total_weight": 0,
                },
                "sessions": [],
            }

        sessions = []
        total_count = 0
        total_weight = 0

        trip_name = raw_sessions[0].get("trip_name", "Ukjent tur")
        trip_start = raw_sessions[0].get("trip_start", "-")
        trip_end = raw_sessions[0].get("trip_end", "-")

        for raw in raw_sessions:
            count = raw.get("total_count", 0)
            weight = raw.get("total_weight", 0)

            total_count += count
            total_weight += weight

            sessions.append(
                {
                    "id": raw.get("session_id", "Ukjent økt"),
                    "start": raw.get("start_time", "-"),
                    "end": raw.get("end_time", "-"),
                    "count": count,
                    "weight": weight,
                    "corrections": raw.get("corrections", 0),
                    "expanded": False,
                    "species": raw.get("species", []),
                }
            )

        return {
            "trip": {
                "name": trip_name,
                "start": trip_start,
                "end": trip_end,
                "total_count": total_count,
                "total_weight": total_weight,
            },
            "sessions": sessions,
        }