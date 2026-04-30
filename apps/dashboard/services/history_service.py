from __future__ import annotations

from collections import defaultdict

from services.history_manager import HistoryManager
from services.settings_service import SettingsService
from services.weight_manager import WeightManager


class HistoryService:
    """
    Adapter mellom lagrede historikkdata og UI.

    Klassen leser økter fra disk via HistoryManager, grupperer dem per tur,
    og beregner aggregerte statistikker som total fangst og estimert vekt.

    Resultatet brukes direkte av History-siden i brukergrensesnittet.
    """

    def __init__(self) -> None:
        self.manager = HistoryManager()
        self.settings_service = SettingsService()

    def get_history_page_data(self) -> dict:
        raw_sessions = self.manager.list_sessions()

        if not raw_sessions:
            return {"trips": []}

        settings = self.settings_service.get()
        species_weights = settings.get("species", {}).get("weights_kg", {})
        weight_manager = WeightManager(species_weights)

        grouped: dict[str, list[dict]] = defaultdict(list)

        for session in raw_sessions:
            trip_id = session.get("trip_id", "unknown_trip")
            grouped[trip_id].append(session)

        trips = []

        for trip_id, sessions_raw in grouped.items():
            sessions_raw = sorted(
                sessions_raw,
                key=lambda s: s.get("started_at", ""),
            )

            trip_name = sessions_raw[0].get("trip_name", "Ukjent tur")
            started_at = sessions_raw[0].get("started_at", "-")
            ended_at = sessions_raw[-1].get("ended_at", "-")

            trip_species_counts: dict[str, int] = {}
            total_uncertain = 0
            total_review_items = 0
            total_corrections = 0

            sessions = []

            for raw in sessions_raw:
                species_counts = raw.get("species_counts", {})

                for species_name, count in species_counts.items():
                    trip_species_counts[species_name] = (
                        trip_species_counts.get(species_name, 0) + int(count)
                    )

                session_weight_summary = weight_manager.calculate(species_counts)

                uncertain_count = int(raw.get("uncertain_count", 0))
                review_items_created = int(raw.get("review_items_created", 0))
                corrections = int(raw.get("corrections", 0))

                total_uncertain += uncertain_count
                total_review_items += review_items_created
                total_corrections += corrections

                sessions.append(
                    {
                        "id": raw.get("session_id", "Ukjent økt"),
                        "start": raw.get("started_at", "-"),
                        "end": raw.get("ended_at", "-"),
                        "count": session_weight_summary["total_count"],
                        "weight_kg": session_weight_summary["total_weight_kg"],
                        "uncertain_count": uncertain_count,
                        "review_items_created": review_items_created,
                        "corrections": corrections,
                        "species": session_weight_summary["species_breakdown"],
                    }
                )

            trip_weight_summary = weight_manager.calculate(trip_species_counts)

            trips.append(
                {
                    "trip_id": trip_id,
                    "trip_name": trip_name,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "total_count": trip_weight_summary["total_count"],
                    "total_weight_kg": trip_weight_summary["total_weight_kg"],
                    "torsk": trip_weight_summary["torsk"],
                    "sei": trip_weight_summary["sei"],
                    "bifangst": trip_weight_summary["bifangst"],
                    "uncertain_count": total_uncertain,
                    "review_items_created": total_review_items,
                    "corrections": total_corrections,
                    "sessions": sessions,
                }
            )

        trips = sorted(
            trips,
            key=lambda t: t.get("started_at", ""),
            reverse=True,
        )

        return {"trips": trips}