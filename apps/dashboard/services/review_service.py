from services.review_manager import ReviewManager
#from src.common.species import CLASS_NAMES, NAME_TO_CLASS_ID
import importlib
import src.common.species as species_module


class ReviewService:
    """
    Adapter mellom Review-siden og ReviewManager.

    Klassen gjør data fra review-køen klar for UI, henter artsvalg fra
    species.py og videresender brukerhandlinger til ReviewManager.
    Selve filflyttingen og active learning-logikken ligger i ReviewManager.
    """

    def __init__(self):
        self.manager = ReviewManager()

    def _reload_species(self):
        """
        Leser species.py på nytt slik at arter lagt til via Innstillinger
        blir tilgjengelige i Review uten å restarte Streamlit.
        """
        return importlib.reload(species_module)

    def _get_species_options(self) -> list[str]:
        species = self._reload_species()
        return [
            species.CLASS_NAMES[class_id]
            for class_id in sorted(species.CLASS_NAMES)
        ]

    def get_review_page_data(self, selected_index: int = 0) -> dict:
        species = self._reload_species()
        pending_items = self.manager.list_pending_items()
        species_options = self._get_species_options()
        


        if not pending_items:
            return {
                "trip_name": "Tur_2026_03_14",
                "catch_id": "Okt_003",
                "pending_count": 0,
                "selected_item": None,
                "queue": [],
                "species_options": species_options,
            }

        if selected_index < 0 or selected_index >= len(pending_items):
            selected_index = 0

        enriched_items = []

        for item in pending_items:
            class_id = item.get("class_id")
            metadata = item.get("metadata", {})

            species_name = metadata.get("corrected_species_name")

            if not species_name:
                species_name = species.CLASS_NAMES.get(class_id, f"Ukjent ({class_id})")

            enriched_items.append(
                {
                    "filename": item["filename"],
                    "path": item["path"],
                    "class_id": class_id,
                    "species_name": species_name,
                    "polygon": item.get("polygon", []),
                    "confidence": item.get("confidence"),
                    "timestamp": item.get("timestamp"),
                    "session_id": item.get("session_id"),
                    "track_id": item.get("track_id"),
                    "was_counted": item.get("was_counted", False),
                }
            )

        selected_item = enriched_items[selected_index]

        queue = [
            {
                "filename": item["filename"],
                "timestamp": item["timestamp"],
                "prediction": item["species_name"],
                "confidence": item["confidence"],
            }
            for item in enriched_items
        ]

        return {
            "trip_name": "Tur_2026_03_14",
            "catch_id": "Okt_003",
            "pending_count": len(queue),
            "selected_item": selected_item,
            "queue": queue,
            "species_options": species_options,
        }

    def approve(self, filename: str) -> None:
        self.manager.action_approve(filename)

    def reject(self, filename: str, class_id: int | None = None) -> None:
        self.manager.action_reject(filename)

    def send_to_land(self, filename: str) -> None:
        self.manager.action_send_to_land(filename)

    def change_species(self, filename: str, new_species_name: str) -> None:
        species = self._reload_species()
        known_class_id = species.NAME_TO_CLASS_ID.get(new_species_name)

        self.manager.action_change_species(
            filename=filename,
            new_species_name=new_species_name,
            new_class_id=known_class_id,
        )

    def get_pending_count(self) -> int:
        return len(self.manager.list_pending_items())
    
