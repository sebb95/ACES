from services.review_manager import ReviewManager


CLASS_NAMES = {
    0: "Breiflab",
    1: "Brosme",
    2: "Flyndre",
    3: "Hyse",
    4: "Kveite",
    5: "Lange",
    6: "Lyr",
    7: "Sei",
    8: "Torsk",
    9: "Uer",
}


class ReviewService:
    """UI-facing adapter for the review page."""

    def __init__(self):
        self.manager = ReviewManager()

    def get_review_page_data(self, selected_index: int = 0) -> dict:
        pending_items = self.manager.list_pending_items()

        species_options = list(CLASS_NAMES.values())

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
            class_id = item["class_id"]
            species_name = CLASS_NAMES.get(class_id, f"Ukjent ({class_id})")

            enriched_items.append(
                {
                    "filename": item["filename"],
                    "path": item["path"],
                    "class_id": class_id,
                    "species_name": species_name,
                    "polygon": item["polygon"],
                    "confidence": item.get("confidence"),
                    "timestamp": item.get("timestamp"),
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

    def reject(self, filename: str) -> None:
        self.manager.action_reject(filename)

    def send_to_land(self, filename: str) -> None:
        self.manager.action_send_to_land(filename)

    def change_species(self, filename: str, new_species_name: str) -> None:
        reverse_map = {name: class_id for class_id, name in CLASS_NAMES.items()}
        new_class_id = reverse_map[new_species_name]
        self.manager.action_change_species(filename, new_class_id)