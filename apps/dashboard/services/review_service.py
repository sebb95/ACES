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
    """
    UI-facing adapter for the review page.

    Wraps ReviewManager and enriches the returned data with:
    - species name from class_id
    - placeholder confidence
    - placeholder queue list for future UI expansion
    """

    def __init__(self):
        self.manager = ReviewManager()

    def get_review_page_data(self) -> dict:
        item = self.manager.get_next_item()

        if item is None:
            return {
                "trip_name": "Tur_2026_03_14",
                "catch_id": "Okt_003",
                "pending_count": 0,
                "selected_item": None,
                "queue": [],
                "species_options": list(CLASS_NAMES.values()),
            }

        class_id = item["class_id"]
        species_name = CLASS_NAMES.get(class_id, f"Ukjent ({class_id})")

        selected_item = {
            "filename": item["filename"],
            "path": item["path"],
            "class_id": class_id,
            "species_name": species_name,
            "polygon": item["polygon"],
            # Placeholder until confidence exists in backend
            "confidence": 52,
            # Placeholder until timestamp exists in backend
            "timestamp": "12:03:21",
        }

        # Placeholder queue list for UI layout.
        # Later this should come from a real list_pending_items() method.
        queue = [
            {"timestamp": "12:03:21", "prediction": "Sei?", "confidence": 52},
            {"timestamp": "12:05:03", "prediction": "Ukjent", "confidence": 41},
            {"timestamp": "12:08:11", "prediction": "Torsk?", "confidence": 58},
        ]

        return {
            "trip_name": "Tur_2026_03_14",
            "catch_id": "Okt_003",
            "pending_count": len(queue),  # placeholder for now
            "selected_item": selected_item,
            "queue": queue,
            "species_options": list(CLASS_NAMES.values()),
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