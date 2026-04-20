#MAIN för programmet 

from src.common.species import CLASS_NAMES


class HomeManager:
    def __init__(self, tracker, counter, session_service, image_source_factory):
        self.tracker = tracker
        self.counter = counter
        self.session_service = session_service
        self.image_source_factory = image_source_factory

        self.image_iterator = None
        self.is_running = False
        self.frame_index = 0

    def start(self):
        self.session_service.ensure_session_exists()
        self.tracker.reset()
        self.counter.reset()

        self.image_iterator = self.image_source_factory()
        self.frame_index = 0
        self.is_running = True

    def step(self):
        if not self.is_running:
            return

        try:
            image_path = next(self.image_iterator)
        except StopIteration:
            self.stop()
            return

        result = self.tracker.update(image_path)
        tracked_objects = result["tracked_objects"]

        new_counts = self.counter.update(
            tracked_objects=tracked_objects,
            frame_index=self.frame_index
        )

        if new_counts > 0:
            counted_ids = set(self.counter.get_counted_track_ids())

            for obj in tracked_objects:
                track_id = obj.get("track_id")

                if track_id in counted_ids:
                    class_id = obj.get("class_id")
                    species_name = CLASS_NAMES.get(class_id, f"Ukjent ({class_id})")
                    self.session_service.increment_species_count(species_name)

        self.frame_index += 1

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False
        self.session_service.stop_session()

    def get_total_count(self) -> int:
        return self.counter.get_total_count()

    def get_species_summary(self) -> list[dict]:
        session = self.session_service.get_active_session()
        if not session:
            return []

        species_counts = session.get("species_counts", {})
        return [
            {"name": name, "count": count, "weight_kg": 0}
            for name, count in species_counts.items()
        ]