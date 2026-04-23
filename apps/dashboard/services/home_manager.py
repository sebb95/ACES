from pathlib import Path

from src.common.species import CLASS_NAMES
from services.settings_service import SettingsService


class HomeManager:
    def __init__(self, tracker, counter, session_service):
        self.tracker = tracker
        self.counter = counter
        self.session_service = session_service
        self.settings_service = SettingsService()

        self.image_paths = []
        self.image_iterator = None
        self.is_running = False
        self.frame_index = 0

    def _get_settings(self) -> dict:
        return self.settings_service.get()

    def _resolve_model_path(self) -> Path:
        settings = self._get_settings()
        selected_model = settings["model"]["selected_model"]
        return Path("outputs/weights") / selected_model

    def _resolve_dataset_path(self) -> Path:
        settings = self._get_settings()
        return Path(settings["input"]["dataset_path"])

    def _collect_image_paths(self) -> list[Path]:
        dataset_path = self._resolve_dataset_path()

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        if not dataset_path.is_dir():
            raise ValueError(f"Dataset path is not a folder: {dataset_path}")

        allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        image_paths = sorted(
            [
                path for path in dataset_path.iterdir()
                if path.is_file() and path.suffix.lower() in allowed_suffixes
            ]
        )

        if not image_paths:
            raise ValueError(f"No image files found in dataset folder: {dataset_path}")

        return image_paths

    def _configure_tracker(self) -> None:
        model_path = self._resolve_model_path()

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if hasattr(self.tracker, "set_weights_path"):
            self.tracker.set_weights_path(str(model_path))
        else:
            raise AttributeError(
                "FishTracker must support set_weights_path(weights_path)."
            )

    def start(self):
        self.session_service.ensure_session_exists()
        self._configure_tracker()

        self.tracker.reset()
        self.counter.reset()

        self.image_paths = self._collect_image_paths()
        self.image_iterator = iter(self.image_paths)

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

        counted_before = set(self.counter.get_counted_track_ids())

        result = self.tracker.update(str(image_path))
        tracked_objects = result.get("tracked_objects", [])

        self.counter.update(
            tracked_objects=tracked_objects,
            frame_index=self.frame_index,
        )

        counted_after = set(self.counter.get_counted_track_ids())
        newly_counted_ids = counted_after - counted_before

        if newly_counted_ids:
            for obj in tracked_objects:
                track_id = obj.get("track_id")

                if track_id in newly_counted_ids:
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
        session = self.session_service.get_active_session()
        if not session:
            return 0
        return session.get("total_count", 0)

    def get_species_summary(self) -> list[dict]:
        session = self.session_service.get_active_session()
        if not session:
            return []

        settings = self._get_settings()
        species_counts = session.get("species_counts", {})
        species_weights = settings.get("species", {})

        weight_map = {
            "Torsk": species_weights.get("torsk_weight", 0),
            "Sei": species_weights.get("sei_weight", 0),
            "Bifangst": species_weights.get("bifangst_weight", 0),
        }

        summary = []

        for name, count in species_counts.items():
            avg_weight = weight_map.get(name, 0)
            summary.append(
                {
                    "name": name,
                    "count": count,
                    "weight_kg": round(count * avg_weight, 2),
                }
            )

        return summary

    def get_status(self) -> dict:
        model_path = self._resolve_model_path()
        dataset_path = self._resolve_dataset_path()

        return {
            "is_running": self.is_running,
            "frame_index": self.frame_index,
            "selected_model": model_path.name,
            "model_exists": model_path.exists(),
            "dataset_path": str(dataset_path),
            "dataset_exists": dataset_path.exists(),
        }