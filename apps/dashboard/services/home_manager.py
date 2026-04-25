from pathlib import Path

from src.common.species import CLASS_NAMES
from services.settings_service import SettingsService
from src.vision.active_learning_logic import trigger_hard_example_save
from services.weight_manager import WeightManager


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

    def _get_review_thresholds(self) -> tuple[float, float]:
        """
        Confidence range for active learning.

        Below min_confidence:
            ignore as too uncertain / garbage / noise

        Between min_confidence and max_confidence:
            counted fish goes to review

        Above max_confidence:
            trusted prediction, no review
        """
        settings = self._get_settings()

        active_learning = settings.get("active_learning", {})

        min_confidence = active_learning.get("review_min_confidence", 0.30)
        max_confidence = active_learning.get("review_max_confidence", 0.80)

        return float(min_confidence), float(max_confidence)

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
        self.session_service.start_session()
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
        original_frame = result.get("original_frame")

        self.counter.update(
            tracked_objects=tracked_objects,
            frame_index=self.frame_index,
        )

        counted_after = set(self.counter.get_counted_track_ids())
        newly_counted_ids = counted_after - counted_before

        if newly_counted_ids:
            review_min_confidence, review_max_confidence = self._get_review_thresholds()

            session = self.session_service.get_active_session()
            session_id = session.get("session_id") if session else None

            for obj in tracked_objects:
                track_id = obj.get("track_id")

                if track_id not in newly_counted_ids:
                    continue

                class_id = obj.get("class_id")
                species_name = CLASS_NAMES.get(class_id, f"Ukjent ({class_id})")

                self.session_service.increment_species_count(species_name)

                confidence = obj.get("confidence", 0.0)
                mask_coords = obj.get("mask_coords", [])

                saved_to_review = trigger_hard_example_save(
                    frame=original_frame,
                    mask_coords=mask_coords,
                    conf=confidence,
                    cls_id=class_id,
                    track_id=track_id,
                    min_confidence=review_min_confidence,
                    max_confidence=review_max_confidence,
                    session_id=session_id,
                    image_path=str(image_path),
                )

                if saved_to_review:
                    self.session_service.increment_uncertain_count()

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

    def get_weight_summary(self) -> dict:
        session = self.session_service.get_active_session()
        if not session:
            return {
                "total_count": 0,
                "total_weight_kg": 0.0,
                "torsk": {
                    "name": "Torsk",
                    "count": 0,
                    "average_weight_kg": 0.0,
                    "weight_kg": 0.0,
                },
                "sei": {
                    "name": "Sei",
                    "count": 0,
                    "average_weight_kg": 0.0,
                    "weight_kg": 0.0,
                },
                "bifangst": {
                    "name": "Bifangst",
                    "count": 0,
                    "weight_kg": 0.0,
                    "species": [],
                },
                "species_breakdown": [],
            }

        settings = self._get_settings()
        species_counts = session.get("species_counts", {})
        species_weights = settings.get("species", {}).get("weights_kg", {})

        weight_manager = WeightManager(species_weights)
        return weight_manager.calculate(species_counts)

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