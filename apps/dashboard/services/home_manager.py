from pathlib import Path
import cv2

from src.common.species import CLASS_NAMES
from services.settings_service import SettingsService
from src.vision.active_learning.active_learning_logic import trigger_hard_example_save
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
        self.input_mode = "images"
        self.video_capture = None
        self.frame_skip = 0
        self.processing_finished = False

    def _get_settings(self) -> dict:
        return self.settings_service.get()
    
    def _get_input_mode(self) -> str:
        settings = self._get_settings()
        input_settings = settings.get("input", {})

        input_type = input_settings.get("input_type", "image_folder")

        if input_type == "video_file":
            return "video"

        if input_type == "image_folder":
            return "images"

        return "images"
    
    def _resolve_video_path(self) -> Path:
        settings = self._get_settings()
        return Path(settings["input"]["video_path"])

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

        settings = self._get_settings()
        input_mode = self._get_input_mode()
        self.input_mode = input_mode

        if input_mode == "video":
            video_path = self._resolve_video_path()
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            self.video_capture = cv2.VideoCapture(str(video_path))
            if not self.video_capture.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            # === NY: Optimaliseringer for video ===
            self.frame_skip = settings.get("processing", {}).get("frame_skip", 2)  # Justerbar
            self.imgsz = settings.get("processing", {}).get("imgsz", 640)
            self.half = settings.get("processing", {}).get("half", True)
            self.device = settings.get("processing", {}).get("device", 0)

            self.image_paths = []
            self.image_iterator = None
            print(f"[START] Video opened: {video_path} (frame_skip={self.frame_skip})")

        else:
            self.image_paths = self._collect_image_paths()
            self.image_iterator = iter(self.image_paths)
            print(f"[START] Image folder loaded: {len(self.image_paths)} images")

        self.frame_index = 0
        self.processing_finished = False
        self.is_running = True

        print(f"[START] mode={input_mode}")
        print(f"[START] total frames loaded: {len(self.image_paths)}")

    def step(self):
        if not self.is_running:
            return

        print(f"[STEP] frame_index={self.frame_index}")

        if self.input_mode == "video":
            frame = None
            success = False

            # LØSNING: Bruk grab() for å hoppe over frames lynraskt uten CPU-dekoding
            for _ in range(self.frame_skip - 1):
                self.video_capture.grab() 
            
            # Bruk read() KUN på den framen du faktisk skal sende til YOLO
            success, frame = self.video_capture.read()

            if not success:
                print("[VIDEO] No more frames → processing finished")
                if self.video_capture is not None:
                    self.video_capture.release()
                    self.video_capture = None
                self.is_running = False
                self.processing_finished = True
                return

            image_path = f"video_frame_{self.frame_index:06d}"

            # === KORRIGERT KALL (uten ugyldige parametere) ===
            result = self.tracker.update_frame(
                frame=frame,
                frame_name=image_path,
            )

        else:
            try:
                image_path = next(self.image_iterator)
            except StopIteration:
                print("[IMAGES] No more images → processing finished")
                self.is_running = False
                self.processing_finished = True
                return

            result = self.tracker.update(str(image_path))

        # === RESTEN AV LOGIKKEN (uendret) ===
        counted_before = set(self.counter.get_counted_track_ids())
        tracked_objects = result.get("tracked_objects", [])
        original_frame = result.get("original_frame")

        print(f"[TRACK] objects={len(tracked_objects)}")

        self.counter.update(
            tracked_objects=tracked_objects,
            frame_index=self.frame_index,
        )

        counted_after = set(self.counter.get_counted_track_ids())
        newly_counted_ids = counted_after - counted_before
        new_count = len(newly_counted_ids)

        if new_count > 0:
            print(f"[COUNT] +{new_count} fish")

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
                )

                if saved_to_review:
                    self.session_service.increment_uncertain_count()

        self.frame_index += 1

    def stop(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        session = self.session_service.get_active_session()

        if not session:
            self.is_running = False
            self.processing_finished = False
            print("[STOP] no active session to save")
            return

        self.is_running = False
        self.processing_finished = False

        self.session_service.stop_session()
        print("[STOP] session finished and saved")

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