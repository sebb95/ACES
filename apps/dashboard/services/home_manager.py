import os
import cv2
import time
import queue
import threading
import platform
from pathlib import Path

# Project-specific imports
from src.common.species import CLASS_NAMES
from services.settings_service import SettingsService
from services.weight_manager import WeightManager
from src.vision.active_learning.active_learning_logic import trigger_hard_example_save

class HomeManager:
    """
    Orkestrerer runtime for fangstregistrering i ACES med Parallel Processing.
    """
    def __init__(self, tracker, counter, session_service):
        self.tracker = tracker
        self.counter = counter
        self.session_service = session_service
        self.settings_service = SettingsService()
        
        self.settings_service = SettingsService()
        self._settings_cache = None # NEW: Cache for high-speed access

        # --- PARALLEL PROCESSING ---
        self.frame_queue = queue.Queue(maxsize=128)
        self.producer_thread = None
        self.batch_size = 8 
        # ---------------------------

        self.image_paths = []
        self.image_iterator = None
        self.is_running = False
        self.frame_index = 0
        self.input_mode = "images"
        self.video_capture = None
        self.frame_skip = 0
        self.processing_finished = False

    # ==========================================
    # CORE LOGIC (START / STEP / STOP)
    # ==========================================

    def start(self):
        """Starter en ny telleøkt og fyrer opp bakgrunnstråden for video."""
        # NEW: Load and cache settings ONCE at the start
        self._settings_cache = self.settings_service.get() 
        
        self.session_service.start_session()
        self._configure_tracker()
        self.tracker.reset()
        self.counter.reset()

        self.input_mode = self._get_input_mode()

        if self.input_mode == "video":
            video_path = self._resolve_video_path()
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            self.video_capture = cv2.VideoCapture(str(video_path))
            
            settings = self.settings_service.get()
            processing_settings = settings.get("processing", {})
            self.frame_skip = max(1, int(processing_settings.get("frame_skip", 1)))

            # Start paralell dekoding
            self.is_running = True
            self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
            self.producer_thread.start()
        else:
            self.image_paths = self._collect_image_paths()
            self.image_iterator = iter(self.image_paths)
            self.is_running = True

        self.frame_index = 0
        self.processing_finished = False

    def step(self):
        """Prosesserer frames i batcher fra køen."""
        if not self.is_running:
            return

        if self.input_mode == "video":
            batch_frames = []
            for _ in range(self.batch_size):
                try:
                    batch_frames.append(self.frame_queue.get_nowait())
                except queue.Empty:
                    break

            if not batch_frames:
                if self.producer_thread and not self.producer_thread.is_alive():
                    self.stop()
                return

            for frame in batch_frames:
                image_path = f"video_frame_{self.frame_index:06d}"
                result = self.tracker.update_frame(frame=frame, frame_name=image_path)
                self._original_logic_step(result)
                self.frame_index += 1
        else:
            try:
                image_path = next(self.image_iterator)
                result = self.tracker.update(str(image_path))
                self._original_logic_step(result)
                self.frame_index += 1
            except StopIteration:
                self.stop()

    def _original_logic_step(self, result):
        """Din originale telle- og review-logikk."""
        counted_before = set(self.counter.get_counted_track_ids())
        tracked_objects = result.get("tracked_objects", [])
        original_frame = result.get("original_frame")

        self.counter.update(tracked_objects=tracked_objects, frame_index=self.frame_index)

        counted_after = set(self.counter.get_counted_track_ids())
        newly_counted_ids = counted_after - counted_before

        if newly_counted_ids:
            review_min_confidence, review_max_confidence = self._get_review_thresholds()
            for obj in tracked_objects:
                track_id = obj.get("track_id")
                if track_id not in newly_counted_ids:
                    continue

                class_id = obj.get("class_id")
                species_name = CLASS_NAMES.get(class_id, f"Ukjent ({class_id})")
                self.session_service.increment_species_count(species_name)

                confidence = float(obj.get("confidence", 0.0))
                if review_min_confidence <= confidence <= review_max_confidence:
                    saved = trigger_hard_example_save(
                        frame=original_frame,
                        mask_coords=obj.get("mask_coords", []),
                        conf=confidence,
                        cls_id=class_id,
                    )
                    if saved:
                        self.session_service.increment_uncertain_count()

    def stop(self):
        self.is_running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.session_service.stop_session()

    # ==========================================
    # PRODUCER THREAD
    # ==========================================

    def _producer_loop(self):
        while self.is_running and self.video_capture and self.video_capture.isOpened():
            if self.frame_queue.full():
                time.sleep(0.001)
                continue
            
            for _ in range(self.frame_skip - 1):
                self.video_capture.grab()
            
            success, frame = self.video_capture.read()
            if success:
                self.frame_queue.put(frame)
            else:
                break
        self.processing_finished = True

    # ==========================================
    # DATA & SUMMARY
    # ==========================================

    def get_weight_summary(self) -> dict:
        session = self.session_service.get_active_session()
        if not session:
            return {
                "total_count": 0,
                "total_weight_kg": 0.0,
                "torsk": {"name": "Torsk", "count": 0, "average_weight_kg": 0.0, "weight_kg": 0.0},
                "sei": {"name": "Sei", "count": 0, "average_weight_kg": 0.0, "weight_kg": 0.0},
                "bifangst": {"name": "Bifangst", "count": 0, "weight_kg": 0.0, "species": []},
                "species_breakdown": [],
            }

        settings = self.settings_service.get()
        species_counts = session.get("species_counts", {})
        species_weights = settings.get("species", {}).get("weights_kg", {})

        weight_manager = WeightManager(species_weights)
        return weight_manager.calculate(species_counts)

    # ==========================================
    # HELPER METHODS (INTERNAL)
    # ==========================================

    def _get_settings(self) -> dict:
        """Returns cached settings if available to avoid disk hammering."""
        if self._settings_cache is not None:
            return self._settings_cache
        return self.settings_service.get()
    
    def _get_input_mode(self) -> str:
        settings = self._get_settings()
        input_type = settings.get("input", {}).get("input_type", "image_folder")
        return "video" if input_type == "video_file" else "images"
    
    def _resolve_video_path(self) -> Path:
        return Path(self._get_settings()["input"]["video_path"])

    def _resolve_model_path(self) -> Path:
        selected_model = self._get_settings()["model"]["selected_model"]
        return Path("outputs/weights") / selected_model

    def _resolve_dataset_path(self) -> Path:
        return Path(self._get_settings()["input"]["dataset_path"])

    def _get_review_thresholds(self) -> tuple[float, float]:
        active_learning = self._get_settings().get("active_learning", {})
        return float(active_learning.get("review_min_confidence", 0.30)), float(active_learning.get("review_max_confidence", 0.80))

    def _collect_image_paths(self) -> list[Path]:
        dataset_path = self._resolve_dataset_path()
        if not dataset_path.exists(): return []
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in dataset_path.iterdir() if p.is_file() and p.suffix.lower() in allowed])

    def _configure_tracker(self) -> None:
        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.tracker.set_weights_path(str(model_path))