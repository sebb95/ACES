from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

# --- NY IMPORT FOR ACTIVE LEARNING ---
from src.vision.active_learning_logic import trigger_hard_example_save


class FishTracker:
    """
    Runtime fish tracker using Ultralytics YOLO tracking mode with ByteTrack.

    Responsibilities:
    - load trained YOLO weights
    - run tracking on sequential frames
    - preserve track identities across frames
    - save annotated tracking outputs
    - return structured tracked objects

    Note:
    This implementation uses Ultralytics tracking mode with tracker='bytetrack.yaml'
    and persist=True so that track state is preserved across consecutive frames.
    """

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        tracker_cfg: str = "bytetrack.yaml",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf = conf
        self.tracker_cfg = tracker_cfg

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))

    def update(
        self,
        image_path: str,
        save_dir: str = "outputs/runs/track",
    ) -> dict[str, Any]:
        """
        Run ByteTrack on one frame.

        Args:
            image_path: Path to input image
            save_dir: Directory for annotated output frame

        Returns:
            Dictionary containing:
            - image_path
            - tracked_objects
            - num_tracks
            - saved_image
        """
        image_path = Path(image_path)
        save_dir = Path(save_dir)

        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.track(
            source=str(image_path),
            conf=self.conf,
            tracker=self.tracker_cfg,
            persist=True,
            save=False,
            verbose=False,
        )

        result = results[0]
        tracked_objects = []

        # --- ACTIVE LEARNING: Hent originalbildet og maskene fra Ultralytics ---
        orig_frame = result.orig_img 
        masks = result.masks
        # ----------------------------------------------------------------------

        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None:
            ids = boxes.id

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                # --- ACTIVE LEARNING: Trigger lagring av usikre fisker ---
                if masks is not None and len(masks.xyn) > i:
                    # xyn er en liste med normaliserte segmenteringskoordinater
                    mask_coords = masks.xyn[i].flatten().tolist()
                    trigger_hard_example_save(orig_frame, mask_coords, confidence, class_id)
                # ---------------------------------------------------------

                track_id = None
                if ids is not None:
                    track_id = int(ids[i].item())

                x1, y1, x2, y2 = xyxy
                center_x = round((x1 + x2) / 2, 2)
                center_y = round((y1 + y2) / 2, 2)

                tracked_obj = {
                    "track_id": track_id,
                    "bbox": [round(v, 2) for v in xyxy],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "center": [center_x, center_y],
                }
                tracked_objects.append(tracked_obj)

        annotated_image = result.plot()
        output_image_path = save_dir / image_path.name
        cv2.imwrite(str(output_image_path), annotated_image)

        return {
            "image_path": str(image_path),
            "tracked_objects": tracked_objects,
            "num_tracks": len(tracked_objects),
            "saved_image": str(output_image_path),
        }

    def reset(self) -> None:
        """
        Reset tracking state by reloading the model.

        Useful when starting a new independent sequence.
        """
        self.model = YOLO(str(self.weights_path))