from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


class FishTracker:
    """
    Runtime fish tracker using Ultralytics YOLO tracking mode with ByteTrack.

    Responsibilities:
    - load trained YOLO weights
    - run tracking on sequential frames
    - preserve track identities across frames
    - save annotated tracking outputs
    - return structured tracked objects

    Important:
    - This class does NOT save anything to review.
    - Active learning/review decisions happen after counting in HomeManager.
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

        Returns:
            {
                "image_path": str,
                "tracked_objects": list[dict],
                "num_tracks": int,
                "track_ids": list[int | None],
                "original_frame": np.ndarray,
                "saved_image": str,
            }
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
        orig_frame = result.orig_img
        masks = result.masks
        boxes = result.boxes

        tracked_objects = []

        if boxes is not None and boxes.xyxy is not None:
            ids = boxes.id

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                track_id = None
                if ids is not None:
                    track_id = int(ids[i].item())

                x1, y1, x2, y2 = xyxy
                center_x = round((x1 + x2) / 2, 2)
                center_y = round((y1 + y2) / 2, 2)

                mask_coords = []
                if masks is not None and len(masks.xyn) > i:
                    mask_coords = masks.xyn[i].flatten().tolist()

                tracked_objects.append(
                    {
                        "track_id": track_id,
                        "bbox": [round(v, 2) for v in xyxy],
                        "confidence": round(confidence, 4),
                        "class_id": class_id,
                        "center": [center_x, center_y],
                        "mask_coords": mask_coords,
                    }
                )

        annotated_image = result.plot()
        output_image_path = save_dir / image_path.name
        cv2.imwrite(str(output_image_path), annotated_image)

        return {
            "image_path": str(image_path),
            "tracked_objects": tracked_objects,
            "num_tracks": len(tracked_objects),
            "track_ids": [obj["track_id"] for obj in tracked_objects],
            "original_frame": orig_frame,
            "saved_image": str(output_image_path),
        }

    def reset(self) -> None:
        """
        Reset tracking state for a new independent sequence.
        """
        self.model = YOLO(str(self.weights_path))

    def set_weights_path(self, weights_path: str) -> None:
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))