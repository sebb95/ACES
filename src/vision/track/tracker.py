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
        print(self.model.names)

    def update(
        self,
        image_path: str,
        save_dir: str = "outputs/runs/track",
    ) -> dict[str, Any]:
        """
        Run ByteTrack on one frame.

        Returns:
            {
                image_path,
                tracked_objects,
                num_tracks,
                saved_image
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
        tracked_objects = []

        boxes = result.boxes
        masks = result.masks
        ids = boxes.id if boxes is not None and boxes.id is not None else None

        if boxes is not None and boxes.xyxy is not None:
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                # --- mask coords (for segmentation models) ---
                mask_coords = []
                if masks is not None and masks.xyn is not None and len(masks.xyn) > i:
                    mask_coords = masks.xyn[i].flatten().tolist()

                # --- track id (safe handling) ---
                track_id = None
                if ids is not None and ids[i] is not None:
                    track_id = int(ids[i].item())

                # --- center point ---
                x1, y1, x2, y2 = xyxy
                center_x = round((x1 + x2) / 2, 2)
                center_y = round((y1 + y2) / 2, 2)

                tracked_obj = {
                    "track_id": track_id,
                    "bbox": [round(v, 2) for v in xyxy],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "center": [center_x, center_y],
                    "mask_coords": mask_coords,
                }

                tracked_objects.append(tracked_obj)

        # --- save annotated frame ---
        #annotated_image = result.plot()
        #output_image_path = save_dir / image_path.name
        #cv2.imwrite(str(output_image_path), annotated_image)

        return {
            "image_path": str(image_path),
            "tracked_objects": tracked_objects,
            "num_tracks": len(tracked_objects),
            "saved_image": None,
            "original_frame": result.orig_img,
        }

    def reset(self) -> None:
        """Reset tracking state."""
        self.model = YOLO(str(self.weights_path))

    def set_weights_path(self, weights_path: str) -> None:
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))

    def update_frame(
        self,
        frame,
        frame_name: str,
        save_dir: str = "outputs/runs/track",
    ) -> dict[str, Any]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.track(
            source=frame,
            conf=self.conf,
            tracker=self.tracker_cfg,
            persist=True,
            save=False,
            verbose=False,
        )

        result = results[0]
        tracked_objects = []

        boxes = result.boxes
        masks = result.masks
        ids = boxes.id if boxes is not None and boxes.id is not None else None

        if boxes is not None and boxes.xyxy is not None:
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                mask_coords = []
                if masks is not None and masks.xyn is not None and len(masks.xyn) > i:
                    mask_coords = masks.xyn[i].flatten().tolist()

                track_id = None
                if ids is not None and ids[i] is not None:
                    track_id = int(ids[i].item())

                x1, y1, x2, y2 = xyxy
                center_x = round((x1 + x2) / 2, 2)
                center_y = round((y1 + y2) / 2, 2)

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

        #annotated_image = result.plot()
        #output_image_path = save_dir / f"{frame_name}.jpg"
        #cv2.imwrite(str(output_image_path), annotated_image)
        
        return {
        "image_path": frame_name,
        "tracked_objects": tracked_objects,
        "num_tracks": len(tracked_objects),
        "saved_image": None,
        "original_frame": result.orig_img,
        }

        