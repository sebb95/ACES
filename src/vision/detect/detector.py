# src/vision/detect/detector.py

from pathlib import Path
from typing import Any
from ultralytics import YOLO
import cv2


class FishDetector:
    """
    Runtime detector for fish images using a trained YOLO model.

    Responsibilities:
    - load trained weights
    - run inference on an input image
    - extract structured detection results
    - save annotated output images
    """

    def __init__(self, weights_path: str, conf: float = 0.25) -> None:
        """
        Initialize the detector.
        Args: weights_path: Path to trained YOLO weights (.pt file)
        conf: Confidence threshold for detections
        """
        self.weights_path = Path(weights_path)
        self.conf = conf

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))

    def detect(self, image_path: str, save_dir: str = "outputs/runs/detect") -> dict[str, Any]:
        """
        Run detection on one image.

        Args:
            image_path: Path to input image
            save_dir: Directory where annotated output image will be stored

        Returns:
            Dictionary containing:
            - image_path
            - detections
            - num_detections
            - saved_image
        """
        image_path = Path(image_path)
        save_dir = Path(save_dir)

        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            save=False,
            verbose=False,
        )

        result = results[0]
        detections = []

        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                detection = {
                    "bbox": [round(v, 2) for v in xyxy],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                }
                detections.append(detection)

        annotated_image = result.plot()
        output_image_path = save_dir / image_path.name
        cv2.imwrite(str(output_image_path), annotated_image)

        output = {
            "image_path": str(image_path),
            "detections": detections,
            "num_detections": len(detections),
            "saved_image": str(output_image_path),
        }

        return output

    def detections_for_tracker(self, detections: list[dict[str, Any]]) -> list[list[float]]:
        """
        Convert structured detections to tracker-friendly format.

        Expected output format:
        [x1, y1, x2, y2, score]

        Args:
            detections: List of structured detection dictionaries

        Returns:
            List of detections in tracker-friendly format
        """
        tracker_input = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["confidence"]
            tracker_input.append([x1, y1, x2, y2, score])

        return tracker_input