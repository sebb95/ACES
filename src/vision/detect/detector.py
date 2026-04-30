# a separate detector class, kept for debugging or eventual modular implementation. 
#runtime detection happens interlally within Ultralytics model.track() module in src/track/tracker.py

from pathlib import Path
from typing import Any
from ultralytics import YOLO
import cv2


class FishDetector:
    """
    Separat deteksjonsmodul for fiskebilder ved bruk av en trent YOLO-modell.

    Klassen brukes til testing, debugging og eventuell videre modulær
    implementasjon av deteksjonssteget. I runtime-pipelinen utføres deteksjon
    hovedsakelig gjennom Ultralytics sin `model.track()`-funksjon i FishTracker.

    Ansvar:
    - laste inn trente YOLO-vekter
    - kjøre inferens på enkeltbilder
    - hente ut strukturerte deteksjonsresultater
    - lagre annoterte bilder for inspeksjon
    """

    def __init__(self, weights_path: str, conf: float = 0.25) -> None:
        """
        Initialiserer detektoren med modellvekter og confidence-threshold.

        Args:
            weights_path: Sti til trente YOLO-vekter (.pt-fil).
            conf: Minimum confidence for at en deteksjon skal tas med.
        """
        self.weights_path = Path(weights_path)
        self.conf = conf

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        self.model = YOLO(str(self.weights_path))

    def detect(self, image_path: str, save_dir: str = "outputs/runs/detect") -> dict[str, Any]:
        """
        Kjører deteksjon på ett bilde og returnerer strukturerte resultater.

        Args:
            image_path: Sti til bildet som skal analyseres.
            save_dir: Mappe der annotert output-bilde lagres.

        Returns:
            Dictionary med bildebane, liste over deteksjoner, antall deteksjoner
            og sti til lagret annotert bilde.
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
        Konverterer strukturerte deteksjoner til tracker-kompatibelt format.

        Output-format:
            [x1, y1, x2, y2, score]

        Args:
            detections: Liste med deteksjoner fra detect().

        Returns:
            Liste med bounding boxes og confidence-score til bruk i tracking.
        """
        tracker_input = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["confidence"]
            tracker_input.append([x1, y1, x2, y2, score])

        return tracker_input