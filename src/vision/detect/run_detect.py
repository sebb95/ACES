#rin file: python src.vision.detect.run_detect data/sample/val/images/

import json
import sys
from pathlib import Path

from detector import FishDetector


def create_run_dir(base_dir: str = "outputs/runs/detect") -> str:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    existing_runs = [
        p.name for p in base_path.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    ]

    run_numbers = []
    for name in existing_runs:
        try:
            run_numbers.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue

    next_run = max(run_numbers, default=0) + 1
    run_dir = base_path / f"run_{next_run:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return str(run_dir)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/vision/detect/run_detect.py <image_or_folder_path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    detector = FishDetector(
        weights_path="outputs/weights/baseline_best.pt",
        conf=0.5,
    )

    run_dir = create_run_dir()
    print(f"Saving results to: {run_dir}")

    if input_path.is_file():
        run_on_image(detector, input_path, run_dir)

    elif input_path.is_dir():
        run_on_folder(detector, input_path, run_dir)

    else:
        print(f"Invalid path: {input_path}")
        sys.exit(1)


def run_on_image(detector: FishDetector, image_path: Path, run_dir: str) -> None:
    result = detector.detect(str(image_path), save_dir=run_dir)

    print(f"\nImage: {image_path.name}")
    print(json.dumps(result, indent=2))

    tracker_ready = detector.detections_for_tracker(result["detections"])
    print("Tracker format:")
    print(json.dumps(tracker_ready, indent=2))


def run_on_folder(detector: FishDetector, folder_path: Path, run_dir: str) -> None:
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in image_extensions])

    print(f"\nFound {len(images)} images in {folder_path}\n")

    for i, image_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] Processing {image_path.name}")
        result = detector.detect(str(image_path), save_dir=run_dir)
        print(f"  -> detections: {result['num_detections']}")


if __name__ == "__main__":
    main()