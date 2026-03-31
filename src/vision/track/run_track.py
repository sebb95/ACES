# run the file: python -m src.vision.track.run_track data/sample/val/images/

import json
import sys
from pathlib import Path

from src.vision.track.tracker import FishTracker


def create_run_dir(base_dir: str = "outputs/runs/track") -> str:
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
        print("Usage: python -m src.vision.track.run_track <image_folder>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Invalid folder path: {folder_path}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        [p for p in folder_path.iterdir() if p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        print(f"No images found in folder: {folder_path}")
        sys.exit(1)

    tracker = FishTracker(
        weights_path="outputs/weights/baseline_best.pt",
        conf=0.5,
        tracker_cfg="bytetrack.yaml",
    )

    run_dir = create_run_dir()
    print(f"Saving tracking outputs to: {run_dir}")
    print(f"Found {len(image_paths)} images in {folder_path}\n")

    for frame_idx, image_path in enumerate(image_paths, start=1):
        print(f"Frame {frame_idx}: {image_path.name}")

        track_result = tracker.update(
            image_path=str(image_path),
            save_dir=run_dir,
        )

        frame_output = {
            "frame_index": frame_idx,
            "image_name": image_path.name,
            "num_tracks": track_result["num_tracks"],
            "tracked_objects": track_result["tracked_objects"],
        }

        print(json.dumps(frame_output, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    main()