#pipline testing for counter, run from bash: python -m src.vision.count.run_count

from __future__ import annotations

from pathlib import Path
import csv

from src.vision.track.tracker import FishTracker
from src.vision.count import CountConfig, LineCounter


def get_image_paths(frames_dir: str) -> list[Path]:
    frames_path = Path(frames_dir)

    if not frames_path.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_path}")

    image_paths = sorted(
        [
            p for p in frames_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not image_paths:
        raise ValueError(f"No image files found in: {frames_path}")

    return image_paths


def main() -> None:
    weights_path = "outputs/weights/baseline_best.pt"
    frames_dir = "data/sample/val/images"
    track_save_dir = "outputs/runs/track"
    csv_path = "outputs/runs/count/count_results.csv"

    tracker = FishTracker(
        weights_path=weights_path,
        conf=0.25,
        tracker_cfg="bytetrack.yaml",
    )

    counter = LineCounter(
        CountConfig(
            line_position=300,   # adjust after looking at images (eg width=600 => middle=300)
            axis="x",            # left-right movement
            line_margin=20,
            min_positions=2,
            max_missing_frames=30,
            direction="any",
        )
    )

    image_paths = get_image_paths(frames_dir)

    csv_output_path = Path(csv_path)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "image_name",
            "num_tracks",
            "new_counts",
            "total_count",
        ])

        for frame_index, image_path in enumerate(image_paths):
            track_result = tracker.update(
                image_path=str(image_path),
                save_dir=track_save_dir,
            )

            new_counts = counter.update(
                tracked_objects=track_result["tracked_objects"],
                frame_index=frame_index,
            )

            total_count = counter.get_total_count()

            print(
                f"frame={frame_index:03d} | "
                f"image={image_path.name} | "
                f"tracks={track_result['num_tracks']} | "
                f"new_counts={new_counts} | "
                f"total={total_count}"
            )

            writer.writerow([
                frame_index,
                image_path.name,
                track_result["num_tracks"],
                new_counts,
                total_count,
            ])

    print("\nDone.")
    print(f"Final total count: {counter.get_total_count()}")
    print(f"CSV saved to: {csv_output_path}")


if __name__ == "__main__":
    main()