from pathlib import Path
from src.vision.active_learning.new_species_queue import get_full_retrain_status


MIN_SAMPLES_PER_NEW_CLASS = 100


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]

    status = get_full_retrain_status(
        base_dir=base_dir,
        min_samples=MIN_SAMPLES_PER_NEW_CLASS,
    )

    print("=== New species queue status ===")

    if not status["counts"]:
        print("No samples found in new_species_queue.")
        return

    for class_id, count in sorted(status["counts"].items()):
        ready = class_id in status["ready_classes"]
        label = "READY" if ready else "WAITING"
        print(f"class_id={class_id}: {count}/{status['min_samples']} samples - {label}")

    if not status["has_ready_classes"]:
        print("\nNo class has enough samples for full retraining yet.")
        return

    print("\nClasses ready for full retraining:")
    for class_id, count in sorted(status["ready_classes"].items()):
        print(f"- class_id={class_id}: {count} samples")

    # TODO: call full retraining pipeline here


if __name__ == "__main__":
    main()