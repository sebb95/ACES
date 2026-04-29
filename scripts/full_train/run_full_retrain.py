import sys
from pathlib import Path

# Sørg for at Python finner src-mappen
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.vision.active_learning.new_species_queue import get_full_retrain_status
from src.vision.active_learning.full_retrain_operations import FullRetrainOperations

MIN_SAMPLES_PER_NEW_CLASS = 100

def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]

    # Tanyas logikk for å telle opp status
    status = get_full_retrain_status(
        base_dir=base_dir,
        min_samples=MIN_SAMPLES_PER_NEW_CLASS,
    )

    print("\n" + "=" * 40)
    print("=== NEW SPECIES QUEUE STATUS ===")
    print("=" * 40)

    if not status["counts"]:
        print("No samples found in new_species_queue.")
        return

    for class_id, count in sorted(status["counts"].items()):
        ready = class_id in status["ready_classes"]
        label = "✅ READY" if ready else "⏳ WAITING"
        print(f"class_id={class_id}: {count}/{status['min_samples']} samples - {label}")

    if not status["has_ready_classes"]:
        print("\nNo class has enough samples for full retraining yet.")
        return

    print("\nClasses ready for full retraining:")
    ready_class_ids = []
    for class_id, count in sorted(status["ready_classes"].items()):
        print(f"- class_id={class_id}: {count} samples")
        ready_class_ids.append(int(class_id))

    # --- HER KOBLER VI PÅ VÅR LOGIKK ---
    print("\n" + "=" * 40)
    print("🚀 STARTER FULL RETRAINING PIPELINE")
    print("=" * 40)
    
    try:
        pipeline = FullRetrainOperations()
        
        # Vi mater Tanyas ferdig-filtrerte liste direkte inn i vår pipeline!
        pipeline.run(ready_classes=ready_class_ids)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Full retrening avbrutt av bruker (Ctrl+C).")
        print("Ingen data er flyttet, Master-datasettet er urørt.")
    except Exception as e:
        print(f"\n\n❌ EN KRITISK FEIL OPPSTOD:\n{e}")


if __name__ == "__main__":
    main()