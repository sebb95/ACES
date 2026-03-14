# Entry point for training a fresh model from labeled data.
# This is the file we run to start baseline training.

import shutil
from pathlib import Path

from .config import TrainConfig
from .model import create_model
from .dataset import validate_data_yaml, dataset_statistics


def main():
    cfg = TrainConfig()

    # Validate dataset before training starts
    data_info = validate_data_yaml(cfg.data_yaml)
    dataset_statistics(data_info)

    # Create output folders
    cfg.runs_dir().mkdir(parents=True, exist_ok=True)
    cfg.weights_dir().mkdir(parents=True, exist_ok=True)

    run_name = cfg.resolved_run_name()

    # Create model
    model = create_model(cfg)

    # Start training
    model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        workers=cfg.workers,
        device=cfg.device,
        project=str(cfg.runs_dir().resolve()),
        name=run_name,
    )

    # Ask Ultralytics where it actually saved the run
    run_dir = Path(model.trainer.save_dir)
    best_pt = run_dir / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"Expected best weights not found: {best_pt}")

    out_best = cfg.weights_dir() / "baseline_best.pt"
    shutil.copy2(best_pt, out_best)

    print(f"\n[OK] Best weights saved to: {out_best}")


if __name__ == "__main__":
    main()