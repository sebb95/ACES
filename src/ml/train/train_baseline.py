# Entry point for å trena modell på annotert data.

import shutil
from pathlib import Path

from .config import TrainConfig
from .model import create_model
from .dataset import validate_data_yaml, dataset_statistics


def main():
    cfg = TrainConfig()

    # Validere datasett
    data_info = validate_data_yaml(cfg.data_yaml)
    dataset_statistics(data_info)

    # Danne output mapper
    cfg.runs_dir().mkdir(parents=True, exist_ok=True)
    cfg.weights_dir().mkdir(parents=True, exist_ok=True)

    run_name = cfg.resolved_run_name()

    # Modell instance
    model = create_model(cfg)

    # Starta träning
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

    # Spara Ultralitics statistics og vikter
    run_dir = Path(model.trainer.save_dir)
    best_pt = run_dir / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"Expected best weights not found: {best_pt}")

    out_best = cfg.weights_dir() / "baseline_best.pt"
    shutil.copy2(best_pt, out_best)

    print(f"\n[OK] Best weights saved to: {out_best}")


if __name__ == "__main__":
    main()