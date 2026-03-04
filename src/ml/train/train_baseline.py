#Entry point for training a fresh model from labeled data. This is the file we run.

from pathlib import Path
import shutil

from .config import TrainConfig
from .model import create_model
from .dataset import assert_data_yaml_exists

def main():
    cfg = TrainConfig()

    # For now, training requires dataset yaml.
    assert_data_yaml_exists(cfg.data_yaml)

    run_name = cfg.resolved_run_name()
    (cfg.runs_dir()).mkdir(parents=True, exist_ok=True)
    (cfg.weights_dir()).mkdir(parents=True, exist_ok=True)

    model = create_model(cfg)

    results = model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        workers=cfg.workers,
        device=cfg.device,
        project=str(cfg.runs_dir()),
        name=run_name,
    )

    # Ultralytics writes best.pt here:
    # outputs/runs/<run_name>/weights/best.pt
    run_dir = cfg.runs_dir() / run_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Expected best weights not found: {best_pt}")

    out_best = cfg.weights_dir() / "baseline_best.pt"
    shutil.copy2(best_pt, out_best)
    print(f"[OK] Saved best weights to: {out_best}")

if __name__ == "__main__":
    main()