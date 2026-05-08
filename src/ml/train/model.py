#importerer modell og danner modell instance

from ultralytics import YOLO
from pathlib import Path
from .config import TrainConfig, InferConfig

def create_model(cfg: TrainConfig) -> YOLO:
    # baseline: laste pretrained modell checkpoint
    return YOLO(cfg.base_weights)

def load_model(weights_path: str) -> YOLO:
    wp = Path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return YOLO(str(wp))