#Handles reading model training configuration (image size, epochs, batch size, dataset path, etc).
#Prevents hardcoding parameters inside training script.

from dataclasses import dataclass
from pathlib import Path
import torch
import time

@dataclass(frozen=True)
class TrainConfig:
    # Model
    base_weights: str = "yolov8n.pt"   # baseline choice
    task: str = "detect"              # "detect" or "segment"

    # Data (Ultralytics expects a data.yaml when training)
    data_yaml: str = ""               # TODO: set later (e.g. "data/fish.yaml")

    # Training hyperparams
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    project_dir: str = "outputs"
    run_name: str = ""                # auto if empty

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"baseline_{self.task}_{ts}"

    def runs_dir(self) -> Path:
        return Path(self.project_dir) / "runs"

    def weights_dir(self) -> Path:
        return Path(self.project_dir) / "weights"


@dataclass(frozen=True)
class InferConfig:
    weights_path: str = ""            # e.g. outputs/weights/baseline_best.pt
    source: str = ""                  # folder, image, video, stream

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    save: bool = True
    save_txt: bool = False
    project_dir: str = "outputs"
    run_name: str = ""                # auto if empty