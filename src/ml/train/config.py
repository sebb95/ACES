#Handles reading model training configuration (image size, epochs, batch size, dataset path, etc).
#Prevents hardcoding parameters inside training script.


from dataclasses import dataclass
from pathlib import Path
import torch
import time


@dataclass(frozen=True)
class TrainConfig:
    # Model
    base_weights: str = "yolo11s-seg.pt"   # baseline segmentation model
    task: str = "segment" 

    # Data
    data_yaml: str = "data/sample/dataset.yaml" #point to dataset yaml file

    # Training hyperparameters
    epochs: int = 1 #adjust as needed
    imgsz: int = 640
    batch: int = 8 #reduced for smaller hardware
    workers: int = 2 #lighter to avoid multiprocessing
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    project_dir: str = "outputs"
    run_name: str = ""

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = time.strftime("%Y%m%d-%H%M%S")
        model_name = self.base_weights.replace(".pt", "")
        return f"baseline_{self.task}_{model_name}_{ts}"

    def runs_dir(self) -> Path:
        return Path(self.project_dir) / "runs"

    def weights_dir(self) -> Path:
        return Path(self.project_dir) / "weights"


@dataclass(frozen=True)
class InferConfig:
    # Inference should use trained weights by default
    weights_path: str = "outputs/weights/baseline_best.pt"
    source: str = "data/sample/val/images" #default sample destination, replace with actual directory with frames

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    save: bool = True
    save_txt: bool = False
    project_dir: str = "outputs"
    run_name: str = "baseline_infer"