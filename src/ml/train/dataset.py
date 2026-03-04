#Defines how our training dataset is loaded and split (train/val/test). 
#Keeps data handling separate from training logic.
#placeholder, write the actual path when data available
from pathlib import Path

def assert_data_yaml_exists(data_yaml: str) -> None:
    if not data_yaml:
        raise ValueError("data_yaml is empty. Provide a path to Ultralytics data.yaml when you have data.")
    p = Path(data_yaml)
    if not p.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")