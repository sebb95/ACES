# Dataset validation utilities for YOLO training
# Ensures dataset.yaml exists and dataset structure is correct
# before starting a training run.

from pathlib import Path
import yaml


def validate_data_yaml(data_yaml: str) -> dict:
    """
    Validate dataset.yaml and return parsed configuration.
    """

    yaml_path = Path(data_yaml)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    required_keys = ["path", "train", "val", "names"]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in dataset.yaml: '{key}'")

    base_path = Path(data["path"])
    train_images = base_path / data["train"]
    val_images = base_path / data["val"]

    if not train_images.exists():
        raise FileNotFoundError(f"Train images folder not found: {train_images}")

    if not val_images.exists():
        raise FileNotFoundError(f"Val images folder not found: {val_images}")

    train_labels = Path(str(train_images).replace("images", "labels"))
    val_labels = Path(str(val_images).replace("images", "labels"))

    if not train_labels.exists():
        raise FileNotFoundError(f"Train labels folder not found: {train_labels}")

    if not val_labels.exists():
        raise FileNotFoundError(f"Val labels folder not found: {val_labels}")

    return data


def dataset_statistics(data: dict) -> None:
    """
    Print simple dataset statistics for verification.
    """

    base_path = Path(data["path"])

    train_images = base_path / data["train"]
    val_images = base_path / data["val"]

    train_count = len(list(train_images.glob("*")))
    val_count = len(list(val_images.glob("*")))

    print("\nDataset summary")
    print("----------------")
    print(f"Classes: {len(data['names'])}")
    print(f"Class names: {data['names']}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")