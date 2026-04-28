from pathlib import Path
import yaml
from src.common.species import CLASS_NAMES


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _get_name_to_id() -> dict[str, int]:
    """Mapper klassenavn → ID"""
    return {name: int(class_id) for class_id, name in CLASS_NAMES.items()}


def _used_class_ids(label_dirs: list[Path]) -> set[int]:
    used = set()
    name_to_id = _get_name_to_id()

    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                first = parts[0]
                if first in name_to_id:
                    used.add(name_to_id[first])
                else:
                    try:
                        used.add(int(first))
                    except ValueError:
                        pass
    return used


def get_class_names_from_species_py() -> dict[int, str]:
    return {
        int(class_id): str(name)
        for class_id, name in CLASS_NAMES.items()
    }


def validate_labels_against_species_py(label_dirs: list[Path]) -> None:
    class_names = get_class_names_from_species_py()
    used_ids = _used_class_ids(label_dirs)
    missing_ids = sorted(
        class_id for class_id in used_ids
        if class_id not in class_names
    )
    if missing_ids:
        raise ValueError(
            f"Training labels contain class IDs not defined in src/common/species.py: {missing_ids}"
        )


def write_master_dataset_yaml(
    master_yaml_path: Path,
    master_root: Path,
    label_dirs: list[Path],
) -> dict[int, str]:
    validate_labels_against_species_py(label_dirs)
    class_names = get_class_names_from_species_py()

    yaml_data = {
        "path": str(master_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {
            class_id: class_names[class_id]
            for class_id in sorted(class_names)
        },
    }
    _save_yaml(master_yaml_path, yaml_data)
    return yaml_data["names"]


def write_night_training_yaml(
    yaml_path: Path,
    run_dir: Path,
    train_paths_file: Path,
    master_val_img_dir: Path,
    dynamic_val_img_dir: Path,
    class_names: dict[int, str],
) -> None:
    yaml_data = {
        "path": str(run_dir.resolve()),
        "train": str(train_paths_file.resolve()),
        "val": [
            str(master_val_img_dir.resolve()),
            str(dynamic_val_img_dir.resolve()),
        ],
        "names": {
            class_id: class_names[class_id]
            for class_id in sorted(class_names)
        },
    }
    _save_yaml(yaml_path, yaml_data)


def write_dynamic_val_yaml(
    yaml_path: Path,
    run_dir: Path,
    train_paths_file: Path,
    dynamic_val_img_dir: Path,
    class_names: dict[int, str],
) -> None:
    yaml_data = {
        "path": str(run_dir.resolve()),
        "train": str(train_paths_file.resolve()),
        "val": str(dynamic_val_img_dir.resolve()),
        "names": {
            class_id: class_names[class_id]
            for class_id in sorted(class_names)
        },
    }
    _save_yaml(yaml_path, yaml_data)