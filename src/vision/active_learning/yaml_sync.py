"""
YAML-generator for trenings- og valideringsdatasett i ACES.

Denne modulen genererer YOLO-kompatible YAML-filer dynamisk basert på:
- gjeldende datastruktur (master, training_reviewed, dynamic_val)
- artsdefinisjoner i species.py

Formål:
- unngå statiske dataset.yaml-filer
- støtte dynamisk utvidelse av arter (class_id)
- sikre konsistens mellom labels og artsregister

Viktig:
- species.py er "source of truth" for class_id → artsnavn
- alle labels må være definert i species.py (valideres eksplisitt)
- YAML genereres først når dataset er klart (night training eller full retrening)

Denne modulen inneholder kun konfigurasjonsgenerering.
Den utfører ikke trening selv.
"""

from pathlib import Path
import yaml
from src.common.species import CLASS_NAMES


def _save_yaml(path: Path, data: dict) -> None:
    """
    Skriver YAML-data til fil på en sikker måte.
    Oppretter nødvendige mapper automatisk og lagrer YAML med korrekt
    encoding og struktur.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _get_name_to_id() -> dict[str, int]:
    """Mapper klassenavn → ID"""
    return {name: int(class_id) for class_id, name in CLASS_NAMES.items()}


def _used_class_ids(label_dirs: list[Path]) -> set[int]:
    """
    Finner hvilke class_id-er som faktisk brukes i et sett med label-mapper.
    Leser alle YOLO-label-filer og samler unike class_id-er.
    Støtter både numeriske ID-er og navn (via mapping fra species.py).
    """
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
    """
    Returnerer mapping fra class_id til artsnavn fra species.py.
    Dette er grunnlaget for 'names' i YOLO YAML-konfigurasjon.
    """
    return {
        int(class_id): str(name)
        for class_id, name in CLASS_NAMES.items()
    }


def validate_labels_against_species_py(label_dirs: list[Path]) -> None:
    """
    Validerer at alle labels samsvarer med arter definert i species.py.
    Kaster ValueError dersom labels inneholder class_id-er som ikke finnes
    i artsregisteret.
    Hindrer:
    - inkonsistent datasett
    - krasj under trening
    - feil mapping mellom ID og artsnavn
    """
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
    """
    Genererer YAML for master-datasettet.
    YAML peker til:
    - train/images
    - val/images
    og inkluderer alle klasser definert i species.py.
    Returnerer:
        dict[class_id → artsnavn], brukt videre i pipeline.
    """
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
    """
    Genererer YAML for nattlig finjustering (night training).
    Bruker:
    - spesifikk liste av treningsbilder (train_paths.txt)
    - validering mot både master val og dynamic val

    Dette gir:
    - stabil ytelse på eksisterende arter
    - samtidig evaluering på nye data
    """
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
    """
    Genererer YAML for evaluering på kun nye data (dynamic validation).
    Brukes til å måle om modellen faktisk lærer fra nye eksempler,
    uavhengig av eksisterende (master) datasett.

    Viktig for quality gate:
    - sikrer at modellen forbedrer seg på nye data
    """
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