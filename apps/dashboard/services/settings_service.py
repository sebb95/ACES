import copy
import json
import re
from pathlib import Path
from typing import Any, Dict


class SettingsService:
    CONFIG_PATH = Path("configs/runtime_config.json")
    SPECIES_PATH = Path("src/common/species.py")

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model": {
            "selected_model": "best.pt",
        },
        "input": {
            "input_type": "video_file",
            "dataset_path": "data/sample",
            "video_path": "data/input/video.mp4",
        },
        "camera": {
            "fps": 30,
        },
        "species": {
            "weights_kg": {
                "Breiflab": 4.0,
                "Brosme": 2.5,
                "Flyndre": 1.0,
                "Hyse": 1.2,
                "Kveite": 5.0,
                "Lange": 3.0,
                "Lyr": 1.5,
                "Sei": 2.0,
                "Torsk": 2.4,
                "Uer": 0.8,
                "Bifangst": 1.0,
                "Ukjent": 1.0,
            }
        },
        "active_learning": {
            "review_min_confidence": 0.30,
            "review_max_confidence": 0.80,
        },
        "training": {
            "status": "idle",
            "night_training_enabled": False,
            "night_training_time": "03:00",
        },
    }

    def __init__(self) -> None:
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        if not self.CONFIG_PATH.exists():
            self._save(copy.deepcopy(self.DEFAULT_CONFIG))
            return

        config = self._load_raw()
        migrated = self._merge_with_defaults(config)

        if migrated != config:
            self._save(migrated)

    def _load_raw(self) -> Dict[str, Any]:
        with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load(self) -> Dict[str, Any]:
        self._ensure_exists()
        return self._load_raw()

    def _save(self, config: Dict[str, Any]) -> None:
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge existing config with defaults to support migration
        when new fields are added.
        """
        merged = copy.deepcopy(self.DEFAULT_CONFIG)

        def deep_update(base: dict, updates: dict):
            for k, v in updates.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    deep_update(base[k], v)
                else:
                    base[k] = v

        deep_update(merged, config)

        species = merged.get("species", {})

        if "weights_kg" not in species:
            merged["species"] = copy.deepcopy(self.DEFAULT_CONFIG["species"])

            merged["species"]["weights_kg"]["Torsk"] = species.get("torsk_weight", 2.4)
            merged["species"]["weights_kg"]["Sei"] = species.get("sei_weight", 2.0)
        return merged

    def get(self) -> Dict[str, Any]:
        return self._load()

    def update(self, new_config: Dict[str, Any]) -> None:
        self._save(new_config)

    def reset(self) -> None:
        self._save(copy.deepcopy(self.DEFAULT_CONFIG))

    def add_species(self, species_name: str) -> int:
        species_name = species_name.strip()

        if not species_name:
            raise ValueError("Species name cannot be empty.")

        content = self.SPECIES_PATH.read_text(encoding="utf-8")

        if re.search(rf':\s*"{re.escape(species_name)}"', content):
            raise ValueError(f"Species already exists: {species_name}")

        matches = re.findall(
            r'^\s*(\d+):\s*"([^"]+)"',
            content,
            flags=re.MULTILINE,
        )

        if not matches:
            raise ValueError("Could not find CLASS_NAMES entries in species.py.")

        max_id = max(int(class_id) for class_id, _ in matches)
        new_id = max_id + 1

        insert_line = f'    {new_id}: "{species_name}",\n'

        content = content.replace(
            "}\n\nNAME_TO_CLASS_ID",
            insert_line + "}\n\nNAME_TO_CLASS_ID",
        )

        self.SPECIES_PATH.write_text(content, encoding="utf-8")

        return new_id