import copy
import json
from pathlib import Path
from typing import Any, Dict


class SettingsService:
    CONFIG_PATH = Path("configs/runtime_config.json")

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model": {
            "selected_model": "best.pt",
        },
        "input": {
            "input_type": "image_folder",
            "dataset_path": "data/sample",
            "video_path": "data/input/video.mp4",
            "frame_output_path": "data/processed/frames/current_run",
        },
        "camera": {
            "fps": 30,
        },
        "species": {
            "weights_kg": {
                "Torsk": 2.4,
                "Sei": 2.0,
                "Hyse": 1.2,
                "Lange": 3.0,
                "Brosme": 2.5,
                "Kveite": 5.0,
                "Flyndre": 1.0,
                "Uer": 0.8,
                "Lyr": 1.5,
                "Breiflabb": 4.0,
            }
        },
        "active_learning": {
            "review_min_confidence": 0.30,
            "review_max_confidence": 0.80,
        },
        "training": {
            "status": "idle",
            "selected_model": "best.pt",
            "dataset_path": "data/training_reviewed",
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