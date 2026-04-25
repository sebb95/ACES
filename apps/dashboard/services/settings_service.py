import json
from pathlib import Path
from typing import Any, Dict


class SettingsService:
    CONFIG_PATH = Path("configs/runtime_config.json")

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model": {
            "selected_model": "baseline_best.pt"
        },
        "input": {
            "dataset_path": "data/sample"
        },
        "camera": {
            "fps": 30
        },
        "species": {
            "torsk_weight": 2.4,
            "sei_weight": 2.0,
            "bifangst_weight": 2.2
        },
        "active_learning": {
            "uncertainty_threshold": 0.6
        }
    }

    def __init__(self) -> None:
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        if not self.CONFIG_PATH.exists():
            self._save(self.DEFAULT_CONFIG)

    def _load(self) -> Dict[str, Any]:
        self._ensure_exists()
        with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, config: Dict[str, Any]) -> None:
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def get(self) -> Dict[str, Any]:
        return self._load()

    def update(self, new_config: Dict[str, Any]) -> None:
        self._save(new_config)

    def reset(self) -> None:
        self._save(self.DEFAULT_CONFIG.copy())