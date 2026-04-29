import copy
import json
import re
import importlib
import src.common.species as species_module
from pathlib import Path
from typing import Any, Dict


class SettingsService:
    """
    Håndterer lasting, lagring og migrering av runtime-innstillinger.

    SettingsService bruker configs/runtime_config.json som persistent
    konfigurasjon for dashboardet, blant annet valgt modell, input-kilde,
    active learning-grenser, treningsstatus og artsvekter.

    Klassen håndterer også oppdatering av species.py når nye arter legges til
    via Innstillinger. Etterpå synkroniseres runtime_config slik at nye arter
    får standard snittvekt og blir tilgjengelige i UI uten restart.
    """
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
        config = self._load_raw()
        config = self._sync_species_weights(config)
        self._save(config)
        return config

    def _save(self, config: Dict[str, Any]) -> None:
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Slår eksisterende config sammen med DEFAULT_CONFIG.
        Brukes som enkel migrering når nye felt legges til i systemet,
        slik at gamle runtime_config-filer fortsatt fungerer.
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

        config = self._load_raw()
        config.setdefault("species", {})
        config["species"].setdefault("weights_kg", {})
        config["species"]["weights_kg"].setdefault(species_name, 1.0)
        self._save(config)

        self._reload_species()

        return new_id
    
    def _reload_species(self):
        """
        Leser species.py på nytt slik at arter lagt til via Innstillinger
        blir tilgjengelige uten restart av Streamlit.
        """
        return importlib.reload(species_module)

    def _sync_species_weights(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sørger for at artsvekter i runtime_config samsvarer med species.py.

        Nye arter får standard snittvekt 1.0 kg. Arter som er fjernet fra
        species.py fjernes også fra config, slik at gamle testarter ikke blir
        liggende igjen i UI.
        """
        species = self._reload_species()

        config.setdefault("species", {})
        config["species"].setdefault("weights_kg", {})

        weights = config["species"]["weights_kg"]

        for class_id in sorted(species.CLASS_NAMES):
            species_name = species.CLASS_NAMES[class_id]

            if species_name not in weights:
                weights[species_name] = 1.0

        valid_species_names = set(species.CLASS_NAMES.values())

        for species_name in list(weights.keys()):
            if species_name not in valid_species_names:
                del weights[species_name]

        return config