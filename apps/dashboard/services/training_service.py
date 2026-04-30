import threading
import logging
import os
from datetime import datetime
from pathlib import Path
from src.vision.active_learning.night_operations import NightOperations
from services.settings_service import SettingsService


class TrainingService:
    """
    Tjenestelag for håndtering av modelltrening i ACES.

    Klassen fungerer som et kontrollpunkt mellom UI, konfigurasjon og
    treningslogikk (NightOperations). Den er ansvarlig for å starte,
    overvåke og planlegge trening, samt oppdatere treningsstatus i
    runtime_config.

    Ansvar:
    - starte trening i bakgrunnstråd (ikke-blokkerende for UI)
    - oppdatere treningsstatus (idle / running / ready / failed)
    - trigge nattkjøring basert på konfigurasjon (scheduled training)
    - håndtere enkel feildeteksjon (f.eks. stuck "running"-status)
    - koble treningslogikk til godkjente data fra review

    Merk:
    - Selve treningslogikken ligger i NightOperations
    - Klassen håndterer kun orkestrering og status
    """
    def __init__(self):
        self.settings_service = SettingsService()

    def _set_status(self, status: str) -> None:
        config = self.settings_service.get()
        config.setdefault("training", {})
        config["training"]["status"] = status
        config["training"]["last_updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.settings_service.update(config)

    def run_training(self):
        self._set_status("running")
        
        # Starter treningen i en egen tråd
        thread = threading.Thread(target=self._training_task, daemon=True)
        thread.start()

    def _training_task(self):
        
        os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"
        logging.getLogger("streamlit").setLevel(logging.ERROR)
        logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.CRITICAL)
        
        try:
            trainer = NightOperations(
                current_model_path=Path("outputs/weights/best.pt"),
                baseline_model_path=Path("outputs/weights/best_backup.pt"),
                approved_data_dir=Path("data/training_reviewed"),
            )
            trainer.run()
            self._set_status("ready")

        except Exception:
            self._set_status("failed")
            raise

    def get_status(self) -> str:
        config = self.settings_service.get()
        return config.get("training", {}).get("status", "idle")
    
    def maybe_run_scheduled_training(self) -> None:
        config = self.settings_service.get()
        training = config.get("training", {})

        if not training.get("night_training_enabled", False):
            return

        if training.get("status") == "running":
            return

        scheduled_time = training.get("night_training_time", "03:00")
        now_str = datetime.now().strftime("%H:%M")

        last_run_date = training.get("last_scheduled_run_date")
        today = datetime.now().strftime("%Y-%m-%d")

        if now_str == scheduled_time and last_run_date != today:
            training["last_scheduled_run_date"] = today
            config["training"] = training
            self.settings_service.update(config)

            self.run_training()

    def recover_if_stuck(self) -> None:
        config = self.settings_service.get()
        training = config.get("training", {})

        status = training.get("status")
        last_update = training.get("last_updated_at")

        if status != "running" or not last_update:
            return

        last_dt = datetime.fromisoformat(last_update)
        now = datetime.now()

        # If "running" but no update for e.g. 10 minutes → assume crash
        if (now - last_dt).total_seconds() > 600:
            training["status"] = "failed"
            config["training"] = training
            self.settings_service.update(config)