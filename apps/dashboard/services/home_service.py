import threading
import time
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx # LEGG TIL DENNE!

from services.home_manager import HomeManager
from services.review_service import ReviewService
from services.session_service import SessionService
from services.settings_service import SettingsService
from services.trip_service import TripService

from src.vision.track.tracker import FishTracker
from src.vision.count.counter import LineCounter, CountConfig


class HomeService:
    """
    Service-lag mellom Streamlit-siden og HomeManager.

    Klassen oppretter og holder på HomeManager i st.session_state, slik at
    tracker, counter og session-tilstand ikke nullstilles ved hver Streamlit-rerun.
    """
    def __init__(self):
        self.review_service = ReviewService()
        self.settings_service = SettingsService()
        self.manager = self._get_or_create_manager()
        self.trip_service = TripService()

    def _get_initial_weights_path(self) -> str:
        settings = self.settings_service.get()
        selected_model = settings.get("model", {}).get("selected_model", "production.engine")
        return f"outputs/weights/{selected_model}"

    def _get_or_create_manager(self) -> HomeManager:
        if "home_manager" not in st.session_state:
            tracker = FishTracker(weights_path=self._get_initial_weights_path())

            counter = LineCounter(
                CountConfig(
                        line_position=800.0,
                        axis="x",
                        line_margin=90.0,
                        min_positions=2,
                        max_missing_frames=30,
                        direction="any",
                    )
                )

            session_service = SessionService()

            st.session_state["home_manager"] = HomeManager(
                tracker=tracker,
                counter=counter,
                session_service=session_service,
            )

        return st.session_state["home_manager"]

    def get_home_page_data(self) -> dict:
        session = self.manager.session_service.get_active_session()
        weight_summary = self.manager.get_weight_summary()
        trip = self.get_active_trip()
        active_session = self.manager.session_service.get_active_session()

        return {
            "trip": trip,
            "trip_name": st.session_state.get("trip_name", "Tur_2026_03_14"),
            "catch_id": session.get("session_id", "Ingen aktiv økt") if session else "Ingen aktiv økt (trykk START)",
            "session_active": active_session is not None,
            "session_running": self.manager.is_running,

            "total_count": weight_summary["total_count"],
            "estimated_weight_kg": weight_summary["total_weight_kg"],

            "torsk": weight_summary["torsk"],
            "sei": weight_summary["sei"],
            "bifangst": weight_summary["bifangst"],
            "species_breakdown": weight_summary["species_breakdown"],

            "status": {
                "review_pending": self.review_service.get_pending_count(),
                "uncertain_count": session.get("uncertain_count", 0) if session else 0,
                "training_status": self.settings_service.get()
                .get("training", {})
                .get("status", "idle"),
            },
        }

    def start(self) -> None:
        """
        Starter telleøkten og oppretter en bakgrunnstråd for prosessering.

        Tråden gjør at videoen kan prosesseres uten at Streamlit-UI blokkeres.
        """
        self.manager.start()
        
        if not hasattr(self.manager, "processing_thread") or not self.manager.processing_thread.is_alive():
            self.manager.processing_thread = threading.Thread(target=self._run_loop, daemon=True)
            
            # KOBLE TRÅDEN TIL STREAMLIT:
            add_script_run_ctx(self.manager.processing_thread) 
            
            self.manager.processing_thread.start()

    def _run_loop(self) -> None:
        """
        Kjører prosessering i bakgrunnstråd så lenge manager er aktiv.

        Loopen ligger utenfor Streamlit sin vanlige rerun-syklus og skal derfor
        kun kalle backend-logikk, ikke skrive direkte til UI.
        """
        while self.manager.is_running:
            self.manager.step()
            time.sleep(0.001)
            # Valgfritt: time.sleep(0.001) for å forhindre at CPU-en låser seg helt 100%
            
        print("[THREAD] Video-prosessering er ferdig/stoppet.")

    # Vi trenger egentlig ikke kalle step() manuelt fra UI lenger, 
    # siden tråden gjør det automatisk, men vi lar den stå:
    def step(self) -> None:
        pass # Tråden håndterer nå dette!

    def stop(self) -> None:
        self.manager.stop()
    def is_running(self) -> bool:
        return self.manager.is_running
    
    def get_active_trip(self) -> dict:
        return self.trip_service.ensure_active_trip()
    
    def start_new_trip(self, name: str | None = None):
        self.trip_service.start_new_trip(name)

    def rename_trip(self, new_name: str):
        self.trip_service.rename_active_trip(new_name)

    def end_trip(self):
        self.trip_service.end_active_trip()