# Bridge till home_page UI

import streamlit as st

from services.home_manager import HomeManager
from services.review_service import ReviewService
from services.session_service import SessionService

from src.vision.track.tracker import FishTracker
from src.vision.count.counter import LineCounter, CountConfig


class HomeService:
    def __init__(self):
        self.review_service = ReviewService()
        self.manager = self._get_or_create_manager()

    def _get_or_create_manager(self) -> HomeManager:
        if "home_manager" not in st.session_state:
            tracker = FishTracker(weights_path="outputs/weights/baseline_best.pt")

            counter = LineCounter(
                CountConfig(
                    line_position=300,
                    axis="y",
                    line_margin=20.0,
                    min_positions=2,
                    max_missing_frames=30,
                    direction="positive",
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
        return {
            "trip_name": st.session_state.get("trip_name", "Tur_2026_03_14"),
            "catch_id": st.session_state.get("catch_id", "Okt_003"),
            "session_running": self.manager.is_running,
            "total_count": self.manager.get_total_count(),
            "estimated_weight_kg": 0,
            "species": self.manager.get_species_summary(),
            "status": {
                "review_pending": self.review_service.get_pending_count(),
            },
        }

    def start(self) -> None:
        self.manager.start()

    def step(self) -> None:
        self.manager.step()

    def stop(self) -> None:
        self.manager.stop()

    def is_running(self) -> bool:
        return self.manager.is_running