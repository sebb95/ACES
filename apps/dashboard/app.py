#kommand till å kjøre fra bash: streamlit run apps/dashboard/app.py

"""
Hovedinngang for ACES Streamlit-dashboard.

Denne filen:
- setter opp importstier for prosjektet
- konfigurerer layout og visning i Streamlit
- kjører oppstartssjekker (f.eks. treningsstatus)
- renderer toppnavigasjonen
- ruter til riktig side i dashboardet
"""

import sys
from pathlib import Path

import logging
import warnings

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").disabled = True
warnings.filterwarnings("ignore", message="missing ScriptRunContext")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import streamlit as st

from components.top_nav import render_top_nav
from pages.history_page import render_history_page
from pages.home_page import render_home_page
from pages.review_page import render_review_page
from pages.settings_page import render_settings_page
from pathlib import Path
from services.training_service import TrainingService

        
        
        
def main():


    st.set_page_config(
        page_title="ACES Dashboard",
        page_icon="🎣",
        layout="wide",
    )

    if "training_startup_checked" not in st.session_state:
        training_service = TrainingService()
        training_service.recover_if_stuck()
        training_service.maybe_run_scheduled_training()
        st.session_state["training_startup_checked"] = True

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            max-width: 100%;
        }

        div[data-testid="stHorizontalBlock"] > div {
            width: 100%;
        }

        .stButton > button {
            height: 3rem;
            font-size: 1rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "home"

    render_top_nav()
    st.write("")

    page = st.container()

    with page:
        current_page = st.session_state["current_page"]

        if current_page == "home":
            render_home_page()
        elif current_page == "review":
            render_review_page()
        elif current_page == "history":
            render_history_page()
        elif current_page == "settings":
            render_settings_page()
  
    

if __name__ == "__main__":
    main()