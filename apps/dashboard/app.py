#run from bash: streamlit run apps/dashboard/app.py
import sys
from pathlib import Path

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


st.set_page_config(
    page_title="ACES Dashboard",
    page_icon="🎣",
    layout="wide",
)

training_service = TrainingService()
training_service.recover_if_stuck()
TrainingService().maybe_run_scheduled_training()

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