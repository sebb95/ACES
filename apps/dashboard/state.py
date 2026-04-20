#shared dashboard/session state

import streamlit as st


ACTIVE_SESSION_KEY = "active_session"
HOME_MANAGER_KEY = "home_manager"


def get_active_session() -> dict | None:
    return st.session_state.get(ACTIVE_SESSION_KEY)


def set_active_session(session_data: dict) -> None:
    st.session_state[ACTIVE_SESSION_KEY] = session_data


def clear_active_session() -> None:
    st.session_state.pop(ACTIVE_SESSION_KEY, None)


def has_active_session() -> bool:
    return ACTIVE_SESSION_KEY in st.session_state


def get_home_manager():
    return st.session_state.get(HOME_MANAGER_KEY)


def set_home_manager(manager) -> None:
    st.session_state[HOME_MANAGER_KEY] = manager


def has_home_manager() -> bool:
    return HOME_MANAGER_KEY in st.session_state