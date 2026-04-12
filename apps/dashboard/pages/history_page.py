#

import streamlit as st
from services.history_service import get_history_data


def _render_trip_summary(trip: dict):
    st.markdown(
        f"""
        <div style="
            background:#cfcfcf;
            padding:1rem;
            margin-bottom:1rem;
            font-size:1.4rem;
            font-weight:600;
        ">
            TUR: "{trip["name"]}"
            &nbsp;&nbsp; Start: {trip["start"]}
            &nbsp;&nbsp; Slutt: {trip["end"]}
            &nbsp;&nbsp; <b>TOTAL ANTALL:</b> {trip["total_count"]} st
            &nbsp;&nbsp; <b>ESTIMERT VEKT:</b> {trip["total_weight"]} kg
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header():
    col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1, 1, 1, 1.5, 0.5])

    with col1:
        st.markdown("**Økt ID**")
    with col2:
        st.markdown("**Start**")
    with col3:
        st.markdown("**Slutt**")
    with col4:
        st.markdown("**Antall**")
    with col5:
        st.markdown("**Estimert vekt**")
    with col6:
        st.markdown("")


def _render_session_row(session: dict):
    if "expanded_rows" not in st.session_state:
        st.session_state["expanded_rows"] = {}

    if session["id"] not in st.session_state["expanded_rows"]:
        st.session_state["expanded_rows"][session["id"]] = session["expanded"]

    expanded = st.session_state["expanded_rows"][session["id"]]

    col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1, 1, 1, 1.5, 0.5])

    with col1:
        st.markdown(f"**{session['id']}**")
    with col2:
        st.write(session["start"])
    with col3:
        st.write(session["end"])
    with col4:
        st.write(session["count"])
    with col5:
        st.write(f"{session['weight']} kg")
    with col6:
        if st.button("▲" if expanded else "▼", key=session["id"]):
            st.session_state["expanded_rows"][session["id"]] = not expanded

    # Expanded section
    if st.session_state["expanded_rows"][session["id"]]:
        st.markdown(
            """
            <div style="
                background:#e6e6e6;
                padding:0.8rem;
                margin-bottom:0.5rem;
            ">
            """,
            unsafe_allow_html=True,
        )

        species_text = " ".join(
            [
                f"<b>{s['name']}</b>: {s['count']} st ({s['weight']} kg)"
                for s in session["species"]
            ]
        )

        col_text, col_btn = st.columns([4, 1])

        with col_text:
            st.markdown(species_text, unsafe_allow_html=True)

        with col_btn:
            st.button("Export", key=f"export_{session['id']}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()


def render_history_page():
    data = get_history_data()

    _render_trip_summary(data["trip"])
    _render_header()

    st.divider()

    for session in data["sessions"]:
        _render_session_row(session)