import streamlit as st

from services.history_service import HistoryService


def _ensure_expanded_state() -> None:
    if "expanded_trips" not in st.session_state:
        st.session_state["expanded_trips"] = {}

    if "expanded_sessions" not in st.session_state:
        st.session_state["expanded_sessions"] = {}


def _render_trip_header() -> None:
    cols = st.columns([1.7, 1, 1, 1, 1.2, 1, 1, 1.2, 0.7, 0.7])

    headers = [
        "Tur",
        "Start",
        "Slutt",
        "Antall",
        "Total kg",
        "Torsk kg",
        "Sei kg",
        "Bifangst kg",
        "Usikre",
        "",
    ]

    for col, header in zip(cols, headers):
        with col:
            st.markdown(f"**{header}**")


def _render_session_header() -> None:
    cols = st.columns([1.7, 1, 1, 1, 1.2, 1, 0.7])

    headers = [
        "Økt",
        "Start",
        "Slutt",
        "Antall",
        "Vekt kg",
        "Usikre",
        "",
    ]

    for col, header in zip(cols, headers):
        with col:
            st.markdown(f"**{header}**")


def _render_species_breakdown(session: dict) -> None:
    species = session.get("species", [])

    if not species:
        st.caption("Ingen artsfordeling lagret.")
        return

    for item in species:
        st.write(
            f"{item['name']}: {item['count']} st "
            f"({item['weight_kg']} kg)"
        )

    st.caption(
        f"Korreksjoner: {session.get('corrections', 0)} | "
        f"Review-elementer: {session.get('review_items_created', 0)}"
    )


def _render_session_row(session: dict) -> None:
    session_id = session["id"]

    if session_id not in st.session_state["expanded_sessions"]:
        st.session_state["expanded_sessions"][session_id] = False

    expanded = st.session_state["expanded_sessions"][session_id]

    cols = st.columns([1.7, 1, 1, 1, 1.2, 1, 0.7])

    with cols[0]:
        st.markdown(f"**{session_id}**")
    with cols[1]:
        st.write(session["start"])
    with cols[2]:
        st.write(session["end"])
    with cols[3]:
        st.write(session["count"])
    with cols[4]:
        st.write(f"{session['weight_kg']} kg")
    with cols[5]:
        st.write(session["uncertain_count"])
    with cols[6]:
        if st.button("▲" if expanded else "▼", key=f"session_{session_id}"):
            st.session_state["expanded_sessions"][session_id] = not expanded
            st.rerun()

    if st.session_state["expanded_sessions"][session_id]:
        st.markdown(
            """
            <div style="
                background:#e6e6e6;
                padding:0.8rem;
                margin-bottom:0.6rem;
            ">
            """,
            unsafe_allow_html=True,
        )
        _render_species_breakdown(session)
        st.markdown("</div>", unsafe_allow_html=True)


def _build_trip_export_text(trip: dict) -> str:
    return f"""TURRAPPORT

Tur: {trip["trip_name"]}
Start: {trip["started_at"]}
Slutt: {trip["ended_at"]}

Total antall: {trip["total_count"]} st
Total vekt: {trip["total_weight_kg"]} kg

Torsk: {trip["torsk"]["count"]} st / {trip["torsk"]["weight_kg"]} kg
Sei: {trip["sei"]["count"]} st / {trip["sei"]["weight_kg"]} kg
Bifangst: {trip["bifangst"]["count"]} st / {trip["bifangst"]["weight_kg"]} kg

Usikre til gjennomgang: {trip["uncertain_count"]}
Review-elementer opprettet: {trip["review_items_created"]}
Antall økter: {len(trip["sessions"])}
"""


def _render_trip_row(trip: dict) -> None:
    trip_id = trip["trip_id"]

    if trip_id not in st.session_state["expanded_trips"]:
        st.session_state["expanded_trips"][trip_id] = False

    expanded = st.session_state["expanded_trips"][trip_id]

    cols = st.columns([1.7, 1, 1, 1, 1.2, 1, 1, 1.2, 0.7, 0.7])

    with cols[0]:
        st.markdown(f"**{trip['trip_name']}**")
    with cols[1]:
        st.write(trip["started_at"])
    with cols[2]:
        st.write(trip["ended_at"])
    with cols[3]:
        st.write(trip["total_count"])
    with cols[4]:
        st.write(f"{trip['total_weight_kg']} kg")
    with cols[5]:
        st.write(f"{trip['torsk']['weight_kg']} kg")
    with cols[6]:
        st.write(f"{trip['sei']['weight_kg']} kg")
    with cols[7]:
        st.write(f"{trip['bifangst']['weight_kg']} kg")
    with cols[8]:
        st.write(trip["uncertain_count"])
    with cols[9]:
        if st.button("▲" if expanded else "▼", key=f"trip_{trip_id}"):
            st.session_state["expanded_trips"][trip_id] = not expanded
            st.rerun()

    export_text = _build_trip_export_text(trip)

    st.download_button(
        label="Eksporter tur",
        data=export_text,
        file_name=f"{trip_id}_rapport.txt",
        mime="text/plain",
        key=f"export_{trip_id}",
        use_container_width=True,
    )

    if st.session_state["expanded_trips"][trip_id]:
        st.markdown("#### Økter")
        _render_session_header()

        for session in trip["sessions"]:
            _render_session_row(session)

    st.divider()


def render_history_page() -> None:
    _ensure_expanded_state()

    history_service = HistoryService()
    data = history_service.get_history_page_data()

    st.markdown("## HISTORIKK")

    trips = data.get("trips", [])

    if not trips:
        st.info("Ingen lagrede turer funnet.")
        return

    _render_trip_header()
    st.divider()

    for trip in trips:
        _render_trip_row(trip)