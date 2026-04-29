"""
Home-side for ACES Streamlit-dashboard.

Siden viser aktiv tur og fangstøkt, starter og stopper økter,
viser live telling, estimert vekt, artsfordeling og antall usikre
deteksjoner til gjennomgang.

Selve prosesseringen kjøres i bakgrunnstråd via HomeService/HomeManager.
Denne siden leser oppdatert status og rerunner periodisk for å vise nye tall.
"""

import streamlit as st
import time

from services.home_service import HomeService
from services.training_service import TrainingService


def _render_trip_controls(service, trip: dict) -> None:
    st.markdown("### TUR")

    trip_name = trip.get("trip_name", "Ingen tur")

    if "editing_trip_name" not in st.session_state:
        st.session_state["editing_trip_name"] = False

    if st.session_state["editing_trip_name"]:
        new_name = st.text_input(
            "Tur navn",
            value=trip_name,
            label_visibility="collapsed",
            key="trip_name_edit_input",
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Lagre navn", use_container_width=True):
                service.rename_trip(new_name)
                st.session_state["editing_trip_name"] = False
                st.success("Navn oppdatert")
                st.rerun()

        with col2:
            if st.button("Avbryt", use_container_width=True):
                st.session_state["editing_trip_name"] = False
                st.rerun()

    else:
        st.markdown(
            f"""
            <div style="
                background:#d3d3d3;
                padding:0.9rem 1rem;
                margin-bottom:0.6rem;
                font-size:1.1rem;
                font-weight:800;
            ">
                {trip_name}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Endre navn", use_container_width=True):
            st.session_state["editing_trip_name"] = True
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Ny tur", use_container_width=True):
            service.start_new_trip()
            st.success("Ny tur startet")
            st.rerun()

    with col2:
        if st.button("Avslutt tur", use_container_width=True):
            service.end_trip()
            st.warning("Tur avsluttet")
            st.rerun()

    st.write("")

def _render_left_panel(
    catch_id: str,
    session_running: bool,
    session_active: bool,
) -> tuple[bool, bool]:
    st.markdown("### FANGSTØKT")

    st.markdown(
    f"""
    <div style="
        background:#d3d3d3;
        padding:0.9rem 1rem;
        margin-bottom:0.6rem;
        font-size:1.1rem;
        font-weight:800;
    ">
        {catch_id}
    </div>
    """,
    unsafe_allow_html=True,
)

    st.write("")

    training_status = TrainingService().get_status()
    training_running = training_status == "running"

    start_clicked = st.button(
        "START",
        use_container_width=True,
        type="primary" if not session_active and not training_running else "secondary",
        disabled=session_active or training_running,
    )

    stop_clicked = st.button(
        "STOP",
        use_container_width=True,
        type="primary" if session_active else "secondary",
        disabled=not session_active,
    )
    st.write("")

    return start_clicked, stop_clicked


def _render_main_metrics(total_count: int, estimated_weight_kg: int) -> None:
    st.markdown(
        f"""
        <div style="
            background:#d3d3d3;
            padding:2.8rem 2rem;
            margin-bottom:0.8rem;
            text-align:center;
            font-size:3.6rem;
            font-weight:800;
        ">
            TOTAL ANTALL: {total_count} st
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            background:#d3d3d3;
            padding:2.8rem 2rem;
            margin-bottom:0.8rem;
            text-align:center;
            font-size:3.2rem;
            font-weight:800;
        ">
            ESTIMERT VEKT: {estimated_weight_kg} kg
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_species_rows(
    torsk: dict,
    sei: dict,
    bifangst: dict,
    uncertain_count: int,
) -> None:
    rows = [
        ("Torsk", torsk["count"], torsk["weight_kg"]),
        ("Sei", sei["count"], sei["weight_kg"]),
        ("Bifangst", bifangst["count"], bifangst["weight_kg"]),
    ]

    for name, count, weight_kg in rows:
        st.markdown(
            f"""
            <div style="
                background:#d3d3d3;
                padding:1rem 1.5rem;
                margin-bottom:0.4rem;
                text-align:center;
                font-size:1.8rem;
                font-weight:700;
            ">
                {name}: {count} ({weight_kg} kg)
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Vis bifangst fordelt på art"):
        bifangst_species = bifangst.get("species", [])

        if not bifangst_species:
            st.write("Ingen bifangst registrert.")
        else:
            for item in bifangst_species:
                st.write(
                    f"{item['name']}: {item['count']} stk "
                    f"({item['weight_kg']} kg)"
                )

    st.markdown(
        f"""
        <div style="
            background:#d3d3d3;
            padding:0.8rem 1.5rem;
            margin-top:0.5rem;
            text-align:center;
            font-size:1.4rem;
            font-weight:700;
        ">
            Usikre til gjennomgang: {uncertain_count}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_page() -> None:
    st.markdown("---")
    try:
        service = HomeService()

    except (FileNotFoundError, ValueError) as e:
        error_msg = str(e).lower()

        if "weights" in error_msg or "model" in error_msg:
            st.error("❌ Modellfil ikke funnet")

            st.markdown(
                """
                Systemet finner ikke valgt modellfil.

                👉 Gå til **Innstillinger → Modell** og velg en gyldig `.pt`-fil.

                Sjekk også at:
                - Filen finnes i `outputs/weights`
                - Riktig modell er valgt i konfigurasjonen
                """
            )

        elif "dataset" in error_msg or "image files" in error_msg:
            st.error("❌ Dataset ikke funnet eller tom")

            st.markdown(
                """
                Systemet finner ikke gyldig input-data.

                👉 Gå til **Innstillinger → Kamera/Input** og velg riktig mappe.

                Sjekk at:
                - Dataset path peker til en eksisterende mappe
                - Mappen inneholder bilder (`.jpg`, `.png`, osv.)
                """
            )

        else:
            st.error("❌ Ukjent feil ved oppstart")
            st.markdown(
                """
                Noe gikk galt under oppstart av systemet.

                👉 Sjekk innstillinger eller kontakt utvikler.
                """
            )

        st.code(str(e))

        st.write("")

        if st.button("Gå til Innstillinger", use_container_width=True):
            st.session_state["current_page"] = "settings"
            st.rerun()

        return

    data = service.get_home_page_data()
    trip = data["trip"]

    st.session_state["review_pending"] = data["status"]["review_pending"]

    left_col, right_col = st.columns([1, 2.15], gap="small")

    with left_col:
        _render_trip_controls(service, trip)

        start_clicked, stop_clicked = _render_left_panel(
            catch_id=data["catch_id"],
            session_running=data["session_running"],
            session_active=data["session_active"],
        )

    if start_clicked:
        service.start()
        st.success("Økt startet.")
        st.rerun()

    if stop_clicked:
        service.stop()
        st.warning("Økt stoppet.")
        st.rerun()

    with right_col:
        _render_main_metrics(
            total_count=data["total_count"],
            estimated_weight_kg=data["estimated_weight_kg"],
        )

        _render_species_rows(
            torsk=data["torsk"],
            sei=data["sei"],
            bifangst=data["bifangst"],
            uncertain_count=data["status"]["uncertain_count"],
        )

    if service.is_running():
        #service.step()
        time.sleep(1.0)
        st.rerun()