import streamlit as st

from services.home_service import HomeService


def _render_left_panel(
    trip_name: str,
    catch_id: str,
    session_running: bool,
) -> tuple[bool, bool]:
    st.markdown("### TUR")
    st.text_input(
        label="Tur",
        value=trip_name,
        label_visibility="collapsed",
        key="trip_name",
    )

    st.markdown("### FANGSTØKT")
    st.text_input(
        label="Fangstøkt",
        value=catch_id,
        label_visibility="collapsed",
        key="catch_id",
    )

    st.write("")

    start_clicked = st.button(
        "START",
        use_container_width=True,
        type="primary",
        disabled=session_running,
    )

    stop_clicked = st.button(
        "STOP",
        use_container_width=True,
        disabled=not session_running,
    )

    st.write("")
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


def _render_species_rows(species: list[dict]) -> None:
    for item in species:
        name = item["name"]
        count = item["count"]
        weight_kg = item["weight_kg"]

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


def render_home_page() -> None:
    service = HomeService()

    if service.is_running():
        service.step()

    data = service.get_home_page_data()

    st.session_state["review_pending"] = data["status"]["review_pending"]

    left_col, right_col = st.columns([1, 2.15], gap="small")

    with left_col:
        start_clicked, stop_clicked = _render_left_panel(
            trip_name=data["trip_name"],
            catch_id=data["catch_id"],
            session_running=data["session_running"],
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
        _render_species_rows(data["species"])