import streamlit as st

from services.review_service import ReviewService


def _render_trip_info(trip_name: str, catch_id: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:#d3d3d3;
            padding:1rem;
            margin-bottom:0.8rem;
            font-size:1.1rem;
            font-weight:700;
            line-height:1.8;
        ">
            <div>TUR: "{trip_name}"</div>
            <div>FANGSTØKT: "{catch_id}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_queue_header() -> None:
    st.markdown(
        """
        <div style="
            background:#d3d3d3;
            padding:0.9rem 1rem;
            margin-bottom:0.8rem;
            font-size:1.1rem;
            font-weight:700;
        ">
            Til gjennomgang:
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_queue_list(queue: list[dict]) -> None:
    if "selected_review_index" not in st.session_state:
        st.session_state["selected_review_index"] = 0

    if not queue:
        st.info("Ingen elementer i kø.")
        return

    labels = [
        f'[ {item["timestamp"]} | {item["prediction"]} | {item["confidence"]}% ]'
        for item in queue
    ]

    st.radio(
        label="Velg item",
        options=range(len(labels)),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
        key="selected_review_index",
    )


def _render_image_area(selected_item: dict | None) -> None:
    if selected_item is None:
        st.markdown(
            """
            <div style="
                background:#d9d9d9;
                height:420px;
                margin-bottom:1rem;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:1.4rem;
                font-weight:600;
                color:#555;
            ">
                Ingen filer til gjennomgang
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # For now we show the actual image if path exists.
    # If later you want to hide image temporarily, replace with placeholder box.
    try:
        st.image(selected_item["path"], use_container_width=True)
    except Exception:
        st.markdown(
            """
            <div style="
                background:#d9d9d9;
                height:420px;
                margin-bottom:1rem;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:1.4rem;
                font-weight:600;
                color:#555;
            ">
                Bilde kunne ikke lastes
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_action_panel(
    review_service: ReviewService,
    selected_item: dict | None,
    species_options: list[str],
) -> None:
    st.markdown(
        """
        <div style="
            background:#d3d3d3;
            padding:1.2rem;
            margin-top:0.5rem;
        ">
        """,
        unsafe_allow_html=True,
    )

    if selected_item is None:
        st.markdown(
            """
            <div style="font-size:1.1rem;">
                Ingen aktiv review-post valgt.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    prediction = selected_item["species_name"]
    confidence = selected_item["confidence"]
    filename = selected_item["filename"]

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.1])

    with col1:
        st.markdown(
            f"""
            <div style="
                font-size:1.1rem;
                line-height:1.6;
                padding-top:0.3rem;
            ">
                AI forslag: {prediction}<br>
                Konfidens: {confidence}%<br>
                Fil: {filename}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button("Bekreft forslag", use_container_width=True, key="approve_btn"):
            review_service.approve(filename)
            st.success("Forslag bekreftet.")
            st.rerun()

    with col3:
        new_species = st.selectbox(
            "Endre art",
            options=species_options,
            index=species_options.index(prediction) if prediction in species_options else 0,
            key="species_select",
        )

        if st.button("Lagre art", use_container_width=True, key="change_species_btn"):
            review_service.change_species(filename, new_species)
            st.success("Art oppdatert.")
            st.rerun()

    with col4:
        if st.button("Avvis", use_container_width=True, key="reject_btn"):
            review_service.reject(filename)
            st.warning("Element avvist.")
            st.rerun()

        if st.button("Til land", use_container_width=True, key="send_to_land_btn"):
            review_service.send_to_land(filename)
            st.info("Element sendt til sync-kø.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_review_page() -> None:
    review_service = ReviewService()
    data = review_service.get_review_page_data()

    trip_name = data["trip_name"]
    catch_id = data["catch_id"]
    queue = data["queue"]
    selected_item = data["selected_item"]
    species_options = data["species_options"]

    left_col, right_col = st.columns([1, 2.2], gap="small")

    with left_col:
        _render_trip_info(trip_name=trip_name, catch_id=catch_id)
        _render_queue_header()
        _render_queue_list(queue)

    with right_col:
        _render_image_area(selected_item)
        _render_action_panel(
            review_service=review_service,
            selected_item=selected_item,
            species_options=species_options,
        )