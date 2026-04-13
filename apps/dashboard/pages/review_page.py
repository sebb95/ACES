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


def _queue_label(item: dict) -> str:
    timestamp = item.get("timestamp") or "--:--:--"
    prediction = item.get("prediction") or "Ukjent"
    confidence = item.get("confidence")

    if confidence is None:
        return f'[ {timestamp} | {prediction} ]'
    return f'[ {timestamp} | {prediction} | {confidence}% ]'


def _render_queue_list(queue: list[dict]) -> None:
    if "selected_review_index" not in st.session_state:
        st.session_state["selected_review_index"] = 0

    if not queue:
        st.info("Ingen elementer i kø.")
        return

    max_index = len(queue) - 1
    if st.session_state["selected_review_index"] > max_index:
        st.session_state["selected_review_index"] = 0

    st.radio(
        label="Velg item",
        options=range(len(queue)),
        format_func=lambda i: _queue_label(queue[i]),
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

    is_disabled = selected_item is None

    if selected_item is None:
        prediction = "Ingen valgt"
        confidence_text = "Ikke tilgjengelig"
        filename = "-"
        timestamp = "--:--:--"
        default_index = 0
    else:
        prediction = selected_item["species_name"]
        confidence = selected_item.get("confidence")
        confidence_text = f"{confidence}%" if confidence is not None else "Ikke tilgjengelig"
        filename = selected_item["filename"]
        timestamp = selected_item.get("timestamp", "--:--:--")
        default_index = species_options.index(prediction) if prediction in species_options else 0

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
                Konfidens: {confidence_text}<br>
                Tidspunkt: {timestamp}<br>
                Fil: {filename}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_disabled:
            st.caption("Ingen elementer tilgjengelig for handling.")

    with col2:
        approve_clicked = st.button(
            "Bekreft forslag",
            use_container_width=True,
            key="approve_btn",
            disabled=is_disabled,
        )

        if approve_clicked and selected_item is not None:
            review_service.approve(filename)
            st.success("Forslag bekreftet.")
            st.rerun()

    with col3:
        new_species = st.selectbox(
            "Endre art",
            options=species_options,
            index=default_index,
            key="species_select",
            disabled=is_disabled,
        )

        change_species_clicked = st.button(
            "Lagre art",
            use_container_width=True,
            key="change_species_btn",
            disabled=is_disabled,
        )

        if change_species_clicked and selected_item is not None:
            review_service.change_species(filename, new_species)
            st.success("Art oppdatert.")
            st.rerun()

    with col4:
        reject_clicked = st.button(
            "Avvis",
            use_container_width=True,
            key="reject_btn",
            disabled=is_disabled,
        )

        if reject_clicked and selected_item is not None:
            review_service.reject(filename)
            st.warning("Element avvist.")
            st.rerun()

        send_to_land_clicked = st.button(
            "Til land",
            use_container_width=True,
            key="send_to_land_btn",
            disabled=is_disabled,
        )

        if send_to_land_clicked and selected_item is not None:
            review_service.send_to_land(filename)
            st.info("Element sendt til sync-kø.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_review_page() -> None:
    if "selected_review_index" not in st.session_state:
        st.session_state["selected_review_index"] = 0

    review_service = ReviewService()
    selected_index = st.session_state["selected_review_index"]
    data = review_service.get_review_page_data(selected_index=selected_index)

    trip_name = data["trip_name"]
    catch_id = data["catch_id"]
    queue = data["queue"]
    selected_item = data["selected_item"]
    species_options = data["species_options"]

    # Keep selected index valid if queue shrank after actions
    if queue and st.session_state["selected_review_index"] >= len(queue):
        st.session_state["selected_review_index"] = 0
        selected_item = data["selected_item"]

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