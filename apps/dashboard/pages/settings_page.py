import streamlit as st


def _render_section_title(title: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:#d3d3d3;
            padding:0.9rem 1rem;
            margin-bottom:0.8rem;
            font-size:1.1rem;
            font-weight:700;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_page() -> None:
    st.markdown(
        """
        <div style="
            font-size:2rem;
            font-weight:800;
            margin-bottom:1rem;
        ">
            INNSTILLINGER
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        _render_section_title("Generelt")
        st.text_input("Tur ID format", value="Tur_YYYY_MM_DD")
        st.text_input("Fangstøkt ID format", value="Økt_001")
        st.checkbox("Start ny økt automatisk", value=False)
        st.checkbox("Lagre historikk lokalt", value=True)

        _render_section_title("Kamera")
        st.text_input("Kilde", value="0")
        st.selectbox("Oppløsning", ["1920x1080", "1280x720", "640x480"], index=0)
        st.slider("FPS", min_value=5, max_value=60, value=30)
        st.checkbox("Vis kamerafeed", value=True)

        _render_section_title("Modell")
        st.text_input("Modellsti", value="outputs/weights/baseline_best.pt")
        st.selectbox("Modelltype", ["YOLO Detect", "YOLO Segment"], index=1)
        st.slider("Konfidensgrense", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        st.slider("IoU-grense", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    with right_col:
        _render_section_title("Art og vekt")
        st.number_input("Torsk snittvekt (kg)", min_value=0.0, value=2.4, step=0.1)
        st.number_input("Sei snittvekt (kg)", min_value=0.0, value=2.0, step=0.1)
        st.number_input("Bifangst snittvekt (kg)", min_value=0.0, value=2.2, step=0.1)

        _render_section_title("Gjennomgang / Active Learning")
        st.checkbox("Send usikre funn til gjennomgang", value=True)
        st.slider("Usikkerhetsterskel", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        st.checkbox("Tillat sending til land", value=True)
        st.text_input("Sync-mappe", value="data/sync_queue")

        _render_section_title("Lagring")
        st.text_input("Sesjonsmappe", value="data/sessions")
        st.text_input("Review-kø", value="data/review_queue/pending")
        st.text_input("Godkjente labels", value="data/review_queue/approved_today/labels")
        st.text_input("Godkjente bilder", value="data/review_queue/approved_today/images")

    st.write("")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])

    with btn_col1:
        st.button("Lagre", use_container_width=True)

    with btn_col2:
        st.button("Tilbakestill", use_container_width=True)