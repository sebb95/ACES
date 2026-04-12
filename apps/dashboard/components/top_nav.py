#navigasjon bar, gjemensam for alle sider

import streamlit as st


def render_top_nav() -> None:
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "home"

    if "review_pending" not in st.session_state:
        st.session_state["review_pending"] = 0

    review_pending = st.session_state["review_pending"]

    col1, col2, col3, col4, col5, col6, col7 = st.columns(
        [1.0, 1.0, 1.0, 1.0, 1.15, 1.0, 1.2],
        gap="small",
    )

    with col1:
        st.markdown(
            """
            <div style="
                background:#bfbfbf;
                padding:0.78rem;
                text-align:center;
                font-weight:700;
                font-size:1.1rem;
                height:3rem;
                display:flex;
                align-items:center;
                justify-content:center;
            ">
                ACES
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button("Hjem", use_container_width=True):
            st.session_state["current_page"] = "home"

    with col3:
        st.button("Modell 🟢", use_container_width=True, disabled=True)

    with col4:
        st.button("Kamera 🟢", use_container_width=True, disabled=True)

    with col5:
        if st.button(f"Gjennomgang 🔴 {review_pending}", use_container_width=True):
            st.session_state["current_page"] = "review"

    with col6:
        if st.button("Historikk", use_container_width=True):
            st.session_state["current_page"] = "history"

    with col7:
        if st.button("Innstillinger", use_container_width=True):
            st.session_state["current_page"] = "settings"