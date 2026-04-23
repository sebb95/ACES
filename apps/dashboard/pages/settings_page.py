import streamlit as st
from pathlib import Path

from services.settings_service import SettingsService


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


def _get_available_weight_files() -> list[str]:
    weights_path = Path("outputs/weights")
    if not weights_path.exists():
        return []
    return sorted([f.name for f in weights_path.glob("*.pt")])


def _get_dataset_folder_options() -> list[str]:
    candidates = [
        Path("data/sample"),
        Path("data/sample/images"),
        Path("data"),
    ]

    folders = []

    for base in candidates:
        if base.exists() and base.is_dir():
            folders.append(str(base))

    # add first-level subfolders under data/
    data_root = Path("data")
    if data_root.exists():
        for child in sorted(data_root.iterdir()):
            if child.is_dir():
                folders.append(str(child))

    # remove duplicates while preserving order
    seen = set()
    unique_folders = []
    for folder in folders:
        if folder not in seen:
            seen.add(folder)
            unique_folders.append(folder)

    return unique_folders


def render_settings_page() -> None:
    settings_service = SettingsService()
    config = settings_service.get()

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
        _render_section_title("Kamera / Input")

        fps = st.number_input(
            "FPS",
            min_value=1,
            max_value=60,
            value=int(config["camera"]["fps"]),
            step=1,
        )

        current_dataset_path = config["input"]["dataset_path"]

        dataset_path = st.text_input(
            "Dataset-mappe",
            value=current_dataset_path,
            help="Oppgi sti til mappe som inneholder bildene som skal brukes i runtime.",
        )

        _render_section_title("Modell")

        weight_files = _get_available_weight_files()
        current_model = config["model"]["selected_model"]

        if weight_files:
            selected_model = st.selectbox(
                "Velg modellfil",
                weight_files,
                index=weight_files.index(current_model) if current_model in weight_files else 0,
            )
            st.caption(f"Valgt modell: {selected_model}")
        else:
            selected_model = current_model
            st.warning("Ingen vektfiler funnet i outputs/weights")

        _render_section_title("Treningsstatus")
        st.info("Ingen trening aktiv")
        st.caption("Plassholder for senere treningsgrensesnitt.")

    with right_col:
        _render_section_title("Art og vekt")

        torsk = st.number_input(
            "Torsk snittvekt (kg)",
            min_value=0.0,
            value=float(config["species"]["torsk_weight"]),
            step=0.1,
        )

        sei = st.number_input(
            "Sei snittvekt (kg)",
            min_value=0.0,
            value=float(config["species"]["sei_weight"]),
            step=0.1,
        )

        bifangst = st.number_input(
            "Bifangst snittvekt (kg)",
            min_value=0.0,
            value=float(config["species"]["bifangst_weight"]),
            step=0.1,
        )

        _render_section_title("Gjennomgang / Active Learning")

        threshold = st.slider(
            "Usikkerhetsterskel",
            min_value=0.0,
            max_value=1.0,
            value=float(config["active_learning"]["uncertainty_threshold"]),
            step=0.01,
        )

        _render_section_title("Systemstatus")

        config_path = Path("configs/runtime_config.json")
        weights_path = Path("outputs/weights")
        dataset_folder = Path(dataset_path)

        if config_path.exists():
            st.success("Konfigurasjon: OK")
        else:
            st.error("Konfigurasjon: MANGEL")

        if weights_path.exists() and any(weights_path.glob("*.pt")):
            st.success("Modellfiler: OK")
        else:
            st.error("Modellfiler: MANGEL")

        if dataset_folder.exists() and dataset_folder.is_dir():
            st.success("Dataset-mappe: OK")
        else:
            st.error("Dataset-mappe: MANGEL")

    st.write("")
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])

    with btn_col1:
        if st.button("Lagre", use_container_width=True):
            new_config = {
                "model": {
                    "selected_model": selected_model,
                },
                "input": {
                    "dataset_path": dataset_path,
                },
                "camera": {
                    "fps": fps,
                },
                "species": {
                    "torsk_weight": torsk,
                    "sei_weight": sei,
                    "bifangst_weight": bifangst,
                },
                "active_learning": {
                    "uncertainty_threshold": threshold,
                },
            }

            settings_service.update(new_config)
            st.success("Innstillinger lagret")

    with btn_col2:
        if st.button("Tilbakestill", use_container_width=True):
            settings_service.reset()
            st.success("Tilbakestilt")
            st.rerun()