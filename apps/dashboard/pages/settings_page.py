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
    folders = []

    data_root = Path("data")
    if data_root.exists():
        for child in sorted(data_root.iterdir()):
            if child.is_dir():
                folders.append(str(child))

    defaults = [
        "data/sample",
        "data/sample/images",
        "data/processed/frames/current_run",
    ]

    for folder in defaults:
        if folder not in folders:
            folders.insert(0, folder)

    return folders


def render_settings_page() -> None:
    settings_service = SettingsService()
    config = settings_service.get()

    model_config = config.get("model", {})
    input_config = config.get("input", {})
    camera_config = config.get("camera", {})
    species_config = config.get("species", {})
    active_learning_config = config.get("active_learning", {})
    training_config = config.get("training", {})

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
            value=int(camera_config.get("fps", 10)),
            step=1,
        )

        input_type = st.selectbox(
            "Input-type",
            options=["image_folder", "video_file"],
            index=0 if input_config.get("input_type", "image_folder") == "image_folder" else 1,
            help="Foreløpig bruker runtime bilde-mappe. Video kan senere pakkes ut til frames.",
        )

        current_dataset_path = input_config.get("dataset_path", "data/sample")
        dataset_path = st.text_input(
            "Dataset / frame-mappe",
            value=current_dataset_path,
            help="Mappe med bilder/frames som skal brukes i runtime.",
        )

        current_video_path = input_config.get("video_path", "data/input/video.mp4")
        video_path = st.text_input(
            "Videofil",
            value=current_video_path,
            help="Plassholder for senere video-input. Video pakkes senere ut til frames.",
        )

        frame_output_path = st.text_input(
            "Frame-output fra video",
            value=input_config.get("frame_output_path", "data/processed/frames/current_run"),
            help="Hvor frames fra video skal lagres når video-støtte kobles på.",
        )

        _render_section_title("Modell")

        weight_files = _get_available_weight_files()
        current_model = model_config.get("selected_model", "best.pt")

        if weight_files:
            selected_model = st.selectbox(
                "Velg modellfil",
                weight_files,
                index=weight_files.index(current_model) if current_model in weight_files else 0,
                help="Bytte av .pt-fil fungerer som modellbytte/rollback.",
            )
            st.caption(f"Valgt modell: {selected_model}")
        else:
            selected_model = current_model
            st.warning("Ingen vektfiler funnet i outputs/weights")

        _render_section_title("Trening placeholder")

        training_model = st.selectbox(
            "Modell for trening",
            options=weight_files if weight_files else [selected_model],
            index=0,
        )

        training_dataset_path = st.text_input(
            "Treningsdataset",
            value=training_config.get("dataset_path", "data/training_reviewed"),
        )

        night_training_enabled = st.checkbox(
            "Aktiver natt-trening",
            value=bool(training_config.get("night_training_enabled", False)),
        )

        night_training_time = st.text_input(
            "Tidspunkt for natt-trening",
            value=training_config.get("night_training_time", "03:00"),
        )

        training_status = training_config.get("status", "idle")
        st.info(f"Treningsstatus: {training_status}")

        if st.button("Start trening", use_container_width=True):
            st.warning("Trening er foreløpig bare en placeholder.")

    with right_col:
        _render_section_title("Art og vekt")

        torsk = st.number_input(
            "Torsk snittvekt (kg)",
            min_value=0.0,
            value=float(species_config.get("torsk_weight", 0.0)),
            step=0.1,
        )

        sei = st.number_input(
            "Sei snittvekt (kg)",
            min_value=0.0,
            value=float(species_config.get("sei_weight", 0.0)),
            step=0.1,
        )

        bifangst = st.number_input(
            "Bifangst snittvekt (kg)",
            min_value=0.0,
            value=float(species_config.get("bifangst_weight", 0.0)),
            step=0.1,
        )

        st.caption("Disse vektene brukes foreløpig bare til estimert fangstvekt i UI.")

        _render_section_title("Gjennomgang / Active Learning")

        review_min_confidence = st.slider(
            "Nedre grense for review",
            min_value=0.0,
            max_value=1.0,
            value=float(active_learning_config.get("review_min_confidence", 0.30)),
            step=0.01,
            help="Under denne verdien ignoreres det som støy/rot.",
        )

        review_max_confidence = st.slider(
            "Øvre grense for review",
            min_value=0.0,
            max_value=1.0,
            value=float(active_learning_config.get("review_max_confidence", 0.80)),
            step=0.01,
            help="Mellom nedre og øvre grense sendes telt fisk til review.",
        )

        if review_min_confidence > review_max_confidence:
            st.error("Nedre grense kan ikke være høyere enn øvre grense.")

        _render_section_title("Systemstatus")

        config_path = Path("configs/runtime_config.json")
        weights_path = Path("outputs/weights")
        dataset_folder = Path(dataset_path)
        video_file = Path(video_path)

        if config_path.exists():
            st.success("Konfigurasjon: OK")
        else:
            st.error("Konfigurasjon: MANGEL")

        if weights_path.exists() and any(weights_path.glob("*.pt")):
            st.success("Modellfiler: OK")
        else:
            st.error("Modellfiler: MANGEL")

        if input_type == "image_folder":
            if dataset_folder.exists() and dataset_folder.is_dir():
                st.success("Bilde-/frame-mappe: OK")
            else:
                st.error("Bilde-/frame-mappe: MANGEL")
        else:
            if video_file.exists() and video_file.is_file():
                st.success("Videofil: OK")
            else:
                st.warning("Videofil finnes ikke ennå")

    st.write("")
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])

    with btn_col1:
        if st.button("Lagre", use_container_width=True):
            if review_min_confidence > review_max_confidence:
                st.error("Kan ikke lagre: nedre review-grense er høyere enn øvre.")
                return

            new_config = {
                "model": {
                    "selected_model": selected_model,
                },
                "input": {
                    "input_type": input_type,
                    "dataset_path": dataset_path,
                    "video_path": video_path,
                    "frame_output_path": frame_output_path,
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
                    "review_min_confidence": review_min_confidence,
                    "review_max_confidence": review_max_confidence,
                },
                "training": {
                    "status": training_status,
                    "selected_model": training_model,
                    "dataset_path": training_dataset_path,
                    "night_training_enabled": night_training_enabled,
                    "night_training_time": night_training_time,
                },
            }

            settings_service.update(new_config)
            st.success("Innstillinger lagret")
            st.rerun()

    with btn_col2:
        if st.button("Tilbakestill", use_container_width=True):
            settings_service.reset()
            st.success("Tilbakestilt")
            st.rerun()