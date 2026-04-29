import streamlit as st
import time
from pathlib import Path
from services.settings_service import SettingsService
from services.training_service import TrainingService


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
    st.markdown("---")
    settings_service = SettingsService()
    config = settings_service.get()

    model_config = config.get("model", {})
    input_config = config.get("input", {})
    camera_config = config.get("camera", {})
    species_config = config.get("species", {})
    active_learning_config = config.get("active_learning", {})
    training_config = config.get("training", {})

    current_input_type = input_config.get("input_type", "image_folder")
    current_dataset_path = input_config.get("dataset_path", "data/sample")
    current_video_path = input_config.get("video_path", "data/input/video.mp4")
    

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

    st.markdown("### SYSTEMSTATUS")

    status_col1, status_col2, status_col3 = st.columns(3)

    config_path = Path("configs/runtime_config.json")
    weights_path = Path("outputs/weights")
    dataset_folder = Path(current_dataset_path)
    video_file = Path(current_video_path)

    with status_col1:
        if config_path.exists():
            st.success("Konfigurasjon: OK")
        else:
            st.error("Konfigurasjon: MANGEL")

    with status_col2:
        if weights_path.exists() and any(weights_path.glob("*.pt")):
            st.success("Modellfiler: OK")
        else:
            st.error("Modellfiler: MANGEL")

    with status_col3:
        if current_input_type == "image_folder":
            if dataset_folder.exists() and dataset_folder.is_dir():
                st.success("Bilde-/frame-mappe: OK")
            else:
                st.error("Bilde-/frame-mappe: MANGEL")
        else:
            if video_file.exists() and video_file.is_file():
                st.success("Videofil: OK")
            else:
                st.warning("Videofil finnes ikke ennå")

    left_col, right_col = st.columns(2, gap="large")

    with left_col:

        with st.expander("Kamera / Input", expanded=False):

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

        with st.expander("Modell", expanded=False):

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

        with st.expander("Trening", expanded=False):

            night_training_enabled = st.checkbox(
                "Aktiver natt-trening",
                value=bool(training_config.get("night_training_enabled", False)),
            )

            night_training_time = st.text_input(
                "Tidspunkt for natt-trening",
                value=training_config.get("night_training_time", "03:00"),
            )

            training_status = training_config.get("status", "idle")
            if training_status == "running":
                st.info("🟡 Treningsstatus: Trening pågår.")
            elif training_status == "ready":
                st.success("🟢 Treningsstatus: Modell klar.")
            elif training_status == "failed":
                st.error("🔴 Treningsstatus: Trening feilet.")
            else:
                st.caption("⚪ Treningsstatus: Ingen aktiv trening.")

            training_status = training_config.get("status", "idle")
            training_running = training_status == "running"

            if st.button(
                "Start trening",
                use_container_width=True,
                disabled=training_running,
            ):
                try:
                    service = TrainingService()

                    with st.spinner("Trening pågår..."):
                        service.run_training()

                    st.success("Trening fullført.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Trening feilet: {e}")
                    st.rerun()

            if training_running:
                if st.button("Avbryt trening", use_container_width=True):
                    service = TrainingService()
                    service._set_status("failed")
                    st.warning("Trening avbrutt.")
                    st.rerun()

    with right_col:
        with st.expander("Art og vekt", expanded=False):

            species_weights = species_config.get("weights_kg", {})

            updated_species_weights = {}

            for species_name in sorted(species_weights.keys()):
                updated_species_weights[species_name] = st.number_input(
                    f"{species_name} snittvekt (kg)",
                    min_value=0.0,
                    value=float(species_weights.get(species_name, 0.0)),
                    step=0.1,
                    key=f"species_weight_{species_name}",
                )

            st.caption("Disse vektene brukes til estimert fangstvekt i UI.")

            st.write("")
            st.markdown("#### Legg til ny art")

            new_species_name = st.text_input(
                "Artsnavn",
                value="",
                key="new_species_name",
            )

            if st.button("Legg til art", use_container_width=True):
                clean_name = new_species_name.strip()

                if not clean_name:
                    st.error("Artsnavn kan ikke være tomt.")
                else:
                    try:
                        new_class_id = settings_service.add_species(clean_name)

                        st.success(
                            f"Art lagt til som klasse {new_class_id}. "
                            "Sett snittvekt i artslisten. Modellen vil først kunne gjenkjenne arten etter ny trening."
                        )

                        time.sleep(1.5)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Kunne ikke legge til art: {e}")

        with st.expander("Gjennomgang / Active Learning", expanded=False):

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
                    "weights_kg": updated_species_weights,
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
            st.toast("Endringer lagret!", icon="✅")
            time.sleep(0.8)
            st.rerun()

    with btn_col2:
        if st.button("Tilbakestill", use_container_width=True):
            settings_service.reset()
            st.toast("Tilbakestilt!", icon="✅")
            time.sleep(0.8)
            st.rerun()