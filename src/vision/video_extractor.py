from pathlib import Path
import cv2


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    every_n_frames: int = 1,
    overwrite: bool = True,
) -> Path:
    """
    Extract frames from a video into a folder of JPG images.

    Output is directly compatible with the existing detect → track → count pipeline.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # --- validation ---
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")

    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1")

    # --- prepare output ---
    if overwrite and output_dir.exists():
        for f in output_dir.glob("*.jpg"):
            f.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- open video ---
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_index = 0
    saved_index = 0

    # --- extraction loop ---
    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % every_n_frames == 0:
            frame_name = f"frame_{saved_index:06d}.jpg"
            frame_path = output_dir / frame_name

            ok = cv2.imwrite(str(frame_path), frame)
            if not ok:
                cap.release()
                raise RuntimeError(f"Failed to write frame: {frame_path}")

            saved_index += 1

        frame_index += 1

    cap.release()

    # --- sanity check ---
    if saved_index == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    return output_dir