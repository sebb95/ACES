import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


PENDING_DIR = Path("data/review_queue/pending")
PENDING_DIR.mkdir(parents=True, exist_ok=True)


def _save_hard_example_worker(
    frame,
    poly_data: str,
    conf: float,
    cls_id: int,
    track_id: int,
    session_id: str | None,
    image_path: str | None,
) -> None:
    timestamp = int(time.time() * 1000)
    base_name = f"counted_uncertain_{timestamp}_track{track_id}_cls{cls_id}_conf{conf:.2f}"

    image_filename = f"{base_name}.jpg"
    label_filename = f"{base_name}.txt"
    metadata_filename = f"{base_name}.json"

    cv2.imwrite(str(PENDING_DIR / image_filename), frame)

    with open(PENDING_DIR / label_filename, "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {poly_data}\n")

    metadata = {
        "filename": image_filename,
        "label_filename": label_filename,
        "metadata_filename": metadata_filename,
        "session_id": session_id,
        "track_id": track_id,
        "class_id": cls_id,
        "confidence": round(conf, 4),
        "was_counted": True,
        "source_image_path": image_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "review_status": "pending",
    }

    with open(PENDING_DIR / metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def should_send_to_review(
    confidence: float,
    min_confidence: float = 0.30,
    max_confidence: float = 0.80,
) -> bool:
    """
    Decide whether a counted detection should be saved for later review.

    Below min_confidence:
        treated as garbage/noise, not useful for review

    Between min_confidence and max_confidence:
        uncertain enough to review later

    Above max_confidence:
        trusted prediction, no review needed
    """
    return min_confidence <= confidence <= max_confidence


def trigger_hard_example_save(
    frame,
    mask_coords,
    conf: float,
    cls_id: int,
    track_id: int,
    min_confidence: float = 0.30,
    max_confidence: float = 0.80,
    session_id: str | None = None,
    image_path: str | None = None,
) -> bool:
    """
    Save a counted uncertain detection to the review queue.

    Returns:
        True if the item was sent to review.
        False if confidence was outside the review range.
    """
    if not should_send_to_review(conf, min_confidence, max_confidence):
        return False

    if frame is None or not mask_coords:
        return False

    frame_copy = frame.copy()
    poly_data = " ".join(map(str, mask_coords))

    threading.Thread(
        target=_save_hard_example_worker,
        args=(
            frame_copy,
            poly_data,
            conf,
            cls_id,
            track_id,
            session_id,
            image_path,
        ),
        daemon=True,
    ).start()

    return True