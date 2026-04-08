import cv2
import threading
import time
from pathlib import Path

# Klargjør mapper
PENDING_DIR = Path("data/review_queue/pending")
PENDING_DIR.mkdir(parents=True, exist_ok=True)

def _save_hard_example_worker(frame, poly_data, conf, cls_id):
    """Kjøres i en asynkron tråd for å unngå FPS-drop"""
    timestamp = int(time.time() * 1000)
    base_name = f"hard_example_{timestamp}_cls{cls_id}_conf{conf:.2f}"
    
    cv2.imwrite(str(PENDING_DIR / f"{base_name}.jpg"), frame)
    
    # Lagre YOLO-seg format
    with open(PENDING_DIR / f"{base_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {poly_data}\n")

def trigger_hard_example_save(frame, mask_coords, conf, cls_id):
    """Filtrerer på konfidens og starter lagringstråden."""
    if 0.30 <= conf <= 0.80:
        frame_copy = frame.copy()
        # Gjøre om liste med koordinater til en streng adskilt med mellomrom
        poly_data = " ".join(map(str, mask_coords))
        threading.Thread(target=_save_hard_example_worker, args=(frame_copy, poly_data, conf, cls_id)).start()