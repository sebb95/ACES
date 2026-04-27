import cv2
import threading
import time
from pathlib import Path

# Klargjør mapper
PENDING_DIR = Path("data/review_queue/pending")
PENDING_DIR.mkdir(parents=True, exist_ok=True)

# Definer maksgrense for køen
MAX_QUEUE_SIZE = 2000

def _enforce_queue_limit():
    """
    Sjekker antall filer i mappen. Hvis det er over MAX_QUEUE_SIZE,
    slettes bildene med HØYEST avstand fra 50% konfidens (minst læringsverdi).
    """
    files = list(PENDING_DIR.glob("*.jpg"))
    if len(files) <= MAX_QUEUE_SIZE:
        return

    # Hjelpefunksjon for å beregne entropi (avstand fra 50%)
    def extract_entropy_value(filepath):
        try:
            # Henter ut konfidensen, f.eks. "hard_example_1714081234_cls0_conf0.45.jpg" -> 0.45
            conf_str = filepath.stem.split("conf")[-1]
            conf = float(conf_str)
            
            # Beregner avstanden fra 50% (0.50). 
            # F.eks: conf 0.50 gir avstand 0.00 (Mest verdifull - beholdes)
            # F.eks: conf 0.80 gir avstand 0.30 (Minst verdifull - slettes først)
            return abs(conf - 0.50)
        except Exception:
            return 1.0  # Failsafe: Hvis feil i filnavn, sett avstanden til maks (slettes uansett)

    # Sorterer slik at de med STØRST avstand fra 0.50 (de dårligste) havner øverst i listen
    files.sort(key=extract_entropy_value, reverse=True)

    # Finner de overskytende filene som skal slettes (de som ligger øverst)
    files_to_delete = files[:len(files) - MAX_QUEUE_SIZE]

    for f in files_to_delete:
        try:
            f.unlink()  # Slett .jpg
            txt_file = f.with_suffix(".txt")
            if txt_file.exists():
                txt_file.unlink()  # Slett tilhørende .txt
        except Exception as e:
            print(f"Feil ved sletting av {f.name}: {e}")

def _save_hard_example_worker(frame, poly_data, conf, cls_id):
    """Kjøres i en asynkron tråd for å unngå FPS-drop"""
    timestamp = int(time.time() * 1000)
    base_name = f"hard_example_{timestamp}_cls{cls_id}_conf{conf:.2f}"
    
    cv2.imwrite(str(PENDING_DIR / f"{base_name}.jpg"), frame)
    
    # Lagre YOLO-seg format
    with open(PENDING_DIR / f"{base_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {poly_data}\n")
    
    # 🧹 Kjør opprydding ETTER lagring for å opprettholde entropi-køen
    _enforce_queue_limit()

def trigger_hard_example_save(frame, mask_coords, conf, cls_id):
    if 0.30 <= conf <= 0.80:
        frame_copy = frame.copy()
        poly_data = " ".join(map(str, mask_coords))

        threading.Thread(
            target=_save_hard_example_worker,
            args=(frame_copy, poly_data, conf, cls_id)
        ).start()

        return True

    return False