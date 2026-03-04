#demo for presentation

from pathlib import Path
from .config import InferConfig
from .model import load_model

def main():
    cfg = InferConfig(
        weights_path="yolov8n.pt", # demo baseline: pretrained weights
        source="data/demo",               # put some images here
    )

    if not cfg.source:
        raise ValueError("InferConfig.source is empty (image / folder / video path).")

    model = load_model(cfg.weights_path)

    run_name = cfg.run_name or "demo_infer"
    out_dir = Path(cfg.project_dir) / "runs"

    results = model.predict(
        source=cfg.source,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        device=cfg.device,
        save=cfg.save,
        save_txt=cfg.save_txt,
        project=str(out_dir),
        name=run_name,
        verbose=False,
    )

    # Print minimal summary to terminal
    n_imgs = len(results)
    total_det = 0
    for r in results:
        # r.boxes is present for detect; for segment there’s r.masks etc.
        if getattr(r, "boxes", None) is not None:
            total_det += len(r.boxes)

    print(f"[OK] Processed {n_imgs} images. Total detections: {total_det}")
    print(f"[OK] Outputs: {out_dir / run_name}")

if __name__ == "__main__":
    main()