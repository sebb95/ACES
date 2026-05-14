from ultralytics import YOLO

if __name__ == '__main__':
    print("Laster inn FP32-modell...")
    model = YOLO(r"outputs\weights\best.pt")

    print("Starter TensorRT-kvantisering (FP16)... Dette kan ta 10-15 minutter!")
    # workspace=4 betyr at den får låne 4GB VRAM under selve byggingen
    model.export(
        format="engine",
        half=True,       # Gjør om 32-bit til 16-bit
        device=0,        # Tvinger bruk av Lenovoens GPU
        workspace=4,
        imgsz=640
    )
    print("Kvanisering fullført, modellen er lagret som best.engine")