from ultralytics import YOLO

if __name__ == '__main__':

    # 1. Last inn en  "Medium" segmenteringsmodell
    model = YOLO('yolo11s-seg.pt') 

    # 2. Konfigurer og start treningen på min 5070 Ti
    # 2. Konfigurer og start treningen på min 5070 Ti
    results = model.train(
        data=r'C:\ACES\Data_Aces\Processed_data\Data_v1_2500_TrainReady\dataset.yaml',
        epochs=200,         # Husk å skru opp fra 1 for den faktiske treningen
        imgsz=640,          # Standard og trygg oppløsning
        batch=16,           # Batch size
        device=0,           # Tvinger bruk av NVIDIA GPU
        workers=8,          # 8 workers mater GPU-en raskt nok, 0 er en enorm flaskehals!
        amp=True,           # Automatic Mixed Precision (Gjør treningen lynrask)

        # --Data Augmentation (Domain Shift Mitigation for nye båter) ---
        hsv_h=0.015,        # Hue shift. Fakes different camera sensors.
        hsv_s=0.7,          # Saturation shift. Fakes washed-out lenses.
        hsv_v=0.4,          # Value (Brightness) shift. Fakes halogen vs LED lights.
        degrees=15.0,       # Rotation. Fakes skewed cameras on new boats.
        translate=0.1,      # Translation. Fakes conveyor belt moving left/right.
        scale=0.5,          # Zoom. Forces model to learn shapes, not absolute size.
        perspective=0.0001, # Perspective tilt. Fakes different camera angles.
        fliplr=0.5,         # 50% chance to flip left/right.
        flipud=0.5,         # 50% chance to flip up/down.
        mosaic=1.0,         # Smashes 4 images together. Breaks background context.
        mixup=0.0,          # MÅ VÆRE 0 FOR SEGMENTERING! Hindrer "ghost"-masker.
        copy_paste=0.0,      # Satt til 0, da dette ofte ødelegger for presise polygon-masker.
        
        project='ACES_Models',
        name='V2_Advanced'
    )

    print("Trening av V1 Baseline er fullført!")