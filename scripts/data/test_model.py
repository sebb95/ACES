from ultralytics import YOLO

# 1. Last inn din trente modell
model = YOLO(r'C:\ACES\ACES\runs\segment\ACES_Models\V1_Baseline7\weights\best.pt') 

# 2. Kjør inferens på videoen din
results = model.predict(
    source=r'C:\ACES\Data_Aces\Raw_data\fisk_test.mp4', # Din råvideo
    save=True,          # VIKTIG: Dette lagrer videoen med bokser/polygon
    conf=0.5,           # Bare vis fisk modellen er >50% sikker på
    imgsz=640,          # Samme størrelse som du trente på
    line_width=1,       # Tykkelse på polygon-strekene
    project='Resultat_Video', 
    agnostic_nms=True,
    name='Presentasjon_1'
)