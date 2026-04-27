from ultralytics import YOLO

# Last inn din beste modell fra mappen på bildet ditt
model = YOLO(r'C:\Users\PC\Documents\Datateknikk-23\1.Batchoppgv\ACES_Github_kode\runs\segment\outputs\runs\night_run\weights\best.pt')

# Eksporter til ONNX format
model.export(format='onnx', imgsz=640, opset=12)