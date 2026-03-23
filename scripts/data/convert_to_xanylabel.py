from ultralytics import YOLO

# Last inn din beste modell fra mappen på bildet ditt
model = YOLO(r'C:\Users\PC\Documents\Datateknikk-23\1.Batchoppgv\Kode\runs\segment\ACES_Models\V1_Baseline28\weights\best.pt')

# Eksporter til ONNX format
model.export(format='onnx', imgsz=640, opset=12)