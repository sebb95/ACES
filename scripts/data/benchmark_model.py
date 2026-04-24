from ultralytics import YOLO
import torch
import time

if __name__ == '__main__':
    # --- BYTT MELLOM 'best.pt' og 'best.engine' HER
    modell_navn = "best.engine" 
    model = YOLO(modell_navn)
    
    print(f"\n--- BENCHMARK FOR {modell_navn} ---")

    # 1. TEST NØYAKTIGHET (mAP)
    print("Tester mAP på valideringssettet...")
    metrics = model.val(data="dataset.yaml", imgsz=640, device=0)
    map50_95 = metrics.box.map  # Henter den strenge mAP-scoren
    print(f"mAP50-95 Score: {map50_95:.4f}")

    # 2. TEST FPS OG VRAM (Gjør et test-run på en video)
    print("\nTester FPS og VRAM-forbruk...")
    # Bruk en kort test video
    results = model.track(source="test_video.mp4", stream=True, device=0, verbose=False)
    
    start_time = time.time()
    frames = 0
    
    for r in results:
        frames += 1
        
    end_time = time.time()
    
    # Matte for resultater
    total_time = end_time - start_time
    fps = frames / total_time
    
    # Henter høyeste VRAM forbruk i Gigabytes
    max_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    
    print(f"Gjennomsnittlig FPS: {fps:.1f}")
    print(f"Peak VRAM-forbruk: {max_vram_gb:.2f} GB")
    print("----------------------------------\n")