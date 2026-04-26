from ultralytics import YOLO
import torch
import time

if __name__ == '__main__':
    modell_navn = r"runs\YOLOv11_Medium_Seg\weights\best.pt" 
    model = YOLO(modell_navn, task="segment")
    
    print(f"\n--- BENCHMARK FOR {modell_navn} ---")

    # 1. HENT MODELL-INFO (GFLOPs)
    print("\nHenter maskinvarekrav for modellen...")
    try:
        # model.info() returnerer (parametere, gradienter, GFLOPs)
        params, gradients, gflops = model.info()
        print(f"Antall parametere: {params:,}")
        print(f"GFLOPs: {gflops:.1f}")
    except Exception as e:
        print("GFLOPs: Ikke tilgjengelig (strippet fra .engine-filen).")

    # 2. TEST NØYAKTIGHET (mAP)
    print("\nTester mAP på valideringssettet...")
    metrics = model.val(data=r"C:\ACES\Data_Aces\Processed_data\Data_v1_2500_TrainReady\dataset.yaml", imgsz=640, device=0)
    # Boks-metrikker (Viktigst for telling)
    box_p = metrics.box.mp         # Mean Precision
    box_r = metrics.box.mr         # Mean Recall
    box_map50 = metrics.box.map50
    box_map50_95 = metrics.box.map
    
    # Maske-metrikker (Viktigst for størrelse/volum)
    seg_p = metrics.seg.mp
    seg_r = metrics.seg.mr
    seg_map50 = metrics.seg.map50
    seg_map50_95 = metrics.seg.map

    print("\n--- NØYAKTIGHET (BOKS / TELLING) ---")
    print(f"Precision (P):   {box_p:.4f}  <-- (Hvor ofte den har rett)")
    print(f"Recall (R):      {box_r:.4f}  <-- (Hvor mye av fisken den finner)")
    print(f"mAP50:           {box_map50:.4f}")
    print(f"mAP50-95:        {box_map50_95:.4f}")

    print("\n--- NØYAKTIGHET (MASKE / SEGMENTERING) ---")
    print(f"Precision (P):   {seg_p:.4f}")
    print(f"Recall (R):      {seg_r:.4f}")
    print(f"mAP50:           {seg_map50:.4f}")
    print(f"mAP50-95:        {seg_map50_95:.4f}")


    # 3. TEST FPS OG VRAM (Gjør et test-run på en video)
    print("\nTester FPS og VRAM-forbruk (kjører i maksimalt 60 sekunder)...")
    results = model.track(source=r"C:\ACES\ACES\data\Test_video_ACES_720p.mp4", stream=True, device=0, verbose=False)

    start_time = time.time()
    frames = 0
    
    for r in results:
        frames += 1
        
        if time.time() - start_time >= 60.0:
            print("60 sekunder har passert. Stopper videobehandlingen...")
            break
            
    end_time = time.time()
    
    # Matte for resultater
    total_time = end_time - start_time
    fps = frames / total_time
    
    # Henter høyeste VRAM forbruk i Gigabytes
    max_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    
    print("\n--- YTELSE ---")
    print(f"Totalt antall frames: {frames}")
    print(f"Gjennomsnittlig FPS:  {fps:.1f}")
    print(f"Peak VRAM-forbruk:    {max_vram_gb:.2f} GB")
    print("----------------------------------\n")