import os, gc, random, yaml, shutil, torch
from pathlib import Path
from ultralytics import YOLO

class NightOperations:
    def __init__(self):
        self.gold_pt = Path("outputs/weights/v1_master_baseline.pt")
        self.curr_pt = Path("outputs/weights/current_best.pt")
        self.prod_engine = Path("outputs/weights/production.engine")
        
        self.golden_img_dir = Path("data/master_dataset/images/train")
        self.approved_img_dir = Path("data/review_queue/approved_today/images")
        
        self.run_dir = Path("outputs/runs/night_run")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_path = self.run_dir / "night_dataset.yaml"

    def run(self):
        new_imgs = list(self.approved_img_dir.glob("*.jpg"))
        if not new_imgs: return print("Ingen nye data i dag.")

        # 1. Bygg Replay Buffer
        all_golden = list(self.golden_img_dir.glob("*.jpg"))
        selected_golden = random.sample(all_golden, min(150, len(all_golden)))
        
        with open(self.run_dir / "train_paths.txt", "w") as f:
            for img in (new_imgs + selected_golden): f.write(f"{img.resolve()}\n")

        with open(self.yaml_path, "w") as f:
            yaml.dump({
                "path": str(self.run_dir.resolve()), "train": str((self.run_dir/"train_paths.txt").resolve()),
                "val": str(Path("data/master_dataset/val_paths.txt").resolve()),
                "names": {0: "Torsk", 1: "Sei", 2: "Hyse", 3: "Bifangst", 4: "Ukjent"}
            }, f, sort_keys=False)

        # 2. Mål Baseline & Gold Master
        m1, m2 = YOLO(self.curr_pt), YOLO(self.gold_pt)
        base_map = m1.val(data=str(self.yaml_path), split='val', plots=False).seg.map
        gold_map = m2.val(data=str(self.yaml_path), split='val', plots=False).seg.map
        del m1, m2
        torch.cuda.empty_cache(); gc.collect()

        # 3. Tren
        model = YOLO(self.curr_pt)
        model.train(data=str(self.yaml_path), epochs=15, batch=4, freeze=10, lr0=0.001, 
                    mosaic=1.0, copy_paste=0.5, exist_ok=True, project="outputs/runs", name="night_run")

        # 4. Evaluer & Double Quality Gate
        new_pt = Path("outputs/runs/night_run/weights/best.pt")
        new_map = YOLO(new_pt).val(data=str(self.yaml_path), split='val', plots=False).seg.map
        
        if new_map >= (base_map - 0.005) and new_map >= (gold_map - 0.020):
            print("✅ Quality Gate bestått. Oppdaterer modell...")
            del model; torch.cuda.empty_cache(); gc.collect()
            
            eng_path = Path(YOLO(new_pt).export(format="engine", half=True, dynamic=False, simplify=True, device=0))
            shutil.copy(new_pt, self.curr_pt)
            if self.prod_engine.exists(): self.prod_engine.unlink()
            shutil.move(str(eng_path), str(self.prod_engine))
            
            for f in self.approved_img_dir.parent.rglob("*"): 
                if f.is_file(): f.unlink()
        else:
            print("❌ Quality Gate FEILET. Endringer forkastet.")

if __name__ == "__main__":
    if torch.cuda.is_available(): NightOperations().run()