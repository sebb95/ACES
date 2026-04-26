import os, gc, random, yaml, shutil, torch, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

class NightOperations:
    def __init__(self):
        # ==========================================
        # 1. DYNAMISK ROT-MAPPE (Relativ struktur)
        # ==========================================
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        
        # ==========================================
        # 2. STIER: MODELLER OG MOTORER
        # ==========================================
        self.models_dir = self.BASE_DIR / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.curr_pt = self.models_dir / "current_best.pt"
        self.baseline_pt = self.models_dir / "baseline_v1.pt"
        self.prod_engine = self.models_dir / "production.engine"
        
        self.backups_dir = self.models_dir / "backups"
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # ==========================================
        # 3. STIER: DATA
        # ==========================================
        self.data_dir = self.BASE_DIR / "data"
        
        # Master (Gullgruven)
        self.golden_img_dir = self.data_dir / "master" / "train" / "images"
        self.golden_lbl_dir = self.data_dir / "master" / "train" / "labels"
        self.master_val_img_dir = self.data_dir / "master" / "val" / "images"
        
        # Innboks (Nye sollysbilder)
        self.approved_img_dir = self.data_dir / "inbox" / "images"
        self.approved_lbl_dir = self.data_dir / "inbox" / "labels"
        
        # Arkiv
        self.archive_img_dir = self.data_dir / "archive" / "images"
        self.archive_lbl_dir = self.data_dir / "archive" / "labels"
        self.archive_img_dir.mkdir(parents=True, exist_ok=True)
        self.archive_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamisk Validering (Eksamen for ny data)
        self.sunlight_val_img_dir = self.data_dir / "dynamic_val" / "images"
        self.sunlight_val_lbl_dir = self.data_dir / "dynamic_val" / "labels"
        self.sunlight_val_img_dir.mkdir(parents=True, exist_ok=True)
        self.sunlight_val_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # ==========================================
        # 4. STIER: SYSTEM OG LOGG
        # ==========================================
        self.outputs_dir = self.BASE_DIR / "outputs"
        self.run_dir = self.outputs_dir / "night_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_path = self.run_dir / "night_dataset.yaml"
        self.sunlight_yaml_path = self.run_dir / "sunlight_eval.yaml"
        
        # Logg lagres i roten av ACES
        log_path = self.BASE_DIR / "night_operations.log"
        logging.basicConfig(
            filename=str(log_path), 
            level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger()
        
    def check_system_health(self):
        free_space_gb = shutil.disk_usage(self.BASE_DIR).free / (1024**3)
        if free_space_gb < 5.0:
            self.logger.error(f"KRITISK FEIL: Lite diskplass ({free_space_gb:.1f} GB igjen). Avbryter.")
            return False
        
        if not self.baseline_pt.exists():
            self.logger.error(f"KRITISK: Mangler baseline_v1.pt i {self.models_dir}")
            print(f"❌ Feil: Finner ikke {self.baseline_pt.name}. Trening avbrutt!")
            return False
            
        return True

    def build_balanced_replay_buffer(self, class_dict, min_instances=50, max_total=300):
        label_files = list(self.golden_lbl_dir.glob("*.txt"))
        class_to_files = defaultdict(list)
        
        for lbl in label_files:
            with open(lbl, 'r') as f:
                classes_in_file = set([line.split()[0] for line in f.readlines() if line.strip()])
                for c in classes_in_file:
                    class_to_files[int(c)].append(lbl.stem)

        selected_stems = set()
        class_counts = {int(k): 0 for k in class_dict.keys()}

        for cls_id in class_dict.keys():
            cls_id = int(cls_id)
            available_files = class_to_files.get(cls_id, [])
            random.shuffle(available_files)

            for stem in available_files:
                if class_counts[cls_id] >= min_instances: break
                if stem not in selected_stems:
                    selected_stems.add(stem)
                    with open(self.golden_lbl_dir / f"{stem}.txt", 'r') as f:
                        for line in f.readlines():
                            if line.strip() and int(line.split()[0]) in class_counts:
                                class_counts[int(line.split()[0])] += 1

        all_stems = [f.stem for f in label_files]
        remaining = list(set(all_stems) - selected_stems)
        random.shuffle(remaining)
        needed = max_total - len(selected_stems)
        if needed > 0: selected_stems.update(remaining[:needed])

        selected_imgs = []
        for stem in selected_stems:
            for ext in ['.jpg', '.jpeg', '.png', '.PNG']:
                img = self.golden_img_dir / f"{stem}{ext}"
                if img.exists():
                    selected_imgs.append(img)
                    break
        return selected_imgs

    def run(self):
        print("\n" + "="*50)
        print("🌙 STARTER NIGHT OPERATIONS V2.0 (RELATIV)")
        print("="*50)
        print(f"📍 ACES Prosjektrot: {self.BASE_DIR}")
        self.logger.info("Starter ny Natt-operasjon.")

        if not self.check_system_health(): return

        # ---------------------------------------------------------
        # MODUL 1: INNDATA-SIKRING & VASK
        # ---------------------------------------------------------
        raw_imgs = []
        for f in self.approved_img_dir.glob("*"):
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                raw_imgs.append(f)
            
        if not raw_imgs:
            self.logger.info("Ingen nye data i innboksen. Avslutter normalt.")
            print("💤 Ingen nye data i dag.")
            return

        valid_new_imgs = []
        for img in raw_imgs:
            if os.path.getsize(img) == 0:
                continue
            
            lbl_file = self.approved_lbl_dir / f"{img.stem}.txt"
            if not lbl_file.exists():
                self.logger.error(f"KRITISK: Mangler label-fil for {img.name}.")
                print(f"❌ Feil: Finner ikke {lbl_file.name}. Trening avbrutt!")
                return
            valid_new_imgs.append(img)

        # ---------------------------------------------------------
        # MODUL 2: DYNAMISK TRAIN/VAL SPLIT (80/20)
        # ---------------------------------------------------------
        random.shuffle(valid_new_imgs)
        split_idx = max(1, int(len(valid_new_imgs) * 0.2)) 
        
        val_imgs_to_move = valid_new_imgs[:split_idx]
        train_imgs_to_keep = valid_new_imgs[split_idx:]
        
        print(f"📊 Deler data: {len(train_imgs_to_keep)} til trening, {len(val_imgs_to_move)} til validerings-eksamen.")
        
        for img in val_imgs_to_move:
            dest_img = self.sunlight_val_img_dir / img.name
            if dest_img.exists(): dest_img.unlink()
            shutil.move(str(img), str(dest_img))

            lbl_file = self.approved_lbl_dir / f"{img.stem}.txt"
            dest_lbl = self.sunlight_val_lbl_dir / f"{img.stem}.txt"
            if dest_lbl.exists(): dest_lbl.unlink()
            shutil.move(str(lbl_file), str(dest_lbl))

        # ---------------------------------------------------------
        # MODUL 3: YML BYGGING FOR TRENING OG EKSAMENER
        # ---------------------------------------------------------
        master_yaml_path = self.data_dir / "master" / "dataset.yaml"
        with open(master_yaml_path, 'r') as f:
            class_dict = yaml.safe_load(f).get("names", {})

        selected_golden = self.build_balanced_replay_buffer(class_dict, min_instances=50, max_total=300)
        
        with open(self.run_dir / "train_paths.txt", "w") as f:
            for img in (train_imgs_to_keep + selected_golden): 
                f.write(f"{img.resolve()}\n")

        with open(self.yaml_path, "w") as f:
            yaml.dump({
                "path": str(self.run_dir.resolve()), 
                "train": str((self.run_dir / "train_paths.txt").resolve()),
                "val": [str(self.master_val_img_dir.resolve()), str(self.sunlight_val_img_dir.resolve())],
                "names": class_dict
            }, f, sort_keys=False)

        # Spesial-YAML KUN for Sollys-eksamenen (Inkluderer dummy "train" for å fikse YOLO-error)
        with open(self.sunlight_yaml_path, "w") as f:
            yaml.dump({
                "path": str(self.run_dir.resolve()), 
                "train": str((self.run_dir / "train_paths.txt").resolve()), 
                "val": str(self.sunlight_val_img_dir.resolve()),
                "names": class_dict
            }, f, sort_keys=False)

        # ---------------------------------------------------------
        # MODUL 4: DEN TODELTE EKSAMENEN (DOUBLE QUALITY GATE)
        # ---------------------------------------------------------
        print("📈 Kjører pre-test av eksisterende modeller...")
        torch.cuda.empty_cache(); gc.collect()
        
        base_map_master = YOLO(str(self.baseline_pt.resolve())).val(data=str(master_yaml_path.resolve()), split='val', plots=False).seg.map50
        curr_map_sun = YOLO(str(self.curr_pt.resolve())).val(data=str(self.sunlight_yaml_path.resolve()), split='val', plots=False).seg.map50
        torch.cuda.empty_cache(); gc.collect()
        
        print(f"🎯 KRAV 1 (Beholde kunnskap): Må score minst {base_map_master - 0.02:.4f} på Master-data (Baseline er {base_map_master:.4f})")
        print(f"🎯 KRAV 2 (Lære sollys): Må slå gårsdagens sollys-score som er {curr_map_sun:.4f}")

        print("🧠 Starter trening...")
        project_dir = self.outputs_dir / "temp_training"
        if project_dir.exists(): shutil.rmtree(project_dir) 
        
        model = YOLO(str(self.curr_pt.resolve()))
        model.train(
            data=str(self.yaml_path.resolve()), epochs=15, batch=16, lr0=0.001, 
            mosaic=0.0, copy_paste=0.0, exist_ok=True, 
            project=str(project_dir.resolve()), name="night_run"
        )

        new_pt = project_dir / "night_run" / "weights" / "best.pt"
        new_map_master = YOLO(str(new_pt.resolve())).val(data=str(master_yaml_path.resolve()), split='val', plots=False).seg.map50
        new_map_sun = YOLO(str(new_pt.resolve())).val(data=str(self.sunlight_yaml_path.resolve()), split='val', plots=False).seg.map50
        
        print("\n" + "="*50)
        print("🏁 EKSAMENSRESULTATER 🏁")
        print(f"MASTER: Baseline = {base_map_master:.4f} | Ny modell = {new_map_master:.4f}")
        print(f"SOLLYS: Current  = {curr_map_sun:.4f} | Ny modell = {new_map_sun:.4f}")
        print("="*50)

        if new_map_sun > curr_map_sun and new_map_master >= (base_map_master - 0.02):
            print("✅ QUALITY GATE BESTÅTT! Oppdaterer produksjonssystem...")
            self.logger.info(f"Suksess! Sun: {new_map_sun:.4f} (slog {curr_map_sun:.4f}). Master: {new_map_master:.4f}")
            del model; torch.cuda.empty_cache(); gc.collect()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backups_dir / f"model_backup_{timestamp}.pt"
            shutil.copy(self.curr_pt, backup_path)
            
            backups = sorted(self.backups_dir.glob("*.pt"), key=os.path.getctime)
            while len(backups) > 5:
                backups[0].unlink()
                backups.pop(0)

            shutil.copy(new_pt, self.curr_pt)

            print("⚙️ Konverterer til TensorRT for Edge Inference...")
            try:
                eng_path = Path(YOLO(str(new_pt.resolve())).export(format="engine", half=True, device=0))
                if self.prod_engine.exists(): self.prod_engine.unlink()
                shutil.move(str(eng_path), str(self.prod_engine))
            except Exception as e:
                self.logger.error(f"TensorRT eksport feilet: {e}")
                print("⚠️ TensorRT eksport feilet. (Ignorer hvis TensorRT ikke er installert)")
        else:
            print("❌ QUALITY GATE FEILET. Kastes. Beholder gammel modell.")
            self.logger.warning(f"Avvist. Sun: {new_map_sun:.4f} (trengte > {curr_map_sun:.4f}). Master: {new_map_master:.4f}")

        # ==========================================
        # ALLTID TØM INNBOKSEN (Uavhengig av om vi besto eller feilet)
        # ==========================================
        print("📦 Tømmer innboksen og arkiverer brukt data...")
        for img in train_imgs_to_keep:
            dest_img = self.archive_img_dir / img.name
            if dest_img.exists(): dest_img.unlink()
            shutil.move(str(img), str(dest_img))
            
            lbl_file = self.approved_lbl_dir / f"{img.stem}.txt"
            if lbl_file.exists():
                dest_lbl = self.archive_lbl_dir / lbl_file.name
                if dest_lbl.exists(): dest_lbl.unlink()
                shutil.move(str(lbl_file), str(dest_lbl))

        print("🎉 Natt-operasjon ferdig! Klar for ny dag.")

if __name__ == "__main__":
    if torch.cuda.is_available(): 
        NightOperations().run()
    else:
        print("❌ CUDA ikke tilgjengelig. Sjekk GPU-oppsettet.")