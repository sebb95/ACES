import os, gc, random, yaml, shutil, torch, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

from src.vision.active_learning.yaml_sync import (
    write_master_dataset_yaml,
    write_night_training_yaml,
    write_dynamic_val_yaml,
)

class NightOperations:
    """
    Hovedklasse for ACES Kontinuerlig Læring (MLOps Pipeline).
    
    Håndterer autonom nattlig finjustering av YOLO-modellen. 
    Inkluderer datavask, dynamisk splitt, stratifisert historisk minne (Replay Buffer),
    og en "Double Quality Gate" for å forhindre "Catastrophic Forgetting" og Model Drift.
    """
    
    def __init__(
            self,
            current_model_path=None,
            baseline_model_path=None,
            approved_data_dir=None,
            ):
        
        """
        Initialiserer filstier, mappe-strukturer, logging og sikkerhetsgrenser.
        Bygger systemet basert på relative stier for å sikre portabilitet.
        """
        # ==========================================
        # 1. DYNAMISK ROT-MAPPE
        # ==========================================
        self.BASE_DIR = Path(__file__).resolve().parents[3]
        
        # ==========================================
        # 2. STIER: MODELLER OG MOTORER
        # ==========================================
        self.models_dir = self.BASE_DIR / "outputs" / "weights"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.curr_pt = Path(current_model_path) if current_model_path else self.models_dir / "best.pt"
        self.baseline_pt = Path(baseline_model_path) if baseline_model_path else self.models_dir / "best_backup.pt"
        self.prod_engine = self.models_dir / "production.engine"
        self.backups_dir = self.models_dir / "backups"
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # ==========================================
        # 3. STIER: DATA
        # ==========================================
        self.data_dir = self.BASE_DIR / "data"
        self.master_yaml_path = self.data_dir / "master" / "dataset.yaml"
        self.golden_img_dir = self.data_dir / "master" / "train" / "images"
        self.golden_lbl_dir = self.data_dir / "master" / "train" / "labels"
        self.master_val_img_dir = self.data_dir / "master" / "val" / "images"
        self.master_val_lbl_dir = self.data_dir / "master" / "val" / "labels"

        approved_root = Path(approved_data_dir) if approved_data_dir else self.data_dir / "training_reviewed"
        self.approved_img_dir = approved_root / "images"
        self.approved_lbl_dir = approved_root / "labels"
        
        self.archive_img_dir = self.data_dir / "archive" / "images"
        self.archive_lbl_dir = self.data_dir / "archive" / "labels"
        self.archive_img_dir.mkdir(parents=True, exist_ok=True)
        self.archive_lbl_dir.mkdir(parents=True, exist_ok=True)

        self.dynamic_val_img_dir = self.data_dir / "dynamic_val" / "images"
        self.dynamic_val_lbl_dir = self.data_dir / "dynamic_val" / "labels"

        self.dynamic_val_img_dir.mkdir(parents=True, exist_ok=True)
        self.dynamic_val_lbl_dir.mkdir(parents=True, exist_ok=True)

        
        # ==========================================
        # 4. STIER: SYSTEM OG LOGG
        # ==========================================
        self.outputs_dir = self.BASE_DIR / "outputs"
        self.run_dir = self.outputs_dir / "night_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_path = self.run_dir / "night_dataset.yaml"
        self.dynamic_val_yaml_path = self.run_dir / "dynamic_val.yaml"

        log_path = self.BASE_DIR / "night_operations.log"
        logging.basicConfig(
            filename=str(log_path), level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger()
        
        # SIKKERHETS-INNSTILLINGER
        self.MAX_NEW_TRAIN_IMAGES = 500  # Maks antall nye bilder per natt
        self.GOLDEN_RATIO = 1.2          # Vi vil ha 20% mer gammel data enn ny data

    # ==========================================
    # HOVEDMOTOR
    # ==========================================
    def run(self):
        """
        Hovedorkestratoren for natt-operasjonen. 
        Kjører logikken steg-for-steg: Helsesjekk -> Datavask -> Sampling -> 
        Data-splitt -> YAML-bygging -> Trening/Evaluering -> Arkivering.
        """
        print("\n" + "="*50)
        print("🌙 STARTER NIGHT OPERATIONS V2.0 (MODULÆR)")
        print("="*50)

        if not self.check_system_health(): return

        nye_bilder = self.hent_og_vask_innboks()
        if not nye_bilder: return

        # 🛡️ FISKER-SIKRING (Rettet variabelnavn)
        if len(nye_bilder) > self.MAX_NEW_TRAIN_IMAGES:
            print(f"⚠️ ADVARSEL: Innboksen er altfor stor ({len(nye_bilder)} bilder)!")
            print(f"🛡️ Sampler {self.MAX_NEW_TRAIN_IMAGES} tilfeldige bilder.")
            nye_bilder = random.sample(nye_bilder, self.MAX_NEW_TRAIN_IMAGES)

        treningsbilder = self.splitt_til_trening_og_eksamen(nye_bilder)
        
        # Dynamisk buffer-størrelse
        buffer_size = max(300, int(len(treningsbilder) * self.GOLDEN_RATIO))
        
        # Send buffer_size videre!
        self.bygg_yaml_filer(treningsbilder, buffer_size)
        
        self.tren_og_evaluer_modell()
        self.rydd_opp_arkiv(treningsbilder)

    # ==========================================
    # HJELPEMETODER
    # ==========================================
    def check_system_health(self):
        """
        Sjekker om systemet er i stand til å kjøre en treningsøkt.
        Verifiserer tilgjengelig diskplass og at baseline-modellen eksisterer.
        
        Returns:
            bool: True hvis systemet er friskt, False hvis operasjonen må avbrytes.
        """
        free_space_gb = shutil.disk_usage(self.BASE_DIR).free / (1024**3)
        if free_space_gb < 5.0:
            self.logger.error(f"KRITISK FEIL: Lite diskplass ({free_space_gb:.1f} GB igjen).")
            return False
        
        if not self.baseline_pt.exists():
            self.logger.error("KRITISK: Mangler baseline_v1.pt")
            print(f"❌ Feil: Finner ikke {self.baseline_pt.name}. Trening avbrutt!")
            return False
        return True

    def hent_og_vask_innboks(self):
        """
        Skanner innboksen for nye bilder godkjent for trening.
        Filtrerer bort korrupte filer (0 bytes) og bilder som mangler annoteringer (.txt).
        
        Returns:
            list: En liste med Path-objekter til de gyldige bildefilene.
        """
        raw_imgs = [f for f in self.approved_img_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not raw_imgs:
            self.logger.info("Ingen nye data. Avslutter.")
            print("💤 Ingen nye data i dag.")
            return []

        valid_imgs = []
        for img in raw_imgs:
            if os.path.getsize(img) == 0: continue
            if not (self.approved_lbl_dir / f"{img.stem}.txt").exists():
                self.logger.error(f"Mangler label for {img.name}.")
                print(f"❌ Feil: Finner ikke label for {img.name}. Trening avbrutt!")
                return []
            valid_imgs.append(img)
        return valid_imgs

    def splitt_til_trening_og_eksamen(self, valid_new_imgs):
        """
        Deler de nye bildene i treningsdata (70%) og valideringsdata (30%).
        Flytter valideringsbildene permanent til det dynamiske testsettet.
        
        Args:
            valid_new_imgs (list): Liste over alle godkjente nye bilder.
            
        Returns:
            list: Bildene som er reservert utelukkende for trening.
        """
        random.shuffle(valid_new_imgs)
        split_idx = max(1, int(len(valid_new_imgs) * 0.3)) 
        
        val_imgs = valid_new_imgs[:split_idx]
        train_imgs = valid_new_imgs[split_idx:]
        print(f"📊 Deler data: {len(train_imgs)} til trening, {len(val_imgs)} til validerings-eksamen.")
        
        for img in val_imgs:
            dest_img = self.dynamic_val_img_dir / img.name
            if dest_img.exists(): dest_img.unlink()
            shutil.move(str(img), str(dest_img))

            lbl_file = self.approved_lbl_dir / f"{img.stem}.txt"
            dest_lbl = self.dynamic_val_lbl_dir / f"{img.stem}.txt"
            if dest_lbl.exists(): dest_lbl.unlink()
            shutil.move(str(lbl_file), str(dest_lbl))
            
        return train_imgs

    def bygg_yaml_filer(self, treningsbilder, buffer_size):

        """
        Genererer nødvendige .yaml konfigurasjonsfiler for YOLO-trening og evaluering.
        Bygger et kombinert treningssett av nye bilder + Replay Buffer.
        
        Args:
            treningsbilder (list): Dagens nye bilder som skal brukes til trening.
            buffer_size (int): Det totale antallet historiske bilder som skal hentes.
        """
        class_dict = write_master_dataset_yaml(
            master_yaml_path=self.master_yaml_path,
            master_root=self.data_dir / "master",
            label_dirs=[
                self.golden_lbl_dir,
                self.data_dir / "master" / "val" / "labels",
                self.approved_lbl_dir,
                self.dynamic_val_lbl_dir,
            ],
        )

        selected_golden = self.build_balanced_replay_buffer(
            class_dict,
            max_total=buffer_size,
        )

        train_paths_file = self.run_dir / "train_paths.txt"

        with open(train_paths_file, "w", encoding="utf-8") as f:
            for img in treningsbilder + selected_golden:
                f.write(f"{img.resolve()}\n")

        write_night_training_yaml(
            yaml_path=self.yaml_path,
            run_dir=self.run_dir,
            train_paths_file=train_paths_file,
            master_val_img_dir=self.master_val_img_dir,
            dynamic_val_img_dir=self.dynamic_val_img_dir,
            class_names=class_dict,
        )

        write_dynamic_val_yaml(
            yaml_path=self.dynamic_val_yaml_path,
            run_dir=self.run_dir,
            train_paths_file=train_paths_file,
            dynamic_val_img_dir=self.dynamic_val_img_dir,
            class_names=class_dict,
        )

    def tren_og_evaluer_modell(self):
        """
        Kjernen i Active Learning-loopen. Trener modellen og utfører 'Double Quality Gate'.
        
        Logikk:
        1. Tester dagens modell mot Master og Ny data.
        2. Trener en oppdatert modell på den kombinerte treningsdataen.
        3. Sammenligner ny modell mot den gamle.
        4. Krever forbedring på ny data og tolererer maks 2% dropp på master-data.
        5. Ruller ut til Edge (TensorRT) ved bestått.
        """
        print("📈 Kjører pre-test av eksisterende modeller...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); 
        gc.collect()
        
        base_map = YOLO(str(self.baseline_pt.resolve())).val(data=str(self.master_yaml_path.resolve()), split='val', plots=False).seg.map50
        curr_map = YOLO(str(self.curr_pt.resolve())).val(data=str(self.dynamic_val_yaml_path.resolve()), split='val', plots=False).seg.map50
        
        print(f"🎯 KRAV 1 (Beholde kunnskap): Må score minst {base_map - 0.02:.4f} på Master-data (Baseline = {base_map:.4f})")
        print(f"🎯 KRAV 2 (Lære Ny Kunskap): Må slå gårsdagens Kunskaps-score som er {curr_map:.4f}")

        print("🧠 Starter trening...")
        project_dir = self.outputs_dir / "temp_training"
        if project_dir.exists(): shutil.rmtree(project_dir, ignore_errors=True) 
        
        model = YOLO(str(self.curr_pt.resolve()))
        model.train(
            data=str(self.yaml_path.resolve()), epochs=15, batch=16, lr0=0.001, 
            mosaic=0.0, copy_paste=0.0, exist_ok=True, project=str(project_dir.resolve()), name="night_run"
        )

        new_pt = project_dir / "night_run" / "weights" / "best.pt"
        new_map_master = YOLO(str(new_pt.resolve())).val(data=str(self.master_yaml_path.resolve()), split='val', plots=False).seg.map50
        new_map_sun = YOLO(str(new_pt.resolve())).val(data=str(self.dynamic_val_yaml_path.resolve()), split='val', plots=False).seg.map50
        
        print("\n" + "="*50)
        print("🏁 EKSAMENSRESULTATER 🏁")
        print(f"MASTER: Baseline = {base_map:.4f} | Ny modell = {new_map_master:.4f}")
        print(f"Kunskap score: Current  = {curr_map:.4f} | Ny modell = {new_map_sun:.4f}")
        print("="*50)

        if new_map_sun > curr_map and new_map_master >= (base_map - 0.02):
            print("✅ QUALITY GATE BESTÅTT! Oppdaterer produksjonssystem...")
            del model; torch.cuda.empty_cache(); gc.collect()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(self.curr_pt, self.backups_dir / f"model_backup_{timestamp}.pt")
            
            backups = sorted(self.backups_dir.glob("*.pt"), key=os.path.getctime)
            while len(backups) > 5: backups.pop(0).unlink()

            shutil.copy(new_pt, self.curr_pt)

            print("⚙️ Konverterer til TensorRT for Edge Inference...")
            try:
                eng_path = Path(YOLO(str(new_pt.resolve())).export(format="engine", half=True, device=0))
                if self.prod_engine.exists(): self.prod_engine.unlink()
                shutil.move(str(eng_path), str(self.prod_engine))
            except Exception as e:
                print("⚠️ TensorRT eksport feilet. (Ignorer hvis TensorRT ikke er installert)")
        else:
            print("❌ QUALITY GATE FEILET. Kastes. Beholder gammel modell.")

    def rydd_opp_arkiv(self, treningsbilder):
        """
        Flytter brukt treningsdata over i et lukket arkiv.
        Dette forhindrer at systemet trener på samme data neste natt (Dataset Poisoning),
        og legger grunnlaget for fremtidig utvidelse av Master-settet.
        
        Args:
            treningsbilder (list): Bildene som ble brukt i nattens treningsøkt.
        """
        print("📦 Tømmer innboksen og arkiverer brukt data...")
        for img in treningsbilder:
            dest_img = self.archive_img_dir / img.name
            if dest_img.exists(): dest_img.unlink()
            shutil.move(str(img), str(dest_img))
            
            lbl_file = self.approved_lbl_dir / f"{img.stem}.txt"
            if lbl_file.exists():
                dest_lbl = self.archive_lbl_dir / lbl_file.name
                if dest_lbl.exists(): dest_lbl.unlink()
                shutil.move(str(lbl_file), str(dest_lbl))

    def build_balanced_replay_buffer(self, class_dict, min_instances=25, max_total=300):
        """
        Bygger en Stratifisert Replay Buffer fra historisk data for å balansere ny læring.
        Sikrer at modellen repeterer alle klasser (Catastrophic Forgetting forsvar).
        
        Args:
            class_dict (dict): Ordbok med alle klassenavn og ID-er.
            min_instances (int): Minimum antall forekomster av hver fiskeart som MÅ velges.
            max_total (int): Totalt antall historiske bilder som skal returneres.
            
        Returns:
            list: Path-objekter til det historiske bilde-utvalget.
        """
        label_files = list(self.golden_lbl_dir.glob("*.txt"))
        class_to_files = defaultdict(list)
        for lbl in label_files:
            with open(lbl, 'r') as f:
                for c in set([line.split()[0] for line in f.readlines() if line.strip()]):
                    class_to_files[int(c)].append(lbl.stem)

        selected_stems = set()
        class_counts = {int(k): 0 for k in class_dict.keys()}

        for cls_id in class_dict.keys():
            cls_id = int(cls_id)
            available = class_to_files.get(cls_id, [])
            random.shuffle(available)
            for stem in available:
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

        return [img for stem in selected_stems for ext in ['.jpg', '.jpeg', '.png', '.PNG'] 
                if (img := self.golden_img_dir / f"{stem}{ext}").exists()]


if __name__ == "__main__":
    if torch.cuda.is_available(): 
        NightOperations().run()
    else:
        NightOperations().run()