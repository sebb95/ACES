import os
import gc
import shutil
import random
import torch
import yaml
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

from src.common.species import CLASS_NAMES
from src.vision.active_learning.yaml_sync import write_master_dataset_yaml


class FullRetrainOperations:
    """
    Håndterer full retrening av modellen når nye arter er klare.
    Bruker "Gullgraver"-metoden med Atomic Commits for å sikre dataintegritet.
    """

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[3]

        # Stier for data
        self.data_dir = self.base_dir / "data"
        self.new_species_dir = self.data_dir / "new_species_queue"
        self.master_dir = self.data_dir / "master"
        
        self.master_train_img = self.master_dir / "train" / "images"
        self.master_train_lbl = self.master_dir / "train" / "labels"
        self.master_val_img = self.master_dir / "val" / "images"
        self.master_val_lbl = self.master_dir / "val" / "labels"

        # Stier for modeller og output
        self.outputs_dir = self.base_dir / "outputs"
        self.weights_dir = self.outputs_dir / "weights"
        self.current_model = self.weights_dir / "best.pt"
        self.master_yaml_path = self.master_dir / "dataset.yaml"

        # Arbeidsmappe for midlertidige filer
        self.temp_dir = self.outputs_dir / "temp_retrain"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Innstillinger
        self.THRESHOLD = 100        # Bilder som kreves for å trigge trening
        self.VAL_SPLIT = 0.2        # 20% til validering
        self.FORGIVENESS = 0.05     # Tillatt dropp i total mAP50 pga ny art

    def run(self, ready_classes: list[int] = None):
        print("\n" + "=" * 50)
        print("🚀 STARTER FULL RETRENING (NYE ARTER)")
        print("=" * 50)

        # 1. Sjekk karantenekøen hvis den ikke ble sendt inn fra start-skriptet
        if not ready_classes:
            ready_classes = self._check_new_species_queue()
            
        if not ready_classes:
            print("💤 Ingen nye arter har nådd grensen enda. Avbryter.")
            return

        # 2. Finn filene (Gullgraver - MEN IKKE FLYTT DEM ENDA)
        train_stems, val_stems = self._get_new_species_files(ready_classes)
        # ... (resten av koden fortsetter som før)
        if not train_stems and not val_stems:
            return

        # 3. Bygg midlertidig YAML for treningen
        temp_yaml = self._build_temp_training_yaml(train_stems, val_stems)

        # 4. Tren og QGate (Commit hvis suksess)
        self._train_and_evaluate(temp_yaml, train_stems, val_stems)

        # 5. Rydd opp midlertidige filer
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


    def _check_new_species_queue(self) -> list[int]:
        """Skanner køen og returnerer class_ids som har >= THRESHOLD bilder."""
        counts = defaultdict(int)
        lbl_dir = self.new_species_dir / "labels"
        
        if not lbl_dir.exists():
            return []

        for txt_file in lbl_dir.glob("*.txt"):
            with open(txt_file, "r") as f:
                classes_in_img = set([line.split()[0] for line in f.readlines() if line.strip()])
                for c in classes_in_img:
                    counts[int(c)] += 1

        ready_classes = [cls_id for cls_id, count in counts.items() if count >= self.THRESHOLD]
        
        if ready_classes:
            print(f"🎯 Fant {len(ready_classes)} art(er) klare for Master: {ready_classes}")
            for c in ready_classes:
                name = CLASS_NAMES.get(c, f"Ukjent ({c})")
                print(f"   - {name}: {counts[c]} bilder")
                
        return ready_classes

    def _get_new_species_files(self, ready_classes: list[int]):
        """Finner kun de filene som inneholder de klare artene, og deler dem 80/20."""
        print("🔍 Plukker ut relevante filer fra karantene...")
        lbl_dir = self.new_species_dir / "labels"
        files_to_train = []

        for txt_file in lbl_dir.glob("*.txt"):
            with open(txt_file, "r") as f:
                classes_in_img = set([int(line.split()[0]) for line in f.readlines() if line.strip()])
                if classes_in_img.intersection(set(ready_classes)):
                    files_to_train.append(txt_file.stem)

        if not files_to_train:
            return [], []

        random.shuffle(files_to_train)
        split_idx = int(len(files_to_train) * (1 - self.VAL_SPLIT))
        return files_to_train[:split_idx], files_to_train[split_idx:]

    def _build_temp_training_yaml(self, train_stems, val_stems) -> Path:
        """Lager tekstfiler med absolutte stier til bildene, og en YAML som peker på disse."""
        print("📝 Genererer midlertidig YAML...")
        
        train_txt = self.temp_dir / "temp_train.txt"
        val_txt = self.temp_dir / "temp_val.txt"
        temp_yaml = self.temp_dir / "temp_retrain.yaml"

        # Skriv alle Master-trening bilder + nye trening bilder til fil
        with open(train_txt, "w") as f:
            for img in self.master_train_img.glob("*"):
                f.write(f"{img.resolve()}\n")
            for stem in train_stems:
                for ext in [".jpg", ".png", ".jpeg"]:
                    p = self.new_species_dir / "images" / f"{stem}{ext}"
                    if p.exists(): f.write(f"{p.resolve()}\n")

        # Skriv alle Master-val bilder + nye val bilder til fil
        with open(val_txt, "w") as f:
            for img in self.master_val_img.glob("*"):
                f.write(f"{img.resolve()}\n")
            for stem in val_stems:
                for ext in [".jpg", ".png", ".jpeg"]:
                    p = self.new_species_dir / "images" / f"{stem}{ext}"
                    if p.exists(): f.write(f"{p.resolve()}\n")

        # Generer YAML
        yaml_data = {
            "path": str(self.temp_dir.resolve()),
            "train": str(train_txt.resolve()),
            "val": str(val_txt.resolve()),
            "names": {int(k): str(v) for k, v in CLASS_NAMES.items()}
        }

        with open(temp_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
            
        return temp_yaml

    def _train_and_evaluate(self, temp_yaml_path, train_stems, val_stems):
        """Kjører opptil 200 epoker, tester mot Quality Gate, og committer."""
        print("🧠 Henter baseline-score før trening starter...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 1. Hent score fra den gamle modellen på de GAMLE dataene (Baseline)
        old_map = YOLO(str(self.current_model), task="segment").val(
            data=str(self.master_yaml_path), split="val", plots=False # <--- ENDRET HER!
        ).seg.map50

        # 2. Start Full Trening
        print("🔥 Starter trening (opptil 200 epoker)...")
        project_dir = self.outputs_dir / "full_retrain_run"
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)

        model = YOLO(str(self.current_model), task="segment")
        model.train(
            data=str(temp_yaml_path),
            epochs=2,
            batch=16,
            lr0=0.01,
            patience=50,  # Stopper tidlig hvis den flater ut
            exist_ok=True,
            project=str(project_dir),
            name="new_species_model",
        )

        new_pt = project_dir / "new_species_model" / "weights" / "best.pt"

        # 3. Hent score fra den NYE modellen
        print("📈 Kjører eksamen på ny modell...")
        new_map = YOLO(str(new_pt), task="segment").val(
            data=str(temp_yaml_path), split="val", plots=False
        ).seg.map50

        print("\n" + "=" * 50)
        print("🏁 RESULTAT AV FULL RETRENING 🏁")
        print(f"Gammel modell mAP50: {old_map:.4f}")
        print(f"Ny modell mAP50:     {new_map:.4f}")
        print("=" * 50)

        # 4. QUALITY GATE & ATOMIC COMMIT
        if new_map >= (old_map - self.FORGIVENESS):
            print("✅ QUALITY GATE BESTÅTT!")
            
            print("💾 Oppdaterer best.pt...")
            shutil.copy(new_pt, self.current_model)
            
            print("📦 Flytter godkjent data til permanent Master...")
            self._commit_to_master(train_stems, self.master_train_img, self.master_train_lbl)
            self._commit_to_master(val_stems, self.master_val_img, self.master_val_lbl)
            
            print("📝 Oppdaterer permanent dataset.yaml...")
            write_master_dataset_yaml(
                master_yaml_path=self.master_yaml_path,
                master_root=self.master_dir,
                label_dirs=[self.master_train_lbl, self.master_val_lbl]
            )
            
            print("⚙️ Konverterer til TensorRT (production.engine)...")
            try:
                eng_path = YOLO(str(self.current_model), task="segment").export(
                    format="engine", half=True, device=0, imgsz=640
                )
                prod_engine = self.weights_dir / "production.engine"
                if prod_engine.exists():
                    prod_engine.unlink()
                shutil.move(str(eng_path), str(prod_engine))
                print(f"🚀 Klar til bruk! Motor lagret som {prod_engine.name}")
            except Exception as e:
                print(f"⚠️ TensorRT eksport feilet: {e}")
                
        else:
            print(f"❌ QUALITY GATE FEILET. Droppet mer enn {self.FORGIVENESS*100}%.")
            print("🗑️ Modellen forkastes. Data forblir i karantene til neste forsøk.")


    def _commit_to_master(self, stems, dest_img_dir, dest_lbl_dir):
        """Flytter filene permanent fra køen til Master."""
        img_dir = self.new_species_dir / "images"
        lbl_dir = self.new_species_dir / "labels"
        
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for stem in stems:
            img_moved = False
            for ext in [".jpg", ".jpeg", ".png", ".PNG"]:
                img_path = img_dir / f"{stem}{ext}"
                if img_path.exists():
                    shutil.move(str(img_path), str(dest_img_dir / img_path.name))
                    img_moved = True
                    break
            
            if img_moved:
                lbl_path = lbl_dir / f"{stem}.txt"
                if lbl_path.exists():
                    shutil.move(str(lbl_path), str(dest_lbl_dir / lbl_path.name))
                count += 1
                
        print(f"   -> Flyttet {count} filer til {dest_img_dir.parent.name}")