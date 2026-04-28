import json
import shutil
from datetime import datetime
from pathlib import Path
from src.vision.active_learning.new_species_queue import move_to_new_species_queue

class ReviewManager:
    """
    Backend for filhåndtering i review/active learning.

    Klassen håndterer elementer som ligger i review-køen etter usikre
    deteksjoner. Brukeren kan godkjenne, avvise, endre art eller sende
    elementet til manuell vurdering.

    Viktig design:
    - Vanlige, kjente arter flyttes til training_reviewed og kan brukes i
      nattlig finjustering.
    - Nye arter flyttes til new_species_queue (karantene) og holdes utenfor
      vanlig night training frem til det finnes nok eksempler for full retrening.
    - Review påvirker ikke allerede lagrede fangsttall i aktiv økt.
    """

    def __init__(self):
        self.pending = Path("data/review_queue/pending")

        self.training_images = Path("data/training_reviewed/images")
        self.training_labels = Path("data/training_reviewed/labels")
        self.training_metadata = Path("data/training_reviewed/metadata")

        self.rejected_images = Path("data/review_queue/rejected/images")
        self.rejected_metadata = Path("data/review_queue/rejected/metadata")

        self.sync = Path("data/sync_queue")

        for directory in [
            self.pending,
            self.training_images,
            self.training_labels,
            self.training_metadata,
            self.rejected_images,
            self.rejected_metadata,
            self.sync,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _paths_for(self, filename: str) -> dict:
        img = self.pending / filename
        stem = img.stem

        return {
            "img": img,
            "txt": self.pending / f"{stem}.txt",
            "json": self.pending / f"{stem}.json",
        }

    def _safe_move(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    def _safe_unlink(self, path: Path) -> None:
        if path.exists():
            path.unlink()

    def _read_metadata(self, img: Path) -> dict:
        metadata_path = img.with_suffix(".json")

        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_metadata(self, metadata_path: Path, metadata: dict) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _read_label(self, img: Path) -> tuple[int | None, list[float]]:
        """
        Leser YOLO-label for et review-element.
        Returnerer class_id og polygonkoordinater. Dersom label mangler
        eller er ugyldig, returneres None og tom polygonliste.
        """
        txt = img.with_suffix(".txt")

        class_id = None
        polygon = []

        if not txt.exists():
            return class_id, polygon

        try:
            with open(txt, "r", encoding="utf-8") as f:
                data = f.readline().strip().split()

            if data:
                class_id = int(data[0])
                polygon = [float(x) for x in data[1:]]
        except (ValueError, OSError):
            pass

        return class_id, polygon

    def _update_label_class(self, label_path: Path, new_class_id: int) -> None:
        """
        Oppdaterer class_id i første linje av YOLO-labelen.
        Polygonkoordinatene beholdes uendret. Brukes når bruker korrigerer
        art i review-grensesnittet.
        """
        if not label_path.exists():
            return

        with open(label_path, "r", encoding="utf-8") as f:
            data = f.readline().strip().split()

        if not data:
            return

        data[0] = str(new_class_id)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write(" ".join(data))

    def _read_item(self, img: Path) -> dict:
        """
        Leser ett review-element og samler bilde, label og metadata.
        Returnerer et dict-objekt som kan brukes direkte av ReviewService/UI.
        """
        metadata = self._read_metadata(img)
        class_id, polygon = self._read_label(img)

        timestamp = datetime.fromtimestamp(img.stat().st_mtime).strftime("%H:%M:%S")

        return {
            "filename": img.name,
            "path": str(img),
            "class_id": metadata.get("class_id", class_id),
            "polygon": polygon,
            "timestamp": timestamp,
            "confidence": metadata.get("confidence"),
            "session_id": metadata.get("session_id"),
            "track_id": metadata.get("track_id"),
            "was_counted": metadata.get("was_counted", True),
            "source_image_path": metadata.get("source_image_path"),
            "created_at": metadata.get("created_at"),
            "metadata": metadata,
        }

    def list_pending_items(self) -> list[dict]:
        images = sorted(
            self.pending.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
        )

        return [self._read_item(img) for img in images]

    def get_next_item(self) -> dict | None:
        items = self.list_pending_items()
        return items[0] if items else None

    #---------NY: QUARANTINE QUEUE LOGIKK BLOKK---------

    def _read_class_ids_from_label_dir(self, label_dir: Path) -> set[int]:
        """
        Leser class_id-er som faktisk finnes i en label-mappe.

        Brukes for å skille mellom arter som allerede er del av stabilt
        treningsgrunnlag og nye arter som må i karantenekø.
        """
        class_ids = set()

        if not label_dir.exists():
            return class_ids

        for label_file in label_dir.glob("*.txt"):
            try:
                lines = label_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    continue

        return class_ids

    def _get_stable_class_ids(self) -> set[int]:
        """
        Returnerer class_id-er som allerede finnes i stabilt treningsgrunnlag.

        Nye arter kan ligge i species.py, men skal ikke automatisk inn i vanlig
        night training. Derfor regnes en art som stabil først når den finnes i
        master-datasettet eller allerede godkjent training_reviewed-data.
        """
        stable_ids = set()

        stable_label_dirs = [
            Path("data/master/train/labels"),
            Path("data/master/val/labels"),
            self.training_labels,
        ]

        for label_dir in stable_label_dirs:
            stable_ids.update(self._read_class_ids_from_label_dir(label_dir))

        return stable_ids

    def _is_new_species_class(self, class_id: int | None) -> bool:
        """
        Sjekker om class_id skal behandles som ny art.

        En ny art finnes i species.py/UI, men er ikke del av stabilt
        treningsgrunnlag ennå. Slike eksempler sendes til karantenekø i stedet
        for vanlig night training.
        """
        if class_id is None:
            return False

        return int(class_id) not in self._get_stable_class_ids()

    def _move_to_training_reviewed(self, paths: dict) -> None:
        """
        Flytter review-element til vanlig godkjent treningsdata.
        """
        self._safe_move(paths["img"], self.training_images / paths["img"].name)
        self._safe_move(paths["txt"], self.training_labels / paths["txt"].name)
        self._safe_move(paths["json"], self.training_metadata / paths["json"].name)

    def _move_to_new_species_queue(self, paths: dict) -> None:
        """
        Flytter review-element til karantenekø for nye arter.
        """
        base_dir = Path(".").resolve()

        move_to_new_species_queue(
            image_path=paths["img"],
            label_path=paths["txt"],
            metadata_path=paths["json"],
            base_dir=base_dir,
        )
    
    # ------------------------------------

    def action_approve(self, filename: str) -> dict:
        """
        Godkjenner AI-label som korrekt.

        Kjente arter flyttes til training_reviewed og kan brukes i vanlig
        night training. Nye arter flyttes til new_species_queue slik at de
        ikke påvirker eksisterende modell før full retrening.
        """
        paths = self._paths_for(filename)
        item = self._read_item(paths["img"]) if paths["img"].exists() else {}

        metadata = item.get("metadata", {})
        if metadata:
            metadata["review_status"] = "approved"
            metadata["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            self._write_metadata(paths["json"], metadata)

        class_id = item.get("class_id")

        if self._is_new_species_class(class_id):
            if metadata:
                metadata["review_status"] = "approved_new_species_quarantine"
                metadata["quarantine_reason"] = "class_not_in_stable_training_data"
                self._write_metadata(paths["json"], metadata)

            self._move_to_new_species_queue(paths)
        else:
            self._move_to_training_reviewed(paths)

        return item

    def action_change_species(
        self,
        filename: str,
        new_species_name: str,
        new_class_id: int | None,
    ) -> dict:
        """
        Oppdaterer artslabel for et bilde og lagrer det som godkjent treningsdata.

        - Oppdaterer class_id i YOLO-label (.txt)
        - Oppdaterer metadata med valgt art
        - Flytter bilde, label og metadata til training_reviewed

        Krever at arten finnes i species.py (må legges til via Settings først).
        """
        paths = self._paths_for(filename)
        item = self._read_item(paths["img"]) if paths["img"].exists() else {}

        old_class_id = item.get("class_id")
        metadata = item.get("metadata", {})

        if new_class_id is None:
            raise ValueError(
                f"Unknown species '{new_species_name}'. Add it in Settings before review."
            )

        self._update_label_class(paths["txt"], new_class_id)

        metadata["old_class_id"] = old_class_id
        metadata["class_id"] = new_class_id
        metadata["corrected_species_name"] = new_species_name
        metadata["needs_class_id_assignment"] = False
        metadata["review_status"] = "corrected"
        metadata["reviewed_at"] = datetime.now().isoformat(timespec="seconds")

        self._write_metadata(paths["json"], metadata)

        if self._is_new_species_class(new_class_id):
            metadata["review_status"] = "corrected_new_species_quarantine"
            metadata["quarantine_reason"] = "class_not_in_stable_training_data"
            self._write_metadata(paths["json"], metadata)

            self._move_to_new_species_queue(paths)
        else:
            self._move_to_training_reviewed(paths)

        return item

    def action_reject(self, filename: str) -> dict:
        """
        Korrigerer art for et review-element.

        Oppdaterer class_id i YOLO-labelen og metadata. Dersom arten er kjent
        i stabilt treningsgrunnlag, flyttes elementet til training_reviewed.
        Dersom arten er ny, flyttes elementet til new_species_queue.
        """
        paths = self._paths_for(filename)
        item = self._read_item(paths["img"]) if paths["img"].exists() else {}

        metadata = item.get("metadata", {})
        if metadata:
            metadata["review_status"] = "rejected"
            metadata["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            self._write_metadata(paths["json"], metadata)

        self._safe_move(paths["img"], self.rejected_images / paths["img"].name)
        self._safe_move(paths["json"], self.rejected_metadata / paths["json"].name)
        self._safe_unlink(paths["txt"])

        return item

    def action_send_to_land(self, filename: str) -> dict:
        """
        Flytter uklart eller korrupt review-element til manuell sync-kø.
        Brukes når elementet ikke bør godkjennes eller avvises direkte om bord.
        """
        paths = self._paths_for(filename)
        item = self._read_item(paths["img"]) if paths["img"].exists() else {}

        metadata = item.get("metadata", {})
        if metadata:
            metadata["review_status"] = "sent_to_land"
            metadata["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            self._write_metadata(paths["json"], metadata)

        self._safe_move(paths["img"], self.sync / paths["img"].name)
        self._safe_move(paths["json"], self.sync / paths["json"].name)
        self._safe_unlink(paths["txt"])

        return item