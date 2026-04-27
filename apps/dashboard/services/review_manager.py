import json
import shutil
from datetime import datetime
from pathlib import Path


class ReviewManager:
    """Backend for active learning review file operations."""

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

    def action_approve(self, filename: str) -> dict:
        """
        Accept AI label as correct training data.
        """
        paths = self._paths_for(filename)
        item = self._read_item(paths["img"]) if paths["img"].exists() else {}

        metadata = item.get("metadata", {})
        if metadata:
            metadata["review_status"] = "approved"
            metadata["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            self._write_metadata(paths["json"], metadata)

        self._safe_move(paths["img"], self.training_images / paths["img"].name)
        self._safe_move(paths["txt"], self.training_labels / paths["txt"].name)
        self._safe_move(paths["json"], self.training_metadata / paths["json"].name)

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

        self._safe_move(paths["img"], self.training_images / paths["img"].name)
        self._safe_move(paths["txt"], self.training_labels / paths["txt"].name)
        self._safe_move(paths["json"], self.training_metadata / paths["json"].name)

        return item

    def action_reject(self, filename: str) -> dict:
        """
        Reject as false positive / unusable.

        Image and metadata are kept for audit.
        Label is deleted because it should not become training data.
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
        Move unclear/corrupt item to manual sync queue.
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