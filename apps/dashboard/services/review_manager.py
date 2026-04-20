import shutil
from pathlib import Path
from datetime import datetime
from src.common.species import CLASS_NAMES, NAME_TO_CLASS_ID


class ReviewManager:
    """Backend for UI. Håndterer alle Active Learning filoperasjoner."""

    def __init__(self):
        self.pending = Path("data/review_queue/pending")
        self.appr_img = Path("data/review_queue/approved_today/images")
        self.appr_lbl = Path("data/review_queue/approved_today/labels")
        self.sync = Path("data/sync_queue")

        for d in [self.pending, self.appr_img, self.appr_lbl, self.sync]:
            d.mkdir(parents=True, exist_ok=True)

    def _read_item(self, img: Path) -> dict:
        txt = img.with_suffix(".txt")

        cls_id = 4
        poly = []

        if txt.exists():
            with open(txt, "r", encoding="utf-8") as f:
                data = f.readline().strip().split()
                if data:
                    cls_id = int(data[0])
                    poly = [float(x) for x in data[1:]]

        timestamp = datetime.fromtimestamp(img.stat().st_mtime).strftime("%H:%M:%S")

        return {
            "filename": img.name,
            "path": str(img),
            "class_id": cls_id,
            "polygon": poly,
            "timestamp": timestamp,
            "confidence": None,  # not available yet
        }

    def list_pending_items(self) -> list[dict]:
        """Returnerer alle ventende bilder i stabil rekkefølge."""
        images = sorted(self.pending.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        return [self._read_item(img) for img in images]

    def get_next_item(self) -> dict | None:
        """Returnerer neste bilde til GUI, eller None hvis tomt."""
        items = self.list_pending_items()
        return items[0] if items else None

    def action_approve(self, filename: str):
        img = self.pending / filename
        txt = img.with_suffix(".txt")

        if img.exists():
            shutil.move(str(img), str(self.appr_img / img.name))
        if txt.exists():
            shutil.move(str(txt), str(self.appr_lbl / txt.name))

    def action_reject(self, filename: str):
        """False positive / slett label, men behold bilde i approved_today/images."""
        img = self.pending / filename
        txt = img.with_suffix(".txt")

        if img.exists():
            shutil.move(str(img), str(self.appr_img / img.name))
        if txt.exists():
            txt.unlink()

    def action_change_species(self, filename: str, new_class_id: int):
        txt = (self.pending / filename).with_suffix(".txt")

        if txt.exists():
            with open(txt, "r", encoding="utf-8") as f:
                data = f.readline().strip().split()

            if data:
                data[0] = str(new_class_id)
                with open(txt, "w", encoding="utf-8") as f:
                    f.write(" ".join(data))

        self.action_approve(filename)

    def action_send_to_land(self, filename: str):
        """Korrupt maske / send til sync-kø."""
        img = self.pending / filename
        txt = img.with_suffix(".txt")

        if img.exists():
            shutil.move(str(img), str(self.sync / img.name))
        if txt.exists():
            txt.unlink()