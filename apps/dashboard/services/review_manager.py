import shutil
from pathlib import Path

class ReviewManager:
    """Backend for UI. Håndterer alle Active Learning filoperasjoner."""
    def __init__(self):
        self.pending = Path("data/review_queue/pending")
        self.appr_img = Path("data/review_queue/approved_today/images")
        self.appr_lbl = Path("data/review_queue/approved_today/labels")
        self.sync = Path("data/sync_queue")
        
        for d in [self.pending, self.appr_img, self.appr_lbl, self.sync]:
            d.mkdir(parents=True, exist_ok=True)

    def get_next_item(self) -> dict:
        """Returnerer neste bilde til GUI, eller None hvis tomt."""
        images = list(self.pending.glob("*.jpg"))
        if not images: return None
        img, txt = images[0], images[0].with_suffix(".txt")
        
        cls_id = 4
        poly = []
        if txt.exists():
            with open(txt, "r") as f:
                data = f.readline().strip().split()
                if data:
                    cls_id, poly = int(data[0]), [float(x) for x in data[1:]]
                    
        return {"filename": img.name, "path": str(img), "class_id": cls_id, "polygon": poly}

    def action_approve(self, filename: str):
        img, txt = self.pending / filename, (self.pending / filename).with_suffix(".txt")
        shutil.move(str(img), str(self.appr_img / img.name))
        if txt.exists(): shutil.move(str(txt), str(self.appr_lbl / txt.name))

    def action_reject(self, filename: str): # False positive / Slett
        img, txt = self.pending / filename, (self.pending / filename).with_suffix(".txt")
        shutil.move(str(img), str(self.appr_img / img.name))
        if txt.exists(): txt.unlink() 

    def action_change_species(self, filename: str, new_class_id: int):
        txt = (self.pending / filename).with_suffix(".txt")
        if txt.exists():
            with open(txt, "r") as f: data = f.readline().strip().split()
            if data:
                data[0] = str(new_class_id)
                with open(txt, "w") as f: f.write(" ".join(data))
        self.action_approve(filename)

    def action_send_to_land(self, filename: str): # Korrupt maske
        img, txt = self.pending / filename, (self.pending / filename).with_suffix(".txt")
        shutil.move(str(img), str(self.sync / img.name))
        if txt.exists(): txt.unlink()