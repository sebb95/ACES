import os
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = r"C:\Users\PC\Documents\Datateknikk-23\1.Batchoppgv\Data"   # Where all 2500 images are
LABELED_DIR = r"C:\Users\PC\Documents\Datateknikk-23\1.Batchoppgv\Data\Labeled_data"
UNLABELED_DIR = r"C:\Users\PC\Documents\Datateknikk-23\1.Batchoppgv\Data\Unlabeled_data"

os.makedirs(LABELED_DIR, exist_ok=True)
os.makedirs(UNLABELED_DIR, exist_ok=True)

# --- SORTING LOGIC ---
moved_labeled = 0
moved_unlabeled = 0

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(SOURCE_DIR, f"{base_name}.json")
        img_path = os.path.join(SOURCE_DIR, filename)
        
        # If a JSON exists, it is labeled!
        if os.path.exists(json_path):
            shutil.move(img_path, os.path.join(LABELED_DIR, filename))
            shutil.move(json_path, os.path.join(LABELED_DIR, f"{base_name}.json"))
            moved_labeled += 1
        else:
            # No JSON means it's unlabeled
            shutil.move(img_path, os.path.join(UNLABELED_DIR, filename))
            moved_unlabeled += 1

print(f"Done! Moved {moved_labeled} labeled images and {moved_unlabeled} unlabeled images.")