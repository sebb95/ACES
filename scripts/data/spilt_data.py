import os
import shutil
import random
from collections import defaultdict

# --- STIER ---
# Mappen der ALLE filene dine ligger nå:
source_dir = r"C:\ACES\Data_Aces\Processed_data\Yolo_exported_2500" 
# Hvor du vil at den ferdige strukturen skal lande:
target_dir = r"C:\ACES\Data_Aces\Processed_data\Data_v1_2500_TrainReady"

# Opprett mapper
for split in ['train', 'val']:
    os.makedirs(f"{target_dir}/{split}/images", exist_ok=True)
    os.makedirs(f"{target_dir}/{split}/labels", exist_ok=True)

# 1. Grupper filer etter art
# Vi leser første tall i hver txt-fil (klassen)
species_files = defaultdict(list)
for label_file in os.listdir(source_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(source_dir, label_file), 'r') as f:
            first_line = f.readline().split()
            if first_line:
                class_id = first_line[0]
                species_files[class_id].append(label_file)

# 2. Fordel hver art 80/20
for class_id, files in species_files.items():
    random.shuffle(files)
    split_idx = int(len(files) * 0.8)
    
    train_set = files[:split_idx]
    val_set = files[split_idx:]
    
    # Flytt filer
    def move_data(file_list, split):
        for f in file_list:
            # Flytt label
            shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, split, "labels", f))
            # Flytt bilde (sjekk både png og jpg)
            img_name = f.replace('.txt', '.png')
            if not os.path.exists(os.path.join(source_dir, img_name)):
                img_name = f.replace('.txt', '.jpg')
            shutil.copy(os.path.join(source_dir, img_name), os.path.join(target_dir, split, "images", img_name))

    move_data(train_set, 'train')
    move_data(val_set, 'val')

print("Datasettet er nå perfekt splittet 80/20 per art!")