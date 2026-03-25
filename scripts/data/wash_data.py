import os
import json

# --- LEGG INN STIEN TIL MAPPEN DIN HER ---
folder_path = r"C:\Users\PC\Documents\Datateknikk-23\1.Batchoppgv\Data\Labeled_data" 

# --- FASITEN DIN (Hvilke navn vil du egentlig ha?) ---
# Scriptet vil gjøre alt til små bokstaver først, og så slå opp i denne listen.
correct_names = {
    "hyse": "Hyse",
    "torsk": "Torsk",
    "sei": "Sei",
    "lange": "Lange",
    "uer": "Uer",
    "bifangst": "Bifangst"
    # Legg til flere hvis du har feilstavelser, f.eks: "torssk": "Torsk"
}

fixed_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        changed = False
        
        # Gå gjennom alle polygoner (fisker) i bildet
        for shape in data['shapes']:
            old_label = shape['label']
            
            # Fjerner mellomrom foran/bak og gjør til små bokstaver
            clean_label = old_label.strip().lower() 
            
            # Hvis vi finner den i fasiten vår, bytt den ut!
            if clean_label in correct_names:
                new_label = correct_names[clean_label]
                if old_label != new_label:
                    shape['label'] = new_label
                    changed = True
            else:
                print(f"ADVARSEL: Fant et ukjent navn '{old_label}' i {filename}")

        # Hvis vi endret noe, lagre filen på nytt
        if changed:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            fixed_count += 1

print(f"Ferdig! Vasket og fikset labels i {fixed_count} filer.")