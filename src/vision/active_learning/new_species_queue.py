"""
Karantenekø for nye arter i ACES (Active Learning pipeline).

Denne modulen håndterer:
- lagring av godkjente eksempler på nye arter som ennå ikke er del av stabilt treningsgrunnlag
- telling av antall eksempler per class_id
- sjekk for når en ny art har nok data til full retrening

Design:
- Nye arter legges først i new_species_queue og holdes utenfor vanlig night training
- Når en art har samlet tilstrekkelig data (f.eks. ≥100 eksempler),
  kan den inkluderes i en full retrening av modellen
- Hindrer at YOLO-modellens head rebuildes under små inkrementelle oppdateringer,
  og beskytter dermed mot tap av eksisterende nøyaktighet (catastrophic forgetting)

Denne modulen inneholder kun ren logikk og filoperasjoner.
Selve treningen håndteres av egne treningsklasser.
"""

from pathlib import Path
from collections import Counter
import shutil
import json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_new_species_root(base_dir: Path) -> Path:
    return base_dir / "data" / "new_species_queue"


def ensure_new_species_dirs(base_dir: Path) -> dict[str, Path]:
    """
    Oppretter mappestruktur for karantenekø for nye arter.

    Nye arter holdes utenfor vanlig night training frem til det finnes
    nok kvalitetssikrede eksempler til full retrening.
    """
    root = get_new_species_root(base_dir)

    paths = {
        "root": root,
        "images": root / "images",
        "labels": root / "labels",
        "metadata": root / "metadata",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def move_to_new_species_queue(
    image_path: Path,
    label_path: Path,
    metadata_path: Path | None,
    base_dir: Path,
) -> None:
    """
    Flytter et godkjent review-element til karantenekø for ny art.

    Brukes når en art finnes i species.py/UI, men ikke skal inn i vanlig
    night training før nok eksempler er samlet.
    """
    paths = ensure_new_species_dirs(base_dir)

    shutil.move(str(image_path), str(paths["images"] / image_path.name))
    shutil.move(str(label_path), str(paths["labels"] / label_path.name))

    if metadata_path and metadata_path.exists():
        shutil.move(str(metadata_path), str(paths["metadata"] / metadata_path.name))


def count_samples_by_class(base_dir: Path) -> dict[int, int]:
    """
    Teller hvor mange label-filer som finnes per class_id i karantenekøen.
    """
    paths = ensure_new_species_dirs(base_dir)
    counts: Counter[int] = Counter()

    for label_file in paths["labels"].glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            counts[class_id] += 1

    return dict(counts)


def get_classes_ready_for_full_retrain(
    base_dir: Path,
    min_samples: int = 100,
) -> dict[int, int]:
    """
    Returnerer nye arter som har nok eksempler til full retrening.
    """
    counts = count_samples_by_class(base_dir)

    return {
        class_id: count
        for class_id, count in counts.items()
        if count >= min_samples
    }

def get_full_retrain_status(
    base_dir: Path,
    min_samples: int = 100,
) -> dict:
    """
    Returnerer status for full retrening basert på karantenekøen.

    Brukes av admin-script eller senere admin-UI for å se hvilke nye arter
    som har nok godkjente eksempler til full retrening.
    """
    counts = count_samples_by_class(base_dir)

    ready_classes = {
        class_id: count
        for class_id, count in counts.items()
        if count >= min_samples
    }

    return {
        "min_samples": min_samples,
        "counts": counts,
        "ready_classes": ready_classes,
        "has_ready_classes": bool(ready_classes),
    }