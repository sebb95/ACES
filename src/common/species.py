"""
Felles artsregister for ACES.

Denne filen er kilde til sannhet for class_id og artsnavn i systemet.
Den brukes av:
- deteksjon og tracking
- review-grensesnittet
- settings-siden
- YAML-generering for trening
- rapportering og vektestimering

Viktig:
Nye arter kan legges til her før modellen er trent på dem.
Slike arter skal holdes utenfor vanlig night training og samles i
new_species_queue frem til det finnes nok data for full retrening.
"""

CLASS_NAMES = {
    0: "Breiflab",
    1: "Brosme",
    2: "Flyndre",
    3: "Hyse",
    4: "Kveite",
    5: "Lange",
    6: "Lyr",
    7: "Sei",
    8: "Torsk",
    9: "Uer",
    10: "Bifangst",
    11: "Ukjent",
}

NAME_TO_CLASS_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}