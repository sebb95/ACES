from typing import Any


class WeightManager:
    """
    Beregner estimert fangstvekt basert på antall fisk per art.

    Klassen mottar artsfordeling fra en økt eller tur, samt konfigurerte
    gjennomsnittsvekter per art fra runtime_config. Den beregner total
    fangstvekt, vekt per art og egne summer for Torsk, Sei og Bifangst.

    Vektberegningen er en prototypebasert estimatmodell:
        estimert vekt = antall fisk * gjennomsnittsvekt per art

    Ansvar:
    - beregne total_count og total_weight_kg
    - beregne artsvis vektfordeling
    - skille ut Torsk og Sei som hovedarter
    - aggregere øvrige arter som Bifangst
    """
    TARGET_SPECIES = {"Torsk", "Sei"}

    def __init__(self, species_weights: dict[str, float]) -> None:
        self.species_weights = species_weights or {}

    def calculate(self, species_counts: dict[str, int]) -> dict[str, Any]:
        """
        Beregner vektoppsummering fra artsfordeling.

        Args:
            species_counts: Dictionary med artsnavn som nøkkel og antall fisk som verdi.

        Returns:
            Dictionary med totalantall, totalvekt, egne summer for Torsk og Sei,
            aggregert Bifangst og full artsvis fordeling.
        """
        species_breakdown = {}

        total_count = 0
        total_weight_kg = 0.0

        for species_name, count in species_counts.items():
            count = int(count)
            avg_weight_kg = float(self.species_weights.get(species_name, 0.0))
            weight_kg = count * avg_weight_kg

            species_breakdown[species_name] = {
                "name": species_name,
                "count": count,
                "average_weight_kg": avg_weight_kg,
                "weight_kg": round(weight_kg, 2),
            }

            total_count += count
            total_weight_kg += weight_kg

        torsk = species_breakdown.get(
            "Torsk",
            {
                "name": "Torsk",
                "count": 0,
                "average_weight_kg": float(self.species_weights.get("Torsk", 0.0)),
                "weight_kg": 0.0,
            },
        )

        sei = species_breakdown.get(
            "Sei",
            {
                "name": "Sei",
                "count": 0,
                "average_weight_kg": float(self.species_weights.get("Sei", 0.0)),
                "weight_kg": 0.0,
            },
        )

        bifangst_count = 0
        bifangst_weight_kg = 0.0
        bifangst_species = []

        for species_name, values in species_breakdown.items():
            if species_name in self.TARGET_SPECIES:
                continue

            bifangst_count += values["count"]
            bifangst_weight_kg += values["weight_kg"]
            bifangst_species.append(values)

        return {
            "total_count": total_count,
            "total_weight_kg": round(total_weight_kg, 2),
            "torsk": torsk,
            "sei": sei,
            "bifangst": {
                "name": "Bifangst",
                "count": bifangst_count,
                "weight_kg": round(bifangst_weight_kg, 2),
                "species": bifangst_species,
            },
            "species_breakdown": list(species_breakdown.values()),
        }