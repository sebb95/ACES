from pathlib import Path


class FullRetrainOperations:
    """
    Håndterer full retrening av modellen når nye arter er klare.

    Denne klassen er ansvarlig for å:
    - identifisere nye arter som har nok data i new_species_queue
    - samle og slå sammen dataset (master + nye arter)
    - regenerere YAML-konfigurasjon for full trening
    - kjøre full YOLO-trening (f.eks. 200 epochs)
    - evaluere modellens kvalitet (quality gate)
    - promotere ny modell til produksjon dersom den er bedre

    Viktig:
    - Skal kun kjøres manuelt (admin / script), ikke automatisk nightly
    - Brukes for å introdusere nye arter i modellen uten å ødelegge eksisterende ytelse
    """

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[3]

        # Paths
        self.data_dir = self.base_dir / "data"
        self.new_species_dir = self.data_dir / "new_species_queue"
        self.master_dir = self.data_dir / "master"

        self.outputs_dir = self.base_dir / "outputs"
        self.weights_dir = self.outputs_dir / "weights"
