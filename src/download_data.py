"""Télécharge les données DVF géolocalisées de la région Centre-Val de Loire.

Source : https://files.data.gouv.fr/geo-dvf/ (Etalab, Licence Ouverte 2.0).
Un fichier par département et par année est enregistré dans data/raw/.
Les fichiers déjà présents ne sont pas retéléchargés.

Utilisation : python src/download_data.py
"""

import sys
import urllib.request
from pathlib import Path

ANNEES = range(2021, 2026)  # de 2021 à 2025 inclus
DEPARTEMENTS = {
    "18": "Cher",
    "28": "Eure-et-Loir",
    "36": "Indre",
    "37": "Indre-et-Loire",
    "41": "Loir-et-Cher",
    "45": "Loiret",
}
URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"

# data/raw/ à la racine du projet, quel que soit le dossier d'où l'on lance le script
DOSSIER_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def chemin_fichier(annee: int, dep: str) -> Path:
    return DOSSIER_RAW / f"dvf_{annee}_{dep}.csv.gz"


def telecharger(annee: int, dep: str) -> bool:
    """Télécharge le fichier d'une année et d'un département.

    Renvoie True si le fichier vient d'être téléchargé, False s'il était déjà là.
    """
    destination = chemin_fichier(annee, dep)
    if destination.exists():
        return False

    # On écrit d'abord dans un fichier temporaire : si le téléchargement est
    # interrompu, aucun fichier incomplet ne reste sous le nom définitif.
    temporaire = destination.with_suffix(".part")
    try:
        urllib.request.urlretrieve(URL.format(annee=annee, dep=dep), temporaire)
        temporaire.rename(destination)
    finally:
        temporaire.unlink(missing_ok=True)
    return True


def main() -> int:
    DOSSIER_RAW.mkdir(parents=True, exist_ok=True)
    fichiers = [(annee, dep) for annee in ANNEES for dep in DEPARTEMENTS]
    echecs = []

    for i, (annee, dep) in enumerate(fichiers, start=1):
        print(f"[{i:2}/{len(fichiers)}] {annee} {dep} {DEPARTEMENTS[dep]:<15}", end="", flush=True)
        try:
            nouveau = telecharger(annee, dep)
        except OSError as erreur:  # coupure réseau, fichier absent sur le serveur…
            print(f"ÉCHEC ({erreur})")
            echecs.append(f"{annee}-{dep}")
            continue
        taille_mo = chemin_fichier(annee, dep).stat().st_size / 1e6
        etat = "téléchargé" if nouveau else "déjà présent"
        print(f"{etat} ({taille_mo:.1f} Mo)")

    if echecs:
        print(f"\n{len(echecs)} échec(s) : {', '.join(echecs)}. Relancez le script pour réessayer.")
        return 1
    print(f"\nTerminé : {len(fichiers)} fichiers dans {DOSSIER_RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
