"""Construit la table propre des ventes d'appartements à partir des fichiers DVF bruts.

Règles issues du notebook 01_exploration :
1. ventes classiques uniquement (nature_mutation == "Vente") ;
2. une vente = un seul appartement, sans maison ni local (les dépendances sont acceptées) ;
3. retrait des valeurs aberrantes (prix, prix au m², surface, nombre de pièces).

Entrée : data/raw/dvf_*.csv.gz
Sortie : data/processed/apparts.parquet (une ligne par vente)

Utilisation : python src/prepare_data.py
"""

from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_RAW = RACINE / "data" / "raw"
FICHIER_SORTIE = RACINE / "data" / "processed" / "apparts.parquet"

# Seuils des valeurs aberrantes (justifiés dans le notebook 01)
PRIX_MIN = 10_000
PRIX_M2_MIN, PRIX_M2_MAX = 300, 8_000
SURFACE_MIN, SURFACE_MAX = 9, 250  # 9 m² : surface minimale légale d'un logement

COLONNES = [
    "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere", "type_local",
    "surface_reelle_bati", "nombre_pieces_principales", "code_departement",
    "code_commune", "nom_commune", "longitude", "latitude",
]
TYPES = {"code_departement": str, "code_commune": str}


def charger_brut() -> pd.DataFrame:
    fichiers = sorted(DOSSIER_RAW.glob("dvf_*.csv.gz"))
    if not fichiers:
        raise FileNotFoundError("Aucun fichier dans data/raw : lancez d'abord python src/download_data.py")
    return pd.concat([pd.read_csv(f, usecols=COLONNES, dtype=TYPES) for f in fichiers], ignore_index=True)


def garder_ventes_un_appartement(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les ventes classiques contenant exactement un appartement et aucune maison ni local.

    Une vente (id_mutation) s'étale sur plusieurs lignes et son prix est recopié sur chacune :
    on raisonne donc vente par vente, puis on garde la seule ligne « Appartement » de chaque vente.
    """
    ventes = df[df["nature_mutation"] == "Vente"]
    type_local = ventes["type_local"]
    composition = (
        ventes.assign(
            n_appartements=type_local.eq("Appartement"),
            n_autres_biens=type_local.eq("Maison") | type_local.str.startswith("Local", na=False),
            n_dependances=type_local.eq("Dépendance"),
        )
        .groupby("id_mutation")[["n_appartements", "n_autres_biens", "n_dependances"]]
        .sum()
    )
    retenues = composition[(composition["n_appartements"] == 1) & (composition["n_autres_biens"] == 0)]

    apparts = ventes[ventes["id_mutation"].isin(retenues.index) & type_local.eq("Appartement")].copy()
    apparts["a_dependance"] = apparts["id_mutation"].map(retenues["n_dependances"] > 0)
    return apparts


def retirer_aberrants(apparts: pd.DataFrame) -> pd.DataFrame:
    prix = apparts["valeur_fonciere"]
    surface = apparts["surface_reelle_bati"]
    garder = (
        (prix >= PRIX_MIN)
        & (prix / surface).between(PRIX_M2_MIN, PRIX_M2_MAX)
        & surface.between(SURFACE_MIN, SURFACE_MAX)
        & (apparts["nombre_pieces_principales"] >= 1)
    )
    return apparts[garder]


def mettre_en_forme(apparts: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(apparts["date_mutation"])
    return pd.DataFrame({
        "id_mutation": apparts["id_mutation"],
        "date": date,
        "annee": date.dt.year,
        "prix": apparts["valeur_fonciere"],
        "surface": apparts["surface_reelle_bati"],
        "pieces": apparts["nombre_pieces_principales"].astype(int),
        "a_dependance": apparts["a_dependance"],
        "code_departement": apparts["code_departement"],
        "code_commune": apparts["code_commune"],
        "nom_commune": apparts["nom_commune"],
        "latitude": apparts["latitude"],
        "longitude": apparts["longitude"],
    }).reset_index(drop=True)


def preparer() -> pd.DataFrame:
    brut = charger_brut()
    print(f"Lignes brutes                          : {len(brut):>9,}")
    apparts = garder_ventes_un_appartement(brut)
    print(f"Ventes d'un seul appartement           : {len(apparts):>9,}")
    apparts = retirer_aberrants(apparts)
    print(f"Après retrait des valeurs aberrantes   : {len(apparts):>9,}")
    return mettre_en_forme(apparts)


def main() -> None:
    apparts = preparer()
    FICHIER_SORTIE.parent.mkdir(parents=True, exist_ok=True)
    apparts.to_parquet(FICHIER_SORTIE, index=False)
    print(f"Table enregistrée : {FICHIER_SORTIE} ({len(apparts):,} ventes)")


if __name__ == "__main__":
    main()
