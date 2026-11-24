"""Variables du modèle retenu (modèle C du notebook 03) : log du prix expliqué par
log(surface), pièces, dépendance, année et une indicatrice par commune.

Ce module est partagé par l'entraînement (src/train.py) et l'application (app.py),
pour que les variables soient construites exactement de la même façon dans les deux cas.
"""

import numpy as np
import pandas as pd

ANNEE_DEBUT = 2021
MIN_VENTES_COMMUNE = 10  # en dessous, la commune est regroupée avec les petites communes du département


def communes_a_garder(code_commune: pd.Series) -> list[str]:
    """Communes ayant assez de ventes pour recevoir leur propre indicatrice."""
    ventes = code_commune.value_counts()
    return sorted(ventes[ventes >= MIN_VENTES_COMMUNE].index)


def zone(df: pd.DataFrame, communes_gardees: list[str]) -> pd.Series:
    """La commune si elle est gardée, sinon « autres communes du département »."""
    gardee = df["code_commune"].isin(communes_gardees)
    return df["code_commune"].where(gardee, "autres_" + df["code_departement"])


def construire_variables(df: pd.DataFrame, communes_gardees: list[str],
                         colonnes: list[str] | None = None) -> pd.DataFrame:
    """Construit la matrice des variables explicatives.

    - Apprentissage (colonnes=None) : une zone de référence est retirée pour éviter
      la colinéarité parfaite entre les indicatrices et la constante.
    - Prédiction (colonnes données) : on réaligne sur les colonnes de l'apprentissage.
    """
    X = pd.DataFrame({
        "log_surface": np.log(df["surface"]),
        "pieces": df["pieces"],
        "a_dependance": df["a_dependance"].astype(int),
        "annee": df["annee"] - ANNEE_DEBUT,
    }, index=df.index)
    zones = pd.get_dummies(zone(df, communes_gardees), prefix="zone", dtype=int)
    if colonnes is None:
        return X.join(zones.iloc[:, 1:])
    return X.join(zones).reindex(columns=colonnes, fill_value=0)
