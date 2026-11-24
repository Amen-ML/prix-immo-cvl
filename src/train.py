"""Entraîne le modèle final et l'enregistre pour l'application Streamlit.

1. Évaluation : apprentissage sur 2021-2024, test sur 2025 (comme dans le notebook 03).
2. Modèle final : réentraîné sur toutes les années, pour profiter des ventes les plus récentes.

Entrée : data/processed/apparts.parquet
Sortie : models/modele_prix.joblib

Utilisation (depuis la racine du projet) : python -m src.train
"""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from src.modele import communes_a_garder, construire_variables

RACINE = Path(__file__).resolve().parent.parent
FICHIER_DONNEES = RACINE / "data" / "processed" / "apparts.parquet"
FICHIER_MODELE = RACINE / "models" / "modele_prix.joblib"


def entrainer(df: pd.DataFrame):
    communes = communes_a_garder(df["code_commune"])
    X = construire_variables(df, communes)
    modele = LinearRegression().fit(X, np.log(df["prix"]))
    return modele, communes, list(X.columns)


def predire(modele, communes, colonnes, df: pd.DataFrame) -> np.ndarray:
    return np.exp(modele.predict(construire_variables(df, communes, colonnes)))


def statistiques_communes(apparts: pd.DataFrame) -> pd.DataFrame:
    """Liste des communes pour l'application, avec le prix médian observé au m²."""
    return (
        apparts.sort_values("date")
        .assign(prix_m2=apparts["prix"] / apparts["surface"])
        .groupby("code_commune", as_index=False)
        .agg(nom_commune=("nom_commune", "last"), code_departement=("code_departement", "first"),
             ventes=("prix", "size"), prix_m2_median=("prix_m2", "median"))
        .sort_values("nom_commune", ignore_index=True)
    )


def main() -> None:
    apparts = pd.read_parquet(FICHIER_DONNEES)
    train = apparts[apparts["annee"] <= 2024]
    test = apparts[apparts["annee"] == 2025]

    # 1. Évaluation honnête : le modèle n'a jamais vu les ventes de 2025
    modele, communes, colonnes = entrainer(train)
    prediction = predire(modele, communes, colonnes, test)
    ratio = test["prix"] / prediction
    evaluation = {
        "mae": mean_absolute_error(test["prix"], prediction),
        "r2": r2_score(test["prix"], prediction),
        "ratio_q10": ratio.quantile(0.10),  # 80 % des prix réels sont entre q10 et q90 fois l'estimation
        "ratio_q90": ratio.quantile(0.90),
        "ventes_test": len(test),
    }
    print(f"Test 2025 : MAE = {evaluation['mae']:,.0f} €, R² = {evaluation['r2']:.3f}")
    print(f"80 % des prix réels entre {evaluation['ratio_q10']:.2f} et {evaluation['ratio_q90']:.2f} fois l'estimation")

    # 2. Modèle final, entraîné sur 2021-2025
    modele, communes, colonnes = entrainer(apparts)
    paquet = {
        "modele": modele,
        "communes_gardees": communes,
        "colonnes": colonnes,
        "communes": statistiques_communes(apparts),
        "annee_max": int(apparts["annee"].max()),
        "ventes_apprentissage": len(apparts),
        "evaluation": evaluation,
        "date_entrainement": date.today().isoformat(),
    }
    FICHIER_MODELE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquet, FICHIER_MODELE)
    print(f"Modèle final ({len(apparts):,} ventes, {len(colonnes)} variables) enregistré : {FICHIER_MODELE}")


if __name__ == "__main__":
    main()
