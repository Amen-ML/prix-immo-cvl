"""Application Streamlit : estimer le prix d'un appartement en Centre-Val de Loire.

Lancement (depuis la racine du projet) : streamlit run app.py
Le modèle doit avoir été entraîné avant : python -m src.train
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.modele import construire_variables

FICHIER_MODELE = Path(__file__).parent / "models" / "modele_prix.joblib"
NOMS_DEPARTEMENTS = {"18": "Cher", "28": "Eure-et-Loir", "36": "Indre",
                     "37": "Indre-et-Loire", "41": "Loir-et-Cher", "45": "Loiret"}


def nombre(valeur: float) -> str:
    return f"{valeur:,.0f}".replace(",", " ")


def euros(montant: float) -> str:
    return f"{nombre(montant)} €"


@st.cache_resource
def charger_modele() -> dict:
    return joblib.load(FICHIER_MODELE)


st.set_page_config(page_title="Prix d'un appartement en Centre-Val de Loire", page_icon="🏠")

paquet = charger_modele()
communes = paquet["communes"]
evaluation = paquet["evaluation"]
annee = paquet["annee_max"]

st.title("Estimer le prix d'un appartement en Centre-Val de Loire")
st.caption(f"Régression linéaire entraînée sur {nombre(paquet['ventes_apprentissage'])} ventes réelles "
           f"(données DVF 2021-{annee})")

# --- Saisie ---------------------------------------------------------------
libelles = (communes["nom_commune"] + " (" + communes["code_departement"] + ")").tolist()
position_tours = libelles.index("Tours (37)") if "Tours (37)" in libelles else 0
choix = st.selectbox("Commune", range(len(libelles)), index=position_tours, format_func=lambda i: libelles[i])
commune = communes.iloc[choix]

colonne_1, colonne_2 = st.columns(2)
surface = colonne_1.number_input("Surface habitable (m²)", min_value=9, max_value=250, value=60, step=1)
pieces = colonne_2.number_input("Nombre de pièces principales", min_value=1, max_value=10, value=3, step=1)
a_dependance = st.checkbox("Vendu avec une cave, un garage ou une place de parking", value=True)

# --- Estimation -----------------------------------------------------------
vente = pd.DataFrame({
    "surface": [surface], "pieces": [pieces], "a_dependance": [a_dependance], "annee": [annee],
    "code_commune": [commune["code_commune"]], "code_departement": [commune["code_departement"]],
})
X = construire_variables(vente, paquet["communes_gardees"], paquet["colonnes"])
prix = float(np.exp(paquet["modele"].predict(X))[0])
bas, haut = prix * evaluation["ratio_q10"], prix * evaluation["ratio_q90"]

st.divider()
resultat_1, resultat_2 = st.columns(2)
resultat_1.metric(f"Prix estimé ({annee})", euros(prix))
resultat_2.metric("Soit au m²", euros(prix / surface))
st.markdown(f"**Fourchette à 80 % : {euros(bas)} – {euros(haut)}**  \n"
            f"Sur les ventes de {annee}, 8 prix réels sur 10 tombaient dans cet intervalle autour de l'estimation.")

if commune["code_commune"] in paquet["communes_gardees"]:
    st.info(f"Prix médian observé à {commune['nom_commune']} : **{euros(commune['prix_m2_median'])}/m²** "
            f"sur {nombre(commune['ventes'])} ventes d'appartements (2021-{annee}).")
else:
    departement = NOMS_DEPARTEMENTS.get(commune["code_departement"], commune["code_departement"])
    st.warning(f"{commune['nom_commune']} compte moins de 10 ventes d'appartements : l'estimation utilise "
               f"le niveau de prix moyen des petites communes du département ({departement}).")

# --- Explications ---------------------------------------------------------
with st.expander("Comment fonctionne ce modèle ?"):
    st.markdown(f"""
- **Données** : ventes d'appartements enregistrées par les notaires (DVF, data.gouv.fr), 2021-{annee}, 6 départements.
- **Modèle** : régression linéaire sur le **logarithme du prix**, avec la surface (en log), le nombre de pièces,
  la présence d'une dépendance, l'année de vente et une variable par commune.
- **Précision** : testé sur les {nombre(evaluation['ventes_test'])} ventes de {annee} sans les avoir vues,
  le modèle se trompe en moyenne de **{euros(evaluation['mae'])}** (R² = {evaluation['r2']:.2f}).
- **Limites** : l'étage, l'état du bien, le DPE ou le quartier précis ne figurent pas dans les données.
  L'estimation est un ordre de grandeur, pas une expertise.
""")

st.caption("Code source et démarche complète : [github.com/Amen-ML/prix-immo-cvl](https://github.com/Amen-ML/prix-immo-cvl)")
