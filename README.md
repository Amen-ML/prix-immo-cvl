# Estimer le prix d'un appartement en Centre-Val de Loire

Projet de régression linéaire sur les ventes immobilières réelles (données DVF) de la région Centre-Val de Loire, de 2021 à 2025.

## Question

À partir de sa surface, de son nombre de pièces, de sa localisation et de la date de vente, peut-on estimer le prix d'un appartement en Centre-Val de Loire, et avec quelle marge d'erreur ?

Pour qui : un particulier qui veut savoir si le prix d'une annonce est cohérent avec le marché, ou un vendeur qui veut fixer un prix réaliste avant de mettre son bien en vente.

## Données

- **Source** : Demandes de valeurs foncières (DVF) géolocalisées, produites par Etalab à partir des données de la DGFiP, disponibles sur data.gouv.fr
- **Licence** : Licence Ouverte / Etalab 2.0
- **Contenu** : chaque vente immobilière enregistrée chez un notaire, avec le prix, la date, le type de bien, la surface, le nombre de pièces, l'adresse et les coordonnées GPS
- Les fichiers bruts (un par département et par année) ne sont pas versionnés : `src/download_data.py` les télécharge dans `data/raw/`.

## Périmètre

- **Type de bien : appartements.** Le prix d'une maison dépend beaucoup de son terrain, ce qui brouille la relation entre surface et prix. Les appartements forment un marché plus homogène, idéal pour commencer par une régression simple.
- **Zone : les 6 départements de la région** : Cher (18), Eure-et-Loir (28), Indre (36), Indre-et-Loire (37), Loir-et-Cher (41) et Loiret (45). Même Tours, la plus grande ville, compte moins de 2 000 ventes d'appartements par an ; la région entière apporte plus de données et permet d'utiliser la localisation comme variable explicative.
- **Période : 2021 à 2025.** La hausse des taux d'intérêt en 2022-2023 a pu faire évoluer les prix : l'année de vente sera donc une variable du modèle.
- **Uniquement les ventes classiques** (`nature_mutation = "Vente"`) : j'écarte les ventes sur plan (VEFA), les adjudications et les échanges, dont les prix suivent d'autres logiques.
- **Variable à prédire : le prix de vente total en €.** Avec une régression simple `prix = a × surface + b`, le coefficient `a` se lit directement comme un prix au mètre carré, ce qui rend le modèle facile à interpréter.

## Démarche prévue

1. Télécharger les données et les explorer (types de mutation, types de biens, distributions des prix et des surfaces)
2. Nettoyer : une même vente peut apparaître sur plusieurs lignes (appartement + cave, parking…), il faut donc la reconstituer ; retirer les surfaces manquantes et les prix aberrants
3. Baseline : prix médian au m² de la commune (calculé sur les seules données d'entraînement) × surface, pour avoir un point de comparaison simple
4. Régression linéaire simple : prix en fonction de la surface
5. Régression linéaire multiple : ajout du nombre de pièces, du département ou de la commune, de la position GPS et de l'année de vente
6. Diagnostic : analyse des résidus, significativité des coefficients (p-values), multicolinéarité (VIF), puis régularisation Ridge et Lasso
7. Évaluation : entraînement sur 2021-2024 et test sur les ventes de 2025, jamais vues par le modèle, pour simuler une prédiction sur des ventes futures
8. Application Streamlit : saisir les caractéristiques d'un appartement et obtenir une estimation de son prix

## Mesure de l'erreur

- **Métrique principale : MAE (erreur absolue moyenne) en €.** Elle se lit directement : « en moyenne, l'estimation se trompe de X € ».
- **Métriques complémentaires** : RMSE, qui pénalise davantage les grosses erreurs, et R², la part de la variation des prix expliquée par le modèle.

## Notions et outils

- **Notions** : régression linéaire simple et multiple, moindres carrés, interprétation des coefficients, p-values, analyse des résidus, multicolinéarité (VIF), régularisation Ridge / Lasso, découpage temporel train/test, MAE / RMSE / R²
- **Outils** : Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, statsmodels, Jupyter, Streamlit, Git

## Structure du projet

```
prix-immo-cvl/
├── data/raw/          # fichiers DVF téléchargés (non versionnés)
├── data/processed/    # table nettoyée des ventes d'appartements (non versionnée)
├── notebooks/         # 01 exploration, 02 régression simple, 03 régression multiple
├── src/
│   ├── download_data.py   # téléchargement des fichiers DVF
│   ├── prepare_data.py    # nettoyage → une ligne par vente d'appartement
│   ├── modele.py          # construction des variables du modèle (partagée)
│   └── train.py           # évaluation sur 2025 puis entraînement du modèle final
├── models/            # modèle entraîné utilisé par l'application
├── app.py             # application Streamlit
├── requirements.txt       # dépendances de l'application (utilisées par Streamlit Cloud)
├── requirements-dev.txt   # environnement complet : notebooks, entraînement, application
└── README.md
```

## Installation

```bash
git clone https://github.com/Amen-ML/prix-immo-cvl.git
cd prix-immo-cvl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python src/download_data.py   # télécharge les 30 fichiers DVF
python src/prepare_data.py    # construit data/processed/apparts.parquet
python -m src.train           # évalue le modèle et enregistre models/modele_prix.joblib
streamlit run app.py          # lance l'application sur http://localhost:8501
```

## Résultats

Après nettoyage : **39 817 ventes d'appartements**. Apprentissage sur 2021-2024 (32 235 ventes), test sur 2025 (7 582 ventes).

| Modèle | MAE sur 2025 | R² sur 2025 |
|---|---|---|
| Baseline : prix médian | 48 203 € | −0,03 |
| Baseline : prix médian au m² de la commune × surface | 31 336 € | 0,58 |
| Régression linéaire simple : prix ~ surface | 38 985 € | 0,42 |
| A — Régression multiple sur le prix (surface, pièces, dépendance, année, département) | 34 967 € | 0,53 |
| B — Régression multiple sur log(prix) | 34 102 € | 0,49 |
| **C — log(prix) + une variable par commune** | **29 212 €** | **0,60** |
| C + Ridge (alpha choisi par validation croisée temporelle) | 29 211 € | 0,60 |
| C + Lasso (106 coefficients non nuls sur 117) | 29 237 € | 0,59 |

- **Régression simple** : la surface seule explique environ 40 % des variations de prix (chaque m² ajoute en moyenne 1 855 €),
  mais le modèle est biaisé géographiquement (prix surestimés de 44 % dans l'Indre, sous-estimés de 20 % en Indre-et-Loire).
- **Régression multiple** : surface et pièces sont corrélées (0,87, VIF ≈ 4) ; à surface égale, une pièce de plus fait baisser le prix.
  Le passage au log rend les effets multiplicatifs (+10 % de surface → +8,6 % de prix ; Indre-et-Loire ≈ +90 % par rapport au Cher)
  et supprime l'hétéroscédasticité.
- **Meilleur modèle (C)** : il bat la baseline par commune de 2 100 € de MAE et ramène le biais par département à moins de ±8 %.
- **Ridge / Lasso** : pas de gain, car avec 32 000 ventes pour 117 variables l'OLS ne sur-apprend pas.
- **Limites** : DVF ne contient ni l'état du bien, ni l'étage, ni le DPE ; un modèle linéaire ne capte pas les effets de quartier.

Détails : [01 — Exploration](notebooks/01_exploration.ipynb) · [02 — Régression simple](notebooks/02_regression_simple.ipynb) · [03 — Régression multiple](notebooks/03_regression_multiple.ipynb)

## Application

`streamlit run app.py` ouvre une application où l'on choisit la commune, la surface, le nombre de pièces et la présence
d'une dépendance. Elle affiche :
- le **prix estimé** et le prix au m², avec le modèle C réentraîné sur toutes les ventes 2021-2025 ;
- une **fourchette à 80 %**, tirée des erreurs réelles mesurées sur 2025 (8 prix réels sur 10 étaient entre 0,65 et 1,39 fois l'estimation) ;
- le prix médian observé dans la commune, ou un avertissement si la commune a moins de 10 ventes.

## Inspirations

- [agrawal-priyank/machine-learning-regression](https://github.com/agrawal-priyank/machine-learning-regression) : démarche de régression appliquée à des prix de maisons
- Documentation des données DVF géolocalisées sur data.gouv.fr
