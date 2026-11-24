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
├── notebooks/         # exploration et modélisation
├── src/               # scripts (téléchargement, préparation des données)
├── requirements.txt   # dépendances Python
└── README.md
```

## Installation

```bash
git clone https://github.com/Amen-ML/prix-immo-cvl.git
cd prix-immo-cvl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
```

## Résultats

À venir.

## Inspirations

- [agrawal-priyank/machine-learning-regression](https://github.com/agrawal-priyank/machine-learning-regression) : démarche de régression appliquée à des prix de maisons
- Documentation des données DVF géolocalisées sur data.gouv.fr
