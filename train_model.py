"""
Entraîne le pipeline de classification des raisins (normalisation -> UMAP -> KNN)
et sauvegarde tout ce dont l'app Streamlit a besoin dans raisin_model.joblib.

À lancer une seule fois (en local ou dans un notebook), AVANT de déployer sur Streamlit :
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP

DATA_PATH = "Raisin_Dataset.xlsx"
MODEL_PATH = "raisin_model.joblib"

# 1. Charger les données
df = pd.read_excel(DATA_PATH)
feature_cols = df.columns[:-1].tolist()  # les 7 variables morphologiques
x = df[feature_cols].values
y = df["Class"].values

# 2. Normaliser (comme dans le notebook)
z = normalize(x)

# 3. Entraîner UMAP (2 composantes) sur toutes les données
umap_model = UMAP(n_components=2, random_state=0)
embedding = umap_model.fit_transform(z)

# 4. Entraîner le classifieur KNN sur l'espace réduit
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(embedding, y)

# 5. Sauvegarder tout ce qu'il faut pour l'app (modèles + données pour le graphique + valeurs par défaut)
bundle = {
    "umap_model": umap_model,
    "knn_model": knn_model,
    "embedding": embedding,       # projection UMAP des 900 raisins (pour le scatter plot)
    "labels": y,                  # classes des 900 raisins
    "feature_cols": feature_cols, # ordre des 7 variables
    "feature_defaults": {c: float(df[c].mean()) for c in feature_cols},
    "feature_ranges": {c: (float(df[c].min()), float(df[c].max())) for c in feature_cols},
}
joblib.dump(bundle, MODEL_PATH)
print(f"Modèle entraîné et sauvegardé dans {MODEL_PATH}")
print(f"Accuracy (train, KNN sur UMAP) : {knn_model.score(embedding, y)*100:.2f}%")
