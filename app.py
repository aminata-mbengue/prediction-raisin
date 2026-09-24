"""
App Streamlit — Classification de raisins secs (Kecimen / Besni)
Pipeline : normalisation -> UMAP (2D) -> KNN

Fichiers nécessaires dans le même dossier :
- app.py
- raisin_model.joblib   (généré par train_model.py)
- requirements.txt
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import normalize

MODEL_PATH = "raisin_model.joblib"

st.set_page_config(page_title="Classification de raisins secs", page_icon="🍇")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
umap_model = bundle["umap_model"]
knn_model = bundle["knn_model"]
embedding = bundle["embedding"]
labels = bundle["labels"]
feature_cols = bundle["feature_cols"]
feature_defaults = bundle["feature_defaults"]
feature_ranges = bundle["feature_ranges"]

st.title("🍇 Classification de raisins secs (Kecimen / Besni)")
st.write(
    "Ce modèle réduit les 7 variables morphologiques à 2 dimensions avec **UMAP**, "
    "puis classe le raisin avec un **KNN** (k=5)."
)

st.subheader("Mesures du raisin")
col1, col2 = st.columns(2)
values = {}
for i, feat in enumerate(feature_cols):
    lo, hi = feature_ranges[feat]
    default = feature_defaults[feat]
    target_col = col1 if i % 2 == 0 else col2
    values[feat] = target_col.number_input(
        feat, min_value=float(lo * 0.5), max_value=float(hi * 1.5),
        value=float(default), format="%.4f",
    )

if st.button("Prédire la classe", type="primary"):
    sample = np.array([[values[c] for c in feature_cols]])
    sample_z = normalize(sample)
    sample_emb = umap_model.transform(sample_z)

    pred = knn_model.predict(sample_emb)[0]
    proba = knn_model.predict_proba(sample_emb)[0]
    proba_dict = {c: float(p) for c, p in zip(knn_model.classes_, proba)}

    st.success(f"Classe prédite : **{pred}**")
    st.write(proba_dict)

    fig, ax = plt.subplots(figsize=(6, 5))
    for classe in np.unique(labels):
        mask = labels == classe
        ax.scatter(embedding[mask, 0], embedding[mask, 1], alpha=0.4, label=classe, s=15)
    ax.scatter(sample_emb[0, 0], sample_emb[0, 1], color="black", marker="X", s=200, label="Nouveau raisin")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Prédiction : {pred}")
    ax.legend()
    st.pyplot(fig)
