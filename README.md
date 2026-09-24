# Classification de raisins secs (Kecimen / Besni)

App Streamlit qui prédit la variété d'un raisin sec (Kecimen ou Besni) à partir de
7 mesures morphologiques, via un pipeline **normalisation → UMAP (2D) → KNN**.

## Fichiers du dépôt

| Fichier | Rôle |
|---|---|
| `app.py` | L'application Streamlit (à déployer) |
| `raisin_model.joblib` | Modèle déjà entraîné (UMAP + KNN), chargé par `app.py` |
| `requirements.txt` | Dépendances Python pour Streamlit Cloud |
| `train_model.py` | Script pour ré-entraîner et régénérer `raisin_model.joblib` |
| `Raisin_Dataset.xlsx` | Données source (utile pour `train_model.py`, pas requis par `app.py`) |

## Déployer sur Streamlit Community Cloud

1. Créer un dépôt GitHub et y pousser au minimum : `app.py`, `raisin_model.joblib`, `requirements.txt`.
2. Aller sur [share.streamlit.io](https://share.streamlit.io), se connecter avec GitHub.
3. "New app" → choisir le dépôt, la branche, et `app.py` comme fichier principal.
4. Déployer.

## Ré-entraîner le modèle (optionnel)

```bash
pip install -r requirements.txt
python train_model.py   # régénère raisin_model.joblib à partir de Raisin_Dataset.xlsx
```

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
