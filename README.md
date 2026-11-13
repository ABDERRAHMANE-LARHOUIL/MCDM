---
title: Plateforme MCDM
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# Plateforme d'aide à la décision multicritère (MCDM)

Application Streamlit qui implémente les méthodes du cours « Aide à la décision »
(Prof. Lamrani Alaoui, EMI Rabat), avec le détail de chaque calcul.

| Famille | Méthodes |
|---|---|
| Pondération subjective | AHP (test de consistance), BWM linéaire et non linéaire, SWARA |
| Pondération objective | Entropie, CRITIC, poids hybrides |
| Classement additif | WSM, WPM, WASPAS, TOPSIS, VIKOR, AHP complet |
| Surclassement | PROMETHEE I / II, ELECTRE III |
| Une seule alternative | MAUT et seuil, seuil par ROC (distance, Youden, coûts), distance à l'idéal, évaluation floue |
| Robustesse | Sensibilité à un poids, Monte-Carlo, Spearman, consensus de Borda |
| Logique floue | Fonctions d'appartenance, coupe-α, arithmétique floue, nombres intuitionnistes, pythagoriciens, q-ROF, picturaux, hésitants |
| MCDM flou | TOPSIS flou (Chen, 2000), TOPSIS flou intuitionniste |
| MCDM et IA | MCDM → classification, pondération par valeurs de Shapley, MCDM dynamique |

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests        # vérifie les exemples chiffrés du cours
```

## Déployer

**Streamlit Community Cloud** — New app → dépôt `mcdmpart2_TP`, branche `main`,
fichier principal `mcdm_app/app.py`. Le thème est lu dans `.streamlit/config.toml`
à la racine du dépôt (une copie se trouve aussi dans `mcdm_app/`).

**Hugging Face Spaces** — créer un Space de type *Docker*, puis pousser le contenu
du dossier `mcdm_app/` (ce `README.md`, le `Dockerfile` et le code) à la racine du Space.

## Structure

```
app.py            navigation et page d'accueil
core/             calculs purs (numpy / scipy), sans dépendance à Streamlit
ui/               une page par fichier, composants partagés dans common.py
utils/            exemples du cours (presets) et validation des entrées
tests/            exemples chiffrés du cours
```
