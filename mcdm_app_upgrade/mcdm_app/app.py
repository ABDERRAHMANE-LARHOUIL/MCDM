"""Plateforme d'aide à la décision multicritère (MCDM).

Cours « Aide à la décision » — Prof. Lamrani Alaoui, EMI Rabat.
Lancement :  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import (page_ai, page_cases, page_evaluation, page_fuzzy, page_fuzzy_logic,  # noqa: E402
                page_sensitivity, page_space, page_weighting)
from ui.common import (init_state, inject_css, load_preset, note, page_header, show_df, ss,  # noqa: E402
                       stats, weights_df)
from utils import presets as P  # noqa: E402

st.set_page_config(page_title="Plateforme MCDM — EMI", page_icon=":material/account_tree:", layout="wide")
inject_css()
init_state()

STEPS = [
    ("01", "Espace de décision", "Alternatives, critères, sens de préférence et matrice de décision. "
     "Exercices du cours ou import CSV.", "space"),
    ("02", "Pondération", "Poids subjectifs (AHP, BWM, SWARA), objectifs (entropie, CRITIC), hybrides ou manuels.", "weighting"),
    ("03", "Évaluation", "Classement (WSM, WPM, WASPAS, TOPSIS, VIKOR, AHP), surclassement (PROMETHEE, ELECTRE III), "
     "évaluation d'une seule alternative.", "evaluation"),
    ("04", "Robustesse", "Perturbation d'un poids, Monte-Carlo, corrélation de Spearman et consensus entre méthodes.", "sensitivity"),
]

CATALOG = [
    ("Pondération subjective", "AHP (test de consistance), BWM linéaire et non linéaire, SWARA", "Pondération"),
    ("Pondération objective", "Entropie de Shannon, CRITIC, poids hybrides", "Pondération"),
    ("Classement additif", "WSM, WPM, WASPAS, TOPSIS, VIKOR (solution de compromis), AHP complet", "Évaluation"),
    ("Surclassement", "PROMETHEE I et II (6 fonctions de préférence), ELECTRE III (veto, distillations)", "Évaluation"),
    ("Une seule alternative", "MAUT et seuil, seuil par ROC (distance, Youden, coûts), distance à l'idéal, évaluation floue", "Évaluation"),
    ("Robustesse", "Sensibilité à un poids, Monte-Carlo (Dirichlet), Spearman, consensus de Borda", "Sensibilité et consensus"),
    ("Logique floue", "Fonctions d'appartenance, coupe-α, arithmétique floue, nombres intuitionnistes, "
     "pythagoriciens, q-ROF, picturaux, hésitants", "Logique floue"),
    ("MCDM flou", "TOPSIS flou (Chen, 2000), TOPSIS flou intuitionniste (IFWA)", "MCDM flou"),
    ("MCDM et IA", "MCDM → classification supervisée, pondération par valeurs de Shapley, MCDM dynamique", "MCDM et IA"),
]


def home() -> None:
    page_header("Cours « Aide à la décision » · Prof. Lamrani Alaoui · EMI Rabat",
                "Plateforme d'aide à la décision multicritère",
                "Formaliser un problème de décision, pondérer les critères, classer les alternatives et vérifier "
                "la robustesse du résultat, avec le détail de chaque calcul.")
    stats([("Problème courant", ss.get("problem_label") or "—"), ("Alternatives", str(len(ss.alts))),
           ("Critères", str(len(ss.crits))), ("Poids actifs", ss.weights_source)])
    for col, (i, title, text, key) in zip(st.columns(4), STEPS):
        with col:
            st.markdown(f'<div class="step"><div class="i">{i}</div><div class="t">{title}</div>'
                        f'<div class="x">{text}</div></div>', unsafe_allow_html=True)
            st.page_link(PAGES[key], label="Ouvrir", icon=":material/arrow_forward:")
    st.subheader("Méthodes implémentées")
    st.dataframe(pd.DataFrame(CATALOG, columns=["Famille", "Méthodes", "Page"]), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Taxonomie (MADM)")
        st.markdown("""
- **Précision des évaluations** : *crisp* (valeurs exactes) ou *fuzzy* (jugements imprécis, linguistiques)
- **Nombre de décideurs** : *single* ou *group* (agrégation des avis)
- **Nombre d'alternatives** : une seule (évaluation absolue, seuil) ou plusieurs (classement)
- **Étape** : pondération des critères, puis classement des alternatives
""")
    with c2:
        st.subheader("Quelle méthode choisir ?")
        st.markdown("""
- Valeurs exactes → méthodes **crisp** ; jugements linguistiques → méthodes **floues**
- Compensation acceptée → **WSM, TOPSIS** ; refusée (sécurité, environnement) → **PROMETHEE, ELECTRE**
- Effort du décideur : **AHP** (n(n−1)/2 comparaisons) > **BWM** (2n−3) > **CRITIC, entropie** (aucune)
""")
    note("Aucune méthode n'est la meilleure dans l'absolu : appliquez-en plusieurs, comparez les classements "
         "et menez une analyse de sensibilité avant de conclure.")


Pg = st.Page
PAGES = {
    "home": Pg(home, title="Accueil", icon=":material/home:", url_path="accueil", default=True),
    "space": Pg(page_space.render, title="Espace de décision", icon=":material/table_chart:", url_path="espace"),
    "weighting": Pg(page_weighting.render, title="Pondération des critères", icon=":material/balance:", url_path="ponderation"),
    "evaluation": Pg(page_evaluation.render, title="Évaluation", icon=":material/leaderboard:", url_path="evaluation"),
    "sensitivity": Pg(page_sensitivity.render, title="Sensibilité et consensus", icon=":material/compare_arrows:", url_path="sensibilite"),
    "fuzzy_logic": Pg(page_fuzzy_logic.render, title="Logique floue", icon=":material/blur_on:", url_path="logique-floue"),
    "fuzzy": Pg(page_fuzzy.render, title="MCDM flou", icon=":material/groups:", url_path="mcdm-flou"),
    "ai": Pg(page_ai.render, title="MCDM et IA", icon=":material/model_training:", url_path="mcdm-ia"),
    "cases": Pg(page_cases.render, title="Études de cas du cours", icon=":material/menu_book:", url_path="etudes-de-cas"),
}

nav = st.navigation({
    "": [PAGES["home"]],
    "MCDM classique": [PAGES["space"], PAGES["weighting"], PAGES["evaluation"], PAGES["sensitivity"]],
    "Flou": [PAGES["fuzzy_logic"], PAGES["fuzzy"]],
    "Aller plus loin": [PAGES["ai"], PAGES["cases"]],
})

with st.sidebar:
    st.markdown(f'<div class="side-box"><b>Problème courant</b><br>{ss.get("problem_label") or "—"}<br>'
                f'{len(ss.alts)} alternatives × {len(ss.crits)} critères<br><b>Poids</b> : {ss.weights_source}</div>',
                unsafe_allow_html=True)
    with st.expander("Poids actifs"):
        show_df(weights_df(ss.weights).T, 3)
    with st.expander("Charger un exercice"):
        choice = st.selectbox("Exercice", list(P.DECISION_PRESETS), key="sb_preset", label_visibility="collapsed")
        if st.button("Charger", width="stretch", key="sb_load"):
            load_preset(choice)
            st.rerun()

nav.run()
