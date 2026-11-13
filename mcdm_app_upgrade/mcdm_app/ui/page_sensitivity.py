"""Page 5 — Analyse de sensibilité, robustesse et consensus entre méthodes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import sensitivity as SE
from utils import presets as P
from utils import validation as V
from ui.common import X_array, color_for, heatmap, page_header, plot, show_df, ss, style_fig
from ui.page_evaluation import METHODS, method_fn, run_methods


def _one_at_a_time(params) -> None:
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Méthode", METHODS, index=3, key="sens_method")
    j = c2.selectbox("Critère dont le poids varie", range(len(ss.crits)), format_func=lambda i: ss.crits[i])
    view = c3.radio("Affichage", ["scores", "ranks"], horizontal=True,
                    format_func={"scores": "Scores", "ranks": "Rangs"}.get)
    st.caption("w′ⱼ parcourt [0, 1] ; les autres poids sont redistribués proportionnellement : "
               "w′ₖ = wₖ·(1 − w′ⱼ)/(1 − wⱼ).")
    grid = np.round(np.linspace(0, 1, 41), 3)
    try:
        r = SE.one_at_a_time(method_fn(method, params), X_array(), ss.weights, ss.types, j, grid)
    except ValueError as exc:
        st.error(str(exc))
        return
    Y = r["scores"] if view == "scores" else r["ranks"]
    if method == "VIKOR" and view == "scores":
        Y = 1 - Y  # retour à Q (plus faible = meilleur)
    fig = go.Figure()
    for i, a in enumerate(ss.alts):
        fig.add_scatter(x=grid, y=Y[:, i], mode="lines", name=a, line=dict(color=color_for(i), width=2,
                        shape="hv" if view == "ranks" else "linear"),
                        hovertemplate=f"{a}<br>w = %{{x:.3f}} : %{{y:.4f}}<extra></extra>")
    fig.add_vline(x=float(ss.weights[j]), line_dash="dot", line_color="#898781",
                  annotation_text=f"poids actuel {ss.weights[j]:.3f}")
    fig.update_xaxes(title=f"Poids de « {ss.crits[j]} »")
    ylab = {"scores": "Q (↓ meilleur)" if method == "VIKOR" else "Score (↑ meilleur)", "ranks": "Rang"}[view]
    fig.update_yaxes(title=ylab, autorange="reversed" if view == "ranks" else True)
    plot(style_fig(fig, 420, f"{method} — sensibilité au poids de « {ss.crits[j]} »"))
    flips = SE.rank_reversal_points(grid, r["ranks"])
    if flips:
        st.warning("Changement de l'alternative classée 1re pour w ≈ " + ", ".join(f"{f:.3f}" for f in flips))
    else:
        st.success("L'alternative classée 1re est la même sur tout l'intervalle [0, 1].")


def _monte_carlo(params) -> None:
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Méthode", METHODS, index=3, key="mc_method")
    n = c2.select_slider("Simulations", [200, 500, 1000, 2000, 5000], value=1000)
    kappa = c3.slider("Concentration κ (↑ = perturbation faible)", 5, 500, 50, 5)
    st.caption("Poids tirés selon une loi de Dirichlet(κ·w) centrée sur les poids actifs.")
    try:
        r = SE.monte_carlo(method_fn(method, params), X_array(), ss.weights, ss.types, n, kappa)
    except ValueError as exc:
        st.error(str(exc))
        return
    m = len(ss.alts)
    freq = pd.DataFrame(r["rank_freq"] * 100, index=ss.alts, columns=[f"Rang {k+1}" for k in range(m)])
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(heatmap(freq, "Fréquence des rangs (%)", fmt=".1f", zmin=0, zmax=100))
    with c2:
        fig = go.Figure(go.Histogram(x=r["rho"], nbinsx=30, marker_color=color_for(0),
                                     hovertemplate="ρ ∈ %{x} : %{y}<extra></extra>"))
        fig.update_xaxes(title="ρ de Spearman vs classement de base")
        plot(style_fig(fig, 340, "Stabilité globale du classement"))
    best = ss.alts[int(np.argmin(r["base_ranks"]))]
    st.metric(f"Probabilité que {best} reste 1re", f"{r['rank_freq'][ss.alts.index(best), 0]:.1%}")


def _consensus(params) -> None:
    src = st.radio("Classements comparés", ["live", "course"], horizontal=True,
                   format_func={"live": "Méthodes appliquées au problème courant",
                                "course": "Exemple du cours (p. 78–79)"}.get)
    if src == "course":
        alts = P.SPEARMAN_EXAMPLE["alternatives"]
        rankings = {k: np.array(v) for k, v in P.SPEARMAN_EXAMPLE["rankings"].items()}
    else:
        names = st.multiselect("Méthodes", METHODS, default=METHODS, key="cons_methods")
        res = run_methods(names, X_array(), ss.weights, ss.types, params)
        rankings = {k: v["ranks"] for k, v in res.items() if "error" not in v}
        for k, v in res.items():
            if "error" in v:
                st.warning(f"{k} : {v['error']}")
        for k, v in ss.get("fuzzy_results", {}).items():
            if len(v["ranks"]) == len(ss.alts) and st.checkbox(f"Inclure {k}", key=f"inc_{k}"):
                rankings[k] = v["ranks"]
        alts = ss.alts
    if len(rankings) < 2:
        st.info("Sélectionnez au moins deux méthodes.")
        return
    st.latex(r"\rho = 1-\frac{6\sum_i d_i^2}{n(n^2-1)}")
    tbl = pd.DataFrame(rankings, index=alts)
    tbl["Consensus (Borda)"] = SE.borda_consensus(rankings)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Rangs (1 = meilleure)**")
        show_df(tbl.sort_values("Consensus (Borda)"), 0)
    rho = SE.spearman_matrix(rankings)
    with c2:
        plot(heatmap(rho, "Matrice de corrélation de Spearman", diverging=True, zmin=-1, zmax=1))
    n = len(alts)
    crit = SE.SPEARMAN_CRITICAL_5.get(n)
    if crit:
        pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(rho.index) for b in rho.columns[i + 1:]]
        sig = [f"{a} – {b} ({v:.2f})" for a, b, v in pairs if v >= crit]
        st.markdown(f"Avec n = {n}, seule une corrélation ρ ≥ **{crit:.3f}** est significative à 5 %. "
                    + ("Paires significatives : " + ", ".join(sig) if sig else "Aucune paire significative."))
    firsts = {k: alts[int(np.argmin(v))] for k, v in rankings.items()}
    if len(set(firsts.values())) == 1:
        st.success(f"Toutes les méthodes désignent **{next(iter(firsts.values()))}** : choix robuste.")
    else:
        st.warning("Les méthodes divergent sur la meilleure alternative : "
                   + ", ".join(f"{k} → {a}" for k, a in firsts.items()))


def render() -> None:
    page_header("Étape 4", "Sensibilité et consensus",
                "Bonnes pratiques du cours : appliquer plusieurs méthodes, comparer les classements, "
                "faire varier les poids et vérifier la stabilité du résultat.")
    errors, _ = V.check_decision_matrix(ss.X, ss.types)
    params = dict(ss.get("results_params", {"lambda": 0.5, "v": 0.5}))
    params.setdefault("norm", "minmax" if (X_array() <= 0).any() else "linear")
    st.caption("Les paramètres des méthodes (λ, v, fonctions de préférence PROMETHEE, seuils ELECTRE III) sont ceux "
               "réglés dans la page Évaluation ; à défaut, les valeurs par défaut s'appliquent.")
    t1, t2, t3 = st.tabs(["Perturbation d'un poids", "Monte Carlo sur les poids", "Consensus (Spearman)"])
    with t1:
        if errors:
            st.error(errors[0])
        else:
            _one_at_a_time(params)
    with t2:
        if errors:
            st.error(errors[0])
        else:
            _monte_carlo(params)
    with t3:
        _consensus(params)
