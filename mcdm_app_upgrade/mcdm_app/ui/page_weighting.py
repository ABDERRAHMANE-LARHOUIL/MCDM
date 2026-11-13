"""Page 2 — Pondération des critères."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st

from core import weighting as W
from utils import presets as P
from utils import validation as V
from ui.common import (X_array, grouped_bars, heatmap, page_header, persistent_editor, plot, set_weights,
                       show_df, ss, weights_df)


def _parse(v) -> float:
    try:
        return float(Fraction(str(v).strip().replace(",", ".")))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _fmt_frac(x: float) -> str:
    if x >= 1:
        return f"{x:g}"
    f = Fraction(1 / x).limit_denominator(9)
    return f"1/{f.numerator}" if f.denominator == 1 else f"{x:.3g}"


def _apply_button(w, source: str, key: str) -> None:
    ss.weight_sets[source] = np.asarray(w, dtype=float)
    if st.button(f"Appliquer ces poids ({source})", key=key, type="primary", icon=":material/check:"):
        set_weights(w, source)
        st.toast(f"Poids actifs : {source}", icon=":material/check_circle:")
        st.rerun()


def _tab_manual() -> None:
    st.markdown("Saisissez des poids d'importance (ils sont renormalisés pour que Σ wⱼ = 1).")
    sig = "|".join(ss.crits)
    df = persistent_editor(f"wman|{sig}", weights_df(ss.weights).round(4), width="stretch")
    try:
        w, msg = V.normalize_weights(df.iloc[0].astype(float).values)
    except ValueError as exc:
        st.error(str(exc))
        return
    if msg:
        st.info(msg)
    show_df(weights_df(w))
    _apply_button(w, "Manuelle", "apply_manual")


def _tab_ahp() -> None:
    st.markdown("Matrice de comparaison par paires (échelle de Saaty 1–9). **Saisissez le triangle "
                "supérieur** (fractions acceptées : `1/3`) ; le triangle inférieur est complété par "
                "réciprocité aᵢⱼ = 1/aⱼᵢ.")
    n = len(ss.crits)
    if n < 2:
        st.warning("Il faut au moins 2 critères.")
        return
    sig = "|".join(ss.crits)
    if ss.crits == P.AHP_CAR["criteria"]:
        default = np.array(P.AHP_CAR["criteria_matrix"])
    else:
        default = np.ones((n, n))
    base = pd.DataFrame([[_fmt_frac(v) for v in row] for row in default], index=ss.crits, columns=ss.crits)
    df = persistent_editor(f"ahp|{sig}", base, width="stretch",
                           column_config={c: st.column_config.TextColumn(c) for c in ss.crits})
    upper = df.map(_parse).to_numpy()
    if np.isnan(np.triu(upper, 1)).any() or (np.triu(upper, 1) < 0).any():
        st.error("Valeurs invalides dans le triangle supérieur.")
        return
    A = W.ahp_reciprocal_from_upper(upper)
    for msg in V.check_ahp_matrix(A):
        st.warning(msg)
    method = st.radio("Calcul du vecteur propre", ["approx", "eigen"], horizontal=True,
                      format_func={"approx": "Méthode approximative (cours)",
                                   "eigen": "Vecteur propre exact"}.get)
    r = W.ahp(A, method)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matrice complétée et sommes des colonnes sⱼ**")
        show_df(pd.concat([pd.DataFrame(A, index=ss.crits, columns=ss.crits),
                           pd.DataFrame([r["col_sums"]], index=["Somme s"], columns=ss.crits)]), 3)
    with c2:
        st.markdown("**Matrice normalisée et priorités wᵢ (moyenne des lignes)**")
        nd = pd.DataFrame(r["normalized"], index=ss.crits, columns=ss.crits)
        nd["Priorité w"] = r["weights"]
        show_df(nd, 4, highlight_max="Priorité w")
    st.latex(r"\lambda_{max}=\sum_i s_i w_i,\quad CI=\frac{\lambda_{max}-n}{n-1},\quad CR=\frac{CI}{RI}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("λmax", f"{r['lambda_max']:.4f}")
    m2.metric("CI", f"{r['CI']:.4f}")
    m3.metric("RI", f"{r['RI']:.2f}")
    m4.metric("CR", f"{r['CR']:.4f}")
    if r["consistent"]:
        st.success("CR < 0,1 : la matrice est consistante, les poids peuvent être conservés.")
    else:
        st.error("CR ≥ 0,1 : jugements incohérents, révisez les comparaisons.")
    _apply_button(r["weights"], "AHP", "apply_ahp")


def _tab_bwm() -> None:
    n = len(ss.crits)
    if n < 2:
        st.warning("Il faut au moins 2 critères.")
        return
    is_supplier = ss.crits == P.BWM_SUPPLIER["criteria"]
    c1, c2, c3 = st.columns(3)
    best = c1.selectbox("Critère le plus important (Best)", range(n), format_func=lambda i: ss.crits[i],
                        index=0, key=f"bwm_best|{n}")
    worst = c2.selectbox("Critère le moins important (Worst)", range(n), format_func=lambda i: ss.crits[i],
                         index=n - 1, key=f"bwm_worst|{n}")
    model = c3.radio("Modèle", ["linear", "nonlinear"], key="bwm_model",
                     format_func={"linear": "Linéaire (cours)", "nonlinear": "Non linéaire (Rezaei 2015)"}.get)
    sig = "|".join(ss.crits)
    bo = P.BWM_SUPPLIER["a_BO"] if is_supplier else [1] * n
    ow = P.BWM_SUPPLIER["a_OW"] if is_supplier else [1] * n
    base = pd.DataFrame([bo, ow], index=["Best → autres (a_Bj)", "Autres → Worst (a_jW)"],
                        columns=ss.crits).astype(float)
    df = persistent_editor(f"bwm|{sig}", base, width="stretch")
    a_BO, a_OW = df.iloc[0].astype(float).values, df.iloc[1].astype(float).values
    for msg in W.bwm_sanity(best, worst, a_BO, a_OW):
        st.warning(msg)
    if best == worst:
        return
    try:
        r = W.bwm(best, worst, a_BO, a_OW) if model == "linear" else W.bwm_nonlinear(best, worst, a_BO, a_OW)
    except ValueError as exc:
        st.error(str(exc))
        return
    if model == "linear":
        st.latex(r"\min \xi \ \text{s.c.}\ |w_B - a_{Bj}w_j|\le\xi,\ |w_j - a_{jW}w_W|\le\xi,"
                 r"\ \sum_j w_j = 1,\ w_j\ge 0")
    else:
        st.latex(r"\min \xi \ \text{s.c.}\ \left|\frac{w_B}{w_j} - a_{Bj}\right|\le\xi,\ "
                 r"\left|\frac{w_j}{w_W} - a_{jW}\right|\le\xi,\ \sum_j w_j = 1")
    show_df(weights_df(r["weights"]))
    m1, m2, m3 = st.columns(3)
    m1.metric("ξ*", f"{r['xi']:.4f}")
    m2.metric("Comparaisons BWM (2n − 3)", r["n_comparisons"])
    m3.metric("Comparaisons AHP n(n − 1)/2", r["n_comparisons_ahp"])
    if model == "nonlinear":
        st.caption(f"Ratio de consistance CR = ξ*/CI = {r['xi']:.4f}/{r['CI']:.2f} = **{r['CR']:.3f}** "
                   "(plus il est proche de 0, plus les jugements sont cohérents).")
    else:
        st.caption("ξ* proche de 0 ⇒ jugements de l'expert cohérents.")
        with st.expander("Forme matricielle A·x ≤ b (x = [w₁ … wₙ, ξ])"):
            show_df(pd.DataFrame(r["A_ub"], index=r["constraint_labels"],
                                 columns=[f"w{j+1}" for j in range(n)] + ["ξ"]), 2)
    _apply_button(r["weights"], "BWM", "apply_bwm")


def _tab_swara() -> None:
    st.markdown("SWARA (*Step-wise Weight Assessment Ratio Analysis*) : l'expert **classe** les critères du plus au "
                "moins important, puis indique pour chacun son **importance comparative sⱼ** par rapport au "
                "critère précédent (0 = même importance, 0,5 = moitié moins important).")
    n = len(ss.crits)
    if n < 2:
        st.warning("Il faut au moins 2 critères.")
        return
    sig = "|".join(ss.crits)
    base = pd.DataFrame({"Rang d'importance": np.arange(1, n + 1), "sⱼ": [0.0] + [0.2] * (n - 1)}, index=ss.crits)
    df = persistent_editor(f"swara|{sig}", base, width="stretch", column_config={
        "Rang d'importance": st.column_config.NumberColumn(min_value=1, max_value=n, step=1, required=True,
                                                           help="1 = critère le plus important"),
        "sⱼ": st.column_config.NumberColumn(min_value=0.0, step=0.05, format="%.2f", required=True,
                                            help="Ignoré pour le critère de rang 1")})
    ranks = df["Rang d'importance"].astype(int).values
    if sorted(ranks) != list(range(1, n + 1)):
        st.error(f"Les rangs doivent être une permutation de 1 à {n} (chaque rang utilisé une seule fois).")
        return
    order = list(np.argsort(ranks))
    try:
        r = W.swara(order, df["sⱼ"].astype(float).values[order])
    except ValueError as exc:
        st.error(str(exc))
        return
    st.latex(r"k_j = 1 + s_j,\qquad q_j=\frac{q_{j-1}}{k_j}\ (q_1 = 1),\qquad w_j=\frac{q_j}{\sum_k q_k}")
    show_df(pd.DataFrame({"sⱼ": r["s"], "kⱼ = 1 + sⱼ": r["k"], "qⱼ": r["q"], "Poids wⱼ": r["weights_sorted"]},
                         index=[ss.crits[i] for i in order]))
    st.caption(f"SWARA demande n − 1 = {n - 1} jugements, contre {2 * n - 3} pour BWM et {n * (n - 1) // 2} pour AHP.")
    _apply_button(r["weights"], "SWARA", "apply_swara")


def _tab_entropy() -> None:
    st.latex(r"p_{ij}=\frac{d_{ij}}{\sum_i d_{ij}},\ E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij},\ "
             r"w_j=\frac{1-E_j}{n-\sum_k E_k}")
    try:
        r = W.entropy(X_array())
    except ValueError as exc:
        st.error(str(exc))
        return
    with st.expander("Matrice des proportions pᵢⱼ"):
        show_df(pd.DataFrame(r["P"], index=ss.alts, columns=ss.crits))
    show_df(pd.DataFrame({"Entropie E": r["E"], "Diversification 1 − E": r["diversification"],
                          "Poids w": r["weights"]}, index=ss.crits).T)
    _apply_button(r["weights"], "Entropie", "apply_entropy")


def _tab_critic() -> None:
    r = W.critic(X_array(), ss.types)
    st.latex(r"r_{ij}^{+}=\frac{x_{ij}-\min x_j}{\max x_j-\min x_j},\ "
             r"r_{ij}^{-}=\frac{\max x_j-x_{ij}}{\max x_j-\min x_j},\ "
             r"C_j=\sigma_j\sum_k(1-|\rho_{jk}|),\ w_j=\frac{C_j}{\sum C_j}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matrice normalisée rᵢⱼ**")
        show_df(pd.DataFrame(r["R"], index=ss.alts, columns=ss.crits))
    with c2:
        plot(heatmap(pd.DataFrame(r["rho"], index=ss.crits, columns=ss.crits),
                     "Corrélations de Pearson ρⱼₖ", diverging=True, zmin=-1, zmax=1))
    show_df(pd.DataFrame({"σⱼ": r["sigma"], "Σ(1 − |ρⱼₖ|)": r["conflict"], "Cⱼ": r["C"],
                          "Poids w": r["weights"]}, index=ss.crits).T)
    _apply_button(r["weights"], "CRITIC", "apply_critic")


def _tab_hybrid() -> None:
    sets = {k: v for k, v in ss.weight_sets.items() if len(v) == len(ss.crits)}
    subj = [k for k in sets if k in ("AHP", "BWM", "SWARA", "Manuelle")]
    obj = [k for k in sets if k in ("Entropie", "CRITIC")]
    if not subj or not obj:
        st.info("Calculez d'abord au moins une pondération subjective (AHP, BWM, SWARA ou manuelle) et une "
                "objective (Entropie ou CRITIC) dans les onglets précédents.")
        return
    c1, c2, c3 = st.columns(3)
    s = c1.selectbox("Poids subjectifs", subj)
    o = c2.selectbox("Poids objectifs", obj)
    mode = c3.radio("Combinaison", ["linear", "multiplicative"],
                    format_func={"linear": "Linéaire α", "multiplicative": "Multiplicative"}.get)
    alpha = st.slider("α (part subjective)", 0.0, 1.0, 0.5, 0.05, disabled=mode != "linear")
    if mode == "linear":
        st.latex(r"w_j = \alpha\, w_j^{subj} + (1-\alpha)\, w_j^{obj}")
    else:
        st.latex(r"w_j = \frac{w_j^{subj} w_j^{obj}}{\sum_k w_k^{subj} w_k^{obj}}")
    w = W.hybrid(sets[s], sets[o], alpha, mode)
    show_df(pd.DataFrame([sets[s], sets[o], w], index=[s, o, "Hybride"], columns=ss.crits))
    _apply_button(w, "Hybride", "apply_hybrid")


def render() -> None:
    page_header("Étape 2", "Pondération des critères",
                "Méthodes subjectives (jugements d'experts : AHP, BWM, SWARA), objectives (calculées sur la "
                "matrice de décision : entropie, CRITIC) ou hybrides. Appliquez un vecteur pour l'utiliser au classement.")
    st.caption(f"Poids actifs : {ss.weights_source}")
    show_df(weights_df(ss.weights))

    tabs = st.tabs(["Manuelle", "AHP", "BWM", "SWARA", "Entropie", "CRITIC", "Hybride", "Comparaison"])
    with tabs[0]:
        _tab_manual()
    with tabs[1]:
        _tab_ahp()
    with tabs[2]:
        _tab_bwm()
    with tabs[3]:
        _tab_swara()
    with tabs[4]:
        _tab_entropy()
    with tabs[5]:
        _tab_critic()
    with tabs[6]:
        _tab_hybrid()
    with tabs[7]:
        series = {k: v for k, v in ss.weight_sets.items() if len(v) == len(ss.crits)}
        series["Actifs"] = ss.weights
        plot(grouped_bars(ss.crits, series, "Comparaison des vecteurs de poids"))
