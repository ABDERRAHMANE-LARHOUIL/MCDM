"""Page 4 — MCDM floue : TOPSIS flou (Chen 2000) et TOPSIS flou intuitionniste."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import fuzzy_mcdm as F
from utils import presets as P
from utils import validation as V
from ui.common import (page_header, PALETTE, bar_scores, persistent_editor, plot, reset_editor, show_df, ss,
                       style_fig, tfn_frame)


def _problem_setup(prefix: str, preset: dict, default_rating: str, default_weight: str) -> dict:
    """Structure (alternatives, critères, décideurs) propre à chaque méthode floue."""
    key = f"{prefix}_problem"
    if key not in ss:
        ss[key] = {k: list(preset[k]) for k in ("alternatives", "criteria", "types", "decision_makers")}
        ss[key]["ratings"] = preset["ratings"]
        ss[key]["weights"] = preset["weights"]
        ss[key]["dm_weights"] = preset.get("dm_weights")
        ss[key]["version"] = 0
    prob = ss[key]
    c1, c2, c3 = st.columns([2, 2, 1])
    if c1.button("Charger l'exemple" + (" (illustratif)" if preset.get("illustrative") else " du cours"),
                 key=f"{prefix}_load", width="stretch"):
        ss[key] = {k: list(preset[k]) for k in ("alternatives", "criteria", "types", "decision_makers")}
        ss[key].update(ratings=preset["ratings"], weights=preset["weights"],
                       dm_weights=preset.get("dm_weights"), version=prob["version"] + 1)
        st.rerun()
    old_k = len(prob["decision_makers"])
    k = int(c3.number_input("Décideurs K", 1, 10, old_k, key=f"{prefix}_K_{prob['version']}"))
    if c2.button("Reprendre l'espace de décision", key=f"{prefix}_space", width="stretch"):
        dms = prob["decision_makers"]
        ss[key] = {"alternatives": list(ss.alts), "criteria": list(ss.crits), "types": list(ss.types),
                   "decision_makers": dms,
                   "ratings": [[[default_rating] * len(ss.crits) for _ in ss.alts] for _ in dms],
                   "weights": [[default_weight] * len(ss.crits) for _ in dms],
                   "dm_weights": [1 / len(dms)] * len(dms), "version": prob["version"] + 1}
        st.rerun()
    if k != old_k:
        v = prob["version"]
        n_a, n_c = len(prob["alternatives"]), len(prob["criteria"])
        cur_r = [ss.get(f"_val_{prefix}_r{i}_{v}", pd.DataFrame(prob["ratings"][i])).values.tolist()
                 for i in range(old_k)]
        cur_w = ss.get(f"_val_{prefix}_w_{v}", pd.DataFrame(prob["weights"])).values.tolist()
        prob.update(
            decision_makers=[f"DM{i+1}" for i in range(k)],
            ratings=[cur_r[i] if i < old_k else [[default_rating] * n_c for _ in range(n_a)] for i in range(k)],
            weights=[cur_w[i] if i < old_k else [default_weight] * n_c for i in range(k)],
            dm_weights=[1 / k] * k, version=v + 1)
        st.rerun()
    st.caption(f"{len(prob['alternatives'])} alternatives · {len(prob['criteria'])} critères "
               f"({', '.join(f'{c} ({t})' for c, t in zip(prob['criteria'], prob['types']))}) · "
               f"{len(prob['decision_makers'])} décideurs")
    return prob


def _linguistic_editors(prefix: str, prob: dict, rating_opts, weight_opts):
    v = prob["version"]
    ratings, weights = [], []
    dm_tabs = st.tabs(prob["decision_makers"])
    cfg_r = {c: st.column_config.SelectboxColumn(c, options=rating_opts, required=True) for c in prob["criteria"]}
    for k, t in enumerate(dm_tabs):
        with t:
            base = pd.DataFrame(prob["ratings"][k], index=prob["alternatives"], columns=prob["criteria"])
            ratings.append(persistent_editor(f"{prefix}_r{k}_{v}", base, width="stretch",
                                             column_config=cfg_r).values.tolist())
    st.markdown("**Importance des critères selon chaque décideur**")
    base = pd.DataFrame(prob["weights"], index=prob["decision_makers"], columns=prob["criteria"])
    cfg_w = {c: st.column_config.SelectboxColumn(c, options=weight_opts, required=True) for c in prob["criteria"]}
    weights = persistent_editor(f"{prefix}_w_{v}", base, width="stretch", column_config=cfg_w).values.tolist()
    return ratings, weights


def _scale_editor(name: str, scale: dict, cols: list[str]) -> dict:
    df = persistent_editor(name, pd.DataFrame(list(scale.values()), index=list(scale), columns=cols),
                           width="stretch")
    return {lab: tuple(float(x) for x in row) for lab, row in zip(df.index, df.values)}


def _chen() -> None:
    st.markdown("Approche de **Chen (2000)** : les décideurs évaluent alternatives et importance des critères "
                "par des variables linguistiques converties en nombres flous triangulaires.")
    prob = _problem_setup("chen", P.CHEN_RECRUITMENT, "F", "M")
    with st.expander("Échelles linguistiques (tableau 1 — modifiables)"):
        c1, c2 = st.columns(2)
        with c1:
            rs = _scale_editor("chen_rscale", F.CHEN_RATING_SCALE, ["a", "b", "c"])
        with c2:
            ws = _scale_editor("chen_wscale", F.CHEN_WEIGHT_SCALE, ["a", "b", "c"])
    for msg in V.check_tfn(np.array(list(rs.values()) + list(ws.values()))):
        st.error(msg)
        return
    ratings, weights = _linguistic_editors("chen", prob, list(rs), list(ws))
    try:
        Xk = np.array([F.linguistic_to_tfn(m, rs) for m in ratings])
        Wk = np.array([F.linguistic_to_tfn(m, ws) for m in weights])
    except KeyError as exc:
        st.error(f"Libellé inconnu : {exc}")
        return
    X = F.aggregate_tfn_mean(Xk)
    Wt = F.aggregate_tfn_mean(Wk)
    r = F.fuzzy_topsis_chen(X, Wt, prob["types"])
    alts, crits = prob["alternatives"], prob["criteria"]

    st.subheader("Résolution")
    s1, s2, s3, s4 = st.tabs(["1) Évaluations collectives", "2) Matrice normalisée",
                              "3) Poids collectifs & matrice pondérée", "4) Distances & CC"])
    with s1:
        st.latex(r"\tilde x_{ij}=\frac{1}{K}\left(\tilde x_{ij}^1\oplus\cdots\oplus\tilde x_{ij}^K\right)")
        st.dataframe(tfn_frame(X, alts, crits, 2), width="stretch")
    with s2:
        st.latex(r"\tilde r_{ij}^{+}=\left(\frac{a_{ij}}{c_j^*},\frac{b_{ij}}{c_j^*},\frac{c_{ij}}{c_j^*}\right),\ "
                 r"\tilde r_{ij}^{-}=\left(\frac{a_j^-}{c_{ij}},\frac{a_j^-}{b_{ij}},\frac{a_j^-}{a_{ij}}\right)")
        st.dataframe(tfn_frame(r["R"], alts, crits, 3), width="stretch")
    with s3:
        st.latex(r"\tilde w_j=\frac{1}{K}\left(\tilde w_j^1\oplus\cdots\oplus\tilde w_j^K\right),\quad "
                 r"\tilde v_{ij}=\tilde r_{ij}\otimes\tilde w_j")
        st.dataframe(tfn_frame(Wt[None, :, :], ["w̃ⱼ"], crits, 3), width="stretch")
        st.dataframe(tfn_frame(r["V"], alts, crits, 3), width="stretch")
    with s4:
        st.latex(r"d(\tilde a,\tilde b)=\sqrt{\tfrac13\sum_k (a_k-b_k)^2},\ A^*=(1,1,1),\ A^-=(0,0,0),\ "
                 r"CC_i=\frac{D_i^-}{D_i^-+D_i^*}")
        with st.expander("Distances élémentaires d(ṽᵢⱼ, v*) et d(ṽᵢⱼ, v⁻)"):
            show_df(pd.DataFrame(r["d_star"], index=alts, columns=crits))
            show_df(pd.DataFrame(r["d_minus"], index=alts, columns=crits))
        show_df(pd.DataFrame({"D*": r["D_star"], "D−": r["D_minus"], "CC": r["scores"], "Rang": r["ranks"]},
                             index=alts), highlight_max="CC")
    c1, c2 = st.columns(2)
    with c1:
        plot(bar_scores(alts, r["scores"], "Coefficient de proximité CCᵢ"))
    with c2:
        fig = go.Figure(go.Scatter(x=r["D_minus"], y=r["D_star"], mode="markers+text", text=alts,
                                   textposition="top center", marker=dict(size=12, color=PALETTE[0]),
                                   hovertemplate="%{text}<br>D− %{x:.3f} · D* %{y:.3f}<extra></extra>"))
        fig.update_xaxes(title="D− (éloignement de l'anti-idéal, ↑)")
        fig.update_yaxes(title="D* (distance à l'idéal, ↓)")
        plot(style_fig(fig, 340, "Positionnement idéal / anti-idéal"))
    _maybe_share("TOPSIS flou (Chen)", alts, r)


def _if_topsis() -> None:
    st.markdown("**TOPSIS flou intuitionniste** : chaque jugement est un nombre α = (μ, ν) avec μ + ν ≤ 1 et "
                "π = 1 − μ − ν le degré d'hésitation. Les avis sont agrégés par l'opérateur **IFWA**.")
    prob = _problem_setup("ift", P.IF_SUPPLIER, "F", "M")
    if P.IF_SUPPLIER.get("illustrative"):
        st.caption("Le cours ne fournit pas de données chiffrées pour cette méthode : l'exemple est illustratif ; "
                   "les échelles linguistiques sont celles de Boran et al. (2009).")
    with st.expander("Échelles linguistiques intuitionnistes (modifiables)"):
        c1, c2 = st.columns(2)
        with c1:
            rs = _scale_editor("ift_rscale", F.IFN_RATING_SCALE, ["μ", "ν"])
        with c2:
            ws = _scale_editor("ift_wscale", F.IFN_WEIGHT_SCALE, ["μ", "ν"])
    for msg in V.check_ifn([v[0] for v in list(rs.values()) + list(ws.values())],
                           [v[1] for v in list(rs.values()) + list(ws.values())]):
        st.error(msg)
        return
    dms = prob["decision_makers"]
    st.markdown("**Poids des décideurs σ**")
    dw = prob.get("dm_weights") or [1 / len(dms)] * len(dms)
    dwdf = persistent_editor(f"ift_dmw_{prob['version']}", pd.DataFrame([dw], index=["σ"], columns=dms),
                             width="stretch")
    sigma = dwdf.iloc[0].astype(float).values
    if sigma.sum() <= 0:
        st.error("La somme des poids des décideurs doit être positive.")
        return
    ratings, weights = _linguistic_editors("ift", prob, list(rs), list(ws))
    try:
        mu = np.array([[[rs[l][0] for l in row] for row in m] for m in ratings])
        nu = np.array([[[rs[l][1] for l in row] for row in m] for m in ratings])
        wm = np.array([[ws[l][0] for l in row] for row in weights])
        wn = np.array([[ws[l][1] for l in row] for row in weights])
    except KeyError as exc:
        st.error(f"Libellé inconnu : {exc}")
        return
    r = F.if_topsis(mu, nu, wm, wn, sigma, prob["types"])
    alts, crits = prob["alternatives"], prob["criteria"]

    def ifn_frame(M, N, index, columns):
        M, N = np.atleast_2d(M), np.atleast_2d(N)
        return pd.DataFrame([[f"({M[i, j]:.3f}, {N[i, j]:.3f})" for j in range(M.shape[1])]
                             for i in range(M.shape[0])], index=index, columns=columns)

    st.subheader("Résolution")
    s1, s2, s3, s4 = st.tabs(["3) Agrégation IFWA", "4–5) Poids & matrice pondérée", "6) Solutions idéales",
                              "7–8) Séparations & Cᵢ"])
    with s1:
        st.latex(r"\alpha_{ij}=IFWA_\sigma=\left(1-\prod_k(1-\mu_{ij}^k)^{\sigma_k},\ \prod_k(\nu_{ij}^k)^{\sigma_k}\right)")
        st.dataframe(ifn_frame(r["A_mu"], r["A_nu"], alts, crits), width="stretch")
    with s2:
        st.latex(r"a'_{ij}=a_{ij}\otimes w_j=(\mu_a\mu_w,\ \nu_a+\nu_w-\nu_a\nu_w)")
        st.dataframe(ifn_frame(r["W_mu"], r["W_nu"], ["wⱼ"], crits), width="stretch")
        st.dataframe(ifn_frame(r["P_mu"], r["P_nu"], alts, crits), width="stretch")
    with s3:
        st.dataframe(pd.concat([ifn_frame(*r["I_plus"], ["I+"], crits), ifn_frame(*r["I_minus"], ["I−"], crits)]),
                     width="stretch")
    with s4:
        st.latex(r"S_i^{\pm}=\sqrt{\frac{1}{2n}\sum_j\left[(\mu_{ij}-\mu_j^{\pm})^2+(\nu_{ij}-\nu_j^{\pm})^2+"
                 r"(\pi_{ij}-\pi_j^{\pm})^2\right]},\quad C_i=\frac{S_i^-}{S_i^-+S_i^+}")
        show_df(pd.DataFrame({"S+": r["S_plus"], "S−": r["S_minus"], "Cᵢ": r["scores"], "Rang": r["ranks"]},
                             index=alts), highlight_max="Cᵢ")
    plot(bar_scores(alts, r["scores"], "Coefficient de proximité Cᵢ"))
    _maybe_share("TOPSIS flou intuitionniste", alts, r)


def _maybe_share(name: str, alts, r) -> None:
    if list(alts) == list(ss.alts):
        ss.setdefault("fuzzy_results", {})[name] = {"scores": r["scores"], "ranks": r["ranks"]}
        st.caption("Les alternatives coïncident avec l'espace de décision : ce classement est disponible "
                   "dans la page « Sensibilité & consensus ».")


def render() -> None:
    page_header("Données imprécises", "MCDM flou",
                "Plusieurs décideurs évaluent les alternatives et l'importance des critères avec des variables "
                "linguistiques, converties en nombres flous triangulaires (Chen, 2000) ou intuitionnistes.")
    t1, t2 = st.tabs(["TOPSIS flou (Chen 2000)", "TOPSIS flou intuitionniste"])
    with t1:
        _chen()
    with t2:
        _if_topsis()
