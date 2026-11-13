"""Page 3 — Évaluation crisp : plusieurs alternatives (classement / surclassement)
ou une seule alternative (évaluation absolue)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import crisp_ranking as CR
from core import outranking as O
from core import single_alternative as SA
from core import weighting as W
from utils import presets as P
from utils import validation as V
from ui.common import (PALETTE, X_array, bar_scores, grouped_bars, heatmap, line_chart, note, page_header,
                       persistent_editor, plot, series_colors, show_df, ss, style_fig, weights_df, winner, yes_no)

METHODS = ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR", "PROMETHEE II", "ELECTRE III"]


# ---------------------------------------------------------------------------
# Calcul générique (réutilisé par la page sensibilité)
# ---------------------------------------------------------------------------
def method_fn(name: str, params: dict | None = None):
    params = params or {}
    norm = params.get("norm", "linear")
    if name == "WSM":
        return lambda X, w, t: CR.wsm(X, w, t, norm)
    if name == "WPM":
        return lambda X, w, t: CR.wpm(X, w, t, norm)
    if name == "WASPAS":
        return lambda X, w, t: CR.waspas(X, w, t, params.get("lambda", 0.5), norm)
    if name == "TOPSIS":
        return CR.topsis
    if name == "VIKOR":
        return lambda X, w, t: CR.vikor(X, w, t, params.get("v", 0.5))
    if name == "PROMETHEE II":
        return lambda X, w, t: O.promethee(X, w, t, params.get("functions"))
    if name == "ELECTRE III":
        th = params.get("electre") or {}
        return lambda X, w, t: O.electre3(X, w, t, th.get("q"), th.get("p"), th.get("v"))
    raise KeyError(name)


def run_methods(names, X, w, types, params) -> dict:
    out = {}
    for m in names:
        try:
            out[m] = method_fn(m, params)(X, w, types)
        except ValueError as exc:
            out[m] = {"error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Plusieurs alternatives
# ---------------------------------------------------------------------------
def _detail(name: str, r: dict, params: dict) -> None:
    alts, crits = ss.alts, ss.crits
    if "error" in r:
        st.error(r["error"])
        return
    if name in ("WSM", "WPM", "WASPAS"):
        if params.get("norm") == "minmax":
            st.latex(r"r_{ij}=\frac{x_{ij}-\min_i x_{ij}}{\max_i x_{ij}-\min_i x_{ij}}\ (+),\qquad "
                     r"r_{ij}=\frac{\max_i x_{ij}-x_{ij}}{\max_i x_{ij}-\min_i x_{ij}}\ (-)")
        else:
            st.latex(r"r_{ij}=\frac{x_{ij}}{\max_i x_{ij}}\ (+),\qquad r_{ij}=\frac{\min_i x_{ij}}{x_{ij}}\ (-)")
        with st.expander("Matrice normalisée rᵢⱼ"):
            show_df(pd.DataFrame(r["R"], index=alts, columns=crits))
        if name == "WSM":
            st.latex(r"Q_i^{(1)}=\sum_j w_j r_{ij}")
            df = pd.DataFrame({"Q(1)": r["scores"], "Rang": r["ranks"]}, index=alts)
        elif name == "WPM":
            st.latex(r"Q_i^{(2)}=\prod_j r_{ij}^{\,w_j}")
            df = pd.DataFrame({"Q(2)": r["scores"], "Rang": r["ranks"]}, index=alts)
        else:
            st.latex(rf"Q_i=\lambda Q_i^{{(1)}}+(1-\lambda)Q_i^{{(2)}},\quad \lambda={r['lambda']:g}")
            df = pd.DataFrame({"Q(1) WSM": r["Q1"], "Q(2) WPM": r["Q2"], "Q WASPAS": r["scores"],
                               "Rang": r["ranks"]}, index=alts)
        show_df(df, highlight_max=df.columns[-2])
    elif name == "TOPSIS":
        st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}},\ v_{ij}=w_j r_{ij},\ "
                 r"S_i^{\pm}=\sqrt{\sum_j (v_{ij}-v_j^{\pm})^2},\ RC_i^*=\frac{S_i^-}{S_i^-+S_i^+}")
        with st.expander("Matrices normalisée et pondérée, solutions idéales"):
            show_df(pd.DataFrame(r["R"], index=alts, columns=crits))
            show_df(pd.concat([pd.DataFrame(r["V"], index=alts, columns=crits),
                               pd.DataFrame([r["I_plus"], r["I_minus"]], index=["I+", "I−"], columns=crits)]))
        show_df(pd.DataFrame({"S+": r["S_plus"], "S−": r["S_minus"], "RC*": r["scores"],
                              "Rang": r["ranks"]}, index=alts), highlight_max="RC*")
    elif name == "VIKOR":
        st.latex(r"S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-},\ R_i=\max_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-},\ "
                 r"Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}")
        with st.expander("Meilleures / pires valeurs x*ⱼ, x⁻ⱼ"):
            show_df(pd.DataFrame([r["x_best"], r["x_worst"]], index=["x*", "x−"], columns=crits))
        show_df(pd.DataFrame({"S": r["S"], "Rang S": r["rank_S"], "R": r["R"], "Rang R": r["rank_R"],
                              "Q": r["Q"], "Rang Q": r["ranks"]}, index=alts), highlight_min="Q")
        c1, c2 = st.columns(2)
        c1.markdown(f"{yes_no(r['C1_advantage'])} **C1 — avantage acceptable** : "
                    f"Q(A″) − Q(A′) ≥ DQ = 1/(m − 1) = {r['DQ']:.3f}")
        c2.markdown(f"{yes_no(r['C2_stability'])} **C2 — stabilité** : A′ est aussi la "
                    "meilleure selon S et R")
        st.success("Solution(s) de compromis : " + ", ".join(alts[k] for k in r["compromise"]))
    elif name == "PROMETHEE II":
        st.latex(r"\pi(a,b)=\sum_j w_j P_j(a,b),\ \phi^+(a)=\frac{1}{m-1}\sum_x\pi(a,x),\ "
                 r"\phi^-(a)=\frac{1}{m-1}\sum_x\pi(x,a),\ \phi=\phi^+-\phi^-")
        with st.expander("Matrice des indices de préférence agrégés π(a, b)"):
            show_df(pd.DataFrame(r["Pi"], index=alts, columns=alts))
        show_df(pd.DataFrame({"φ+": r["phi_plus"], "φ−": r["phi_minus"], "φ net": r["scores"],
                              "Rang": r["ranks"]}, index=alts), highlight_max="φ net")
        with st.expander("Préordre partiel PROMETHEE I"):
            rel = O.promethee_i_relations(r["phi_plus"], r["phi_minus"], alts)
            sym = {"P": "surclasse", "I": "indifférent à", "R": "incomparable à"}
            st.write("\n".join(f"- **{a}** {sym[k]} **{b}**" for a, k, b in rel))
    elif name == "ELECTRE III":
        _detail_electre(r)
        return
    if name == "PROMETHEE II":
        fig = go.Figure()
        order = np.argsort(-r["scores"])[::-1]
        y = [alts[i] for i in order]
        fig.add_bar(x=r["phi_plus"][order], y=y, orientation="h", name="φ+ (flux sortant)", marker=dict(color=PALETTE[0], cornerradius=4),
                    hovertemplate="%{y} · φ+ = %{x:.4f}<extra></extra>")
        fig.add_bar(x=-r["phi_minus"][order], y=y, orientation="h", name="−φ− (flux entrant)", marker=dict(color=PALETTE[1], cornerradius=4),
                    hovertemplate="%{y} · φ− = %{customdata:.4f}<extra></extra>", customdata=r["phi_minus"][order])
        fig.add_scatter(x=r["scores"][order], y=y, mode="markers", name="φ net",
                        marker=dict(color="#0f1b2d", size=10, symbol="diamond", line=dict(color="#fff", width=2)),
                        hovertemplate="%{y} · φ = %{x:.4f}<extra></extra>")
        fig.update_layout(barmode="relative")
        plot(style_fig(fig, max(260, 40 * len(alts) + 110), "Flux de surclassement (triés par flux net)"))
        return
    if name == "TOPSIS":
        c1, c2 = st.columns(2)
        with c1:
            plot(bar_scores(alts, r["scores"], "Coefficient de proximité RC* (plus élevé = meilleur)"))
        with c2:
            fig = go.Figure(go.Scatter(
                x=r["S_plus"], y=r["S_minus"], mode="markers+text" if len(alts) <= 12 else "markers", text=alts,
                textposition="top center", marker=dict(size=11, color=PALETTE[0], line=dict(color="#fff", width=2)),
                hovertemplate="%{text}<br>S+ = %{x:.4f} · S− = %{y:.4f}<extra></extra>"))
            fig.update_xaxes(title="S+ : distance à l'idéal (à minimiser)")
            fig.update_yaxes(title="S− : distance à l'anti-idéal (à maximiser)")
            plot(style_fig(fig, max(260, 40 * len(alts) + 90), "Position par rapport aux deux solutions idéales"))
        return
    if name == "WASPAS":
        plot(grouped_bars(alts, {"Q(1) WSM": r["Q1"], "Q(2) WPM": r["Q2"], "Q WASPAS": r["scores"]},
                          "WSM, WPM et WASPAS par alternative"))
        return
    higher = name != "VIKOR"
    vals = r["scores"] if higher else r["Q"]
    plot(bar_scores(alts, vals, f"{name} — {'score (plus élevé = meilleur)' if higher else 'Q (plus faible = meilleur)'}",
                    higher_better=higher))


def _classes(classes, alts) -> str:
    return "  →  ".join("{" + ", ".join(alts[i] for i in c) + "}" if len(c) > 1 else alts[c[0]] for c in classes)


def _detail_electre(r: dict) -> None:
    alts, crits = ss.alts, ss.crits
    st.latex(r"c_j(a,b)=\begin{cases}1 & g_j(b)-g_j(a)\le q_j\\[2pt] \dfrac{p_j-(g_j(b)-g_j(a))}{p_j-q_j} & q_j<\cdot\le p_j\\[4pt] 0 & \text{sinon}\end{cases}"
             r"\qquad C(a,b)=\sum_j w_j\,c_j(a,b)")
    st.latex(r"d_j(a,b)=\min\!\Big(1,\max\!\Big(0,\frac{g_j(b)-g_j(a)-p_j}{v_j-p_j}\Big)\Big),\qquad "
             r"\sigma(a,b)=C(a,b)\prod_{j:\,d_j>C}\frac{1-d_j(a,b)}{1-C(a,b)}")
    with st.expander("Seuils utilisés (q indifférence, p préférence, v veto)"):
        show_df(pd.DataFrame({"q": r["q"], "p": r["p"], "v": r["v"]}, index=crits).T)
    c1, c2 = st.columns(2)
    with c1:
        plot(heatmap(pd.DataFrame(r["C"], index=alts, columns=alts), "Concordance C(a, b)", zmin=0, zmax=1))
    with c2:
        plot(heatmap(pd.DataFrame(r["S"], index=alts, columns=alts), "Crédibilité σ(a, b) : « a surclasse b »", zmin=0, zmax=1))
    st.markdown("**Distillation descendante** (du meilleur au moins bon) : " + _classes(r["desc"], alts))
    st.markdown("**Distillation ascendante** (du meilleur au moins bon) : " + _classes(r["asc"], alts))
    show_df(pd.DataFrame({"Rang descendant": r["rank_desc"], "Rang ascendant": r["rank_asc"],
                          "Rang moyen": r["mean_rank"], "Rang final": r["ranks"]}, index=alts)
            .sort_values("Rang final", kind="stable"), 1)
    with st.expander("Préordre final (intersection des deux distillations)"):
        st.caption("P : la ligne est préférée à la colonne · P⁻ : la colonne est préférée · I : indifférence · "
                   "R : incomparabilité (les deux distillations se contredisent).")
        st.dataframe(pd.DataFrame(r["relations"], index=alts, columns=alts), width="stretch")
    n_inc = int((r["relations"] == "R").sum() // 2)
    if n_inc:
        note(f"{n_inc} paire(s) d'alternatives sont incomparables : ELECTRE III ne force pas un ordre total. "
             "Le « rang final » départage par le rang moyen des deux distillations.", warn=True)


def _multiple() -> None:
    c1, c2 = st.columns([3, 2])
    names = c1.multiselect("Méthodes", METHODS, default=METHODS)
    has_nonpos = bool((X_array() <= 0).any())
    with c2.popover("Paramètres", icon=":material/tune:", width="stretch"):
        lam = st.slider("λ WASPAS", 0.0, 1.0, 0.5, 0.05)
        v = st.slider("v VIKOR (poids de l'utilité de groupe)", 0.0, 1.0, 0.5, 0.05)
        norm = st.radio("Normalisation de WSM, WPM et WASPAS", ["linear", "minmax"],
                        index=1 if has_nonpos else 0, key=f"eval_norm|{has_nonpos}",
                        format_func={"linear": "Linéaire du cours : x/max, min/x",
                                     "minmax": "Min–max : accepte les valeurs négatives ou nulles"}.get)
    params = {"lambda": lam, "v": v, "norm": norm}
    if has_nonpos and norm == "minmax":
        st.caption("La matrice contient des valeurs négatives ou nulles : WSM, WPM et WASPAS utilisent la normalisation "
                   "min–max au lieu de x/max et min/x (modifiable dans Paramètres). Avec min–max, la pire valeur d'un "
                   "critère vaut 0, ce qui annule le score WPM de l'alternative concernée.")
    if "PROMETHEE II" in names:
        with st.expander("Fonctions de préférence PROMETHEE (par critère)"):
            sig = "|".join(ss.crits)
            X = X_array()
            spread = (X.max(axis=0) - X.min(axis=0))
            base = pd.DataFrame({"Fonction": ["usual"] * len(ss.crits), "q": 0.0,
                                 "p": np.round(spread / 2, 4), "s": np.round(spread / 4, 4)}, index=ss.crits)
            df = persistent_editor(f"prom|{sig}", base, width="stretch", column_config={
                "Fonction": st.column_config.SelectboxColumn(options=list(O.PREFERENCE_FUNCTIONS), required=True)})
            st.caption(" · ".join(f"`{k}` : {v}" for k, v in O.PREFERENCE_FUNCTIONS.items()))
            params["functions"] = [{"kind": row["Fonction"], "q": float(row["q"]), "p": float(row["p"]),
                                    "s": float(row["s"])} for _, row in df.iterrows()]
    if "ELECTRE III" in names:
        with st.expander("Seuils ELECTRE III (par critère)"):
            Xe = X_array()
            spread_e = Xe.max(axis=0) - Xe.min(axis=0)
            base_e = pd.DataFrame({"q (indifférence)": np.round(0.05 * spread_e, 4),
                                   "p (préférence)": np.round(0.25 * spread_e, 4),
                                   "v (veto)": [None] * len(ss.crits)}, index=ss.crits).astype(float)
            de = persistent_editor(f"electre|{'|'.join(ss.crits)}", base_e, width="stretch")
            st.caption("Seuils exprimés dans l'unité du critère, avec q ≤ p ≤ v. Laissez v vide pour un critère sans veto. "
                       "Par défaut : q = 5 % et p = 25 % de l'étendue du critère.")
            params["electre"] = {"q": de.iloc[:, 0].fillna(0).astype(float).values,
                                 "p": de.iloc[:, 1].fillna(0).astype(float).values,
                                 "v": de.iloc[:, 2].astype(float).values}
    if not names:
        return
    results = run_methods(names, X_array(), ss.weights, ss.types, params)
    ss.results = {k: v for k, v in results.items() if "error" not in v}
    ss.results_params = params

    st.subheader("Synthèse des classements")
    rank_tbl = pd.DataFrame({k: v["ranks"] for k, v in ss.results.items()}, index=ss.alts)
    if not rank_tbl.empty:
        rank_tbl["Rang moyen"] = rank_tbl.mean(axis=1)
        show_df(rank_tbl.sort_values("Rang moyen"), 2)
        export = rank_tbl.copy()
        for k, v in ss.results.items():
            export[f"Score {k}"] = v["Q"] if k == "VIKOR" else v.get("mean_rank", v["scores"]) if k == "ELECTRE III" else v["scores"]
        st.download_button("Exporter les classements et les scores (CSV)", icon=":material/download:",
                           data=export.to_csv().encode("utf-8-sig"), file_name="classements_mcdm.csv", mime="text/csv")
        winners = {k: ss.alts[int(np.argmin(v["ranks"]))] for k, v in ss.results.items()}
        top = rank_tbl["Rang moyen"].idxmin()
        agree = sum(a == top for a in winners.values())
        winner(str(top), f"classée 1re par {agree} méthode(s) sur {len(winners)} · rang moyen {rank_tbl['Rang moyen'].min():.2f}",
               kicker="Meilleur rang moyen")
        if len(ss.results) >= 2:
            plot(line_chart(list(ss.results), {a: [int(ss.results[k]["ranks"][i]) for k in ss.results]
                                               for i, a in enumerate(ss.alts)},
                            "Rang de chaque alternative selon la méthode", ytitle="Rang (1 = meilleure)",
                            reverse_y=True, markers=True, fmt=".0f", height=max(320, 26 * len(ss.alts) + 150),
                            colors=series_colors(ss.alts, focus=list(rank_tbl.sort_values("Rang moyen").index[:6]))))

    tabs = st.tabs(names)
    for t, name in zip(tabs, names):
        with t:
            _detail(name, results[name], params)


# ---------------------------------------------------------------------------
# Une seule alternative
# ---------------------------------------------------------------------------
def _maut() -> None:
    st.latex(r"S(A)=\sum_j w_j u_j(x_j),\quad u_j^{+}=\frac{x-x_j^{min}}{x_j^{max}-x_j^{min}},\quad "
             r"u_j^{-}=\frac{x-x_j^{max}}{x_j^{min}-x_j^{max}}")
    src = st.radio("Source", ["preset", "space"], horizontal=True,
                   format_func={"preset": "Exemple (scoring de crédit — illustratif)",
                                "space": "Une alternative de l'espace de décision"}.get)
    if src == "space":
        a = st.selectbox("Alternative évaluée", ss.alts)
        X = X_array()
        base = pd.DataFrame({"Type": ss.types, "Valeur x": X[ss.alts.index(a)], "x_min": X.min(axis=0),
                             "x_max": X.max(axis=0), "Poids": ss.weights}, index=ss.crits)
        key = f"maut_space|{'|'.join(ss.crits)}|{a}"
    else:
        p = P.MAUT_CREDIT
        base = pd.DataFrame({"Type": p["types"], "Valeur x": p["x"], "x_min": p["x_min"], "x_max": p["x_max"],
                             "Poids": p["weights"]}, index=p["criteria"])
        key = "maut_preset"
    df = persistent_editor(key, base, width="stretch", column_config={
        "Type": st.column_config.SelectboxColumn(options=["+", "-"], required=True)})
    w, msg = V.normalize_weights(df["Poids"].astype(float).values)
    if msg:
        st.info(msg)
    r = SA.maut(df["Valeur x"].astype(float).values, w, df["x_min"].astype(float).values,
                df["x_max"].astype(float).values, list(df["Type"]))
    c1, c2 = st.columns([1, 2])
    tau = c1.slider("Seuil de décision τ", 0.0, 1.0, float(P.MAUT_CREDIT["tau"]), 0.01)
    decision = SA.decide(r["score"], tau)
    c1.metric("Score S(A)", f"{r['score']:.4f}")
    (c1.success if decision == "Accepter" else c1.error)(f"Décision : **{decision}** (S {'≥' if decision == 'Accepter' else '<'} τ = {tau:.2f})")
    cls = SA.classify(r["score"], {"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.0})
    c1.markdown(f"Classe (A ≥ 0,8 · B ≥ 0,6 · C ≥ 0,4 · D) : **{cls}**")
    with c2:
        tbl = pd.DataFrame({"Utilité u": r["utilities"], "Poids w": w, "Contribution w·u": r["contributions"]},
                           index=df.index)
        show_df(tbl)
        fig = go.Figure(go.Bar(x=r["contributions"], y=list(df.index), orientation="h",
                               marker_color=PALETTE[0], marker=dict(cornerradius=4),
                               hovertemplate="%{y} : %{x:.4f}<extra></extra>"))
        fig.add_vline(x=0, line_color="#898781")
        plot(style_fig(fig, 260, "Contributions au score"))


def _roc() -> None:
    st.markdown("Seuil issu des données : on s'appuie sur des alternatives passées dont la décision est connue.")
    p = P.ROC_EXAMPLE
    base = pd.DataFrame({"Score S": p["scores"],
                         "Décision réelle": ["Accepter" if y else "Rejeter" for y in p["labels"]]},
                        index=p["alternatives"])
    df = persistent_editor("roc", base, num_rows="dynamic", width="stretch", column_config={
        "Décision réelle": st.column_config.SelectboxColumn(options=["Accepter", "Rejeter"], required=True)})
    df = df.dropna()
    if len(df) < 2 or df["Décision réelle"].nunique() < 2:
        st.warning("Il faut au moins une alternative acceptée et une rejetée.")
        return
    scores = df["Score S"].astype(float).values
    labels = (df["Décision réelle"] == "Accepter").astype(int).values
    c1, c2, c3 = st.columns(3)
    txt = c1.text_input("Seuils à comparer (cas du cours)", ", ".join(f"{t:g}" for t in p["thresholds"]))
    c_fp = c2.number_input("Coût d'un faux positif", 0.0, value=1.0)
    c_fn = c3.number_input("Coût d'un faux négatif", 0.0, value=1.0)
    st.latex(r"TPR=\frac{TP}{TP+FN},\ FPR=\frac{FP}{FP+TN},\ d=\sqrt{FPR^2+(1-TPR)^2},\ J=TPR-FPR")
    try:
        user_t = [float(t) for t in txt.replace(";", ",").split(",") if t.strip()]
    except ValueError:
        user_t = []
    cols = ["tau", "TP", "FP", "FN", "TN", "TPR", "FPR", "d", "J", "accuracy", "cost"]
    names = {"tau": "Seuil τ", "accuracy": "Exactitude", "cost": "Coût"}
    if user_t:
        rows = SA.roc_table(scores, labels, user_t, c_fp, c_fn)
        st.markdown("**Seuils testés**")
        show_df(pd.DataFrame(rows)[cols].rename(columns=names), 2)
    all_rows = SA.roc_table(scores, labels, None, c_fp, c_fn)
    best = SA.best_thresholds(all_rows)
    m = st.columns(4)
    m[0].metric("Max Youden J", f"τ = {best['youden']['tau']:.2f}", f"J = {best['youden']['J']:.2f}", delta_color="off")
    m[1].metric("Min distance d", f"τ = {best['distance']['tau']:.2f}", f"d = {best['distance']['d']:.2f}", delta_color="off")
    m[2].metric("Max exactitude", f"τ = {best['accuracy']['tau']:.2f}", f"{best['accuracy']['accuracy']:.0%}", delta_color="off")
    m[3].metric("Coût minimal", f"τ = {best['cost']['tau']:.2f}", f"C = {best['cost']['cost']:.2f}", delta_color="off")
    ties = [r["tau"] for r in all_rows if abs(r["J"] - best["youden"]["J"]) < 1e-9]
    if len(ties) > 1:
        st.info("Plusieurs seuils à égalité sur J (" + ", ".join(f"{t:.2f}" for t in ties) +
                ") : départager selon le coût d'un faux positif ou d'un faux négatif.")
    with st.expander("Tous les seuils candidats"):
        show_df(pd.DataFrame(all_rows)[cols].rename(columns=names), 2)
    pts = sorted({(r["FPR"], r["TPR"]) for r in all_rows} | {(0.0, 0.0), (1.0, 1.0)})
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#898781", dash="dot", width=1),
                    name="Hasard", hoverinfo="skip")
    fig.add_scatter(x=[q[0] for q in pts], y=[q[1] for q in pts], mode="lines+markers", name="ROC",
                    line=dict(color=PALETTE[0], width=2), marker=dict(size=8),
                    hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>")
    b = best["youden"]
    fig.add_scatter(x=[b["FPR"]], y=[b["TPR"]], mode="markers+text", name="Optimum Youden",
                    marker=dict(size=13, color=PALETTE[1], line=dict(color="white", width=2)),
                    text=[f"τ = {b['tau']:.2f}"], textposition="bottom right")
    fig.update_xaxes(title="FPR (faux positifs)", range=[-0.02, 1.02])
    fig.update_yaxes(title="TPR (vrais positifs)", range=[-0.02, 1.05], scaleanchor="x")
    plot(style_fig(fig, 420, f"Courbe ROC — AUC = {SA.auc(all_rows):.3f}"))


def _distance() -> None:
    st.latex(r"D^+=\sqrt{\sum_j (x_j-x_j^+)^2},\ D^-=\sqrt{\sum_j (x_j-x_j^-)^2},\ C=\frac{D^-}{D^++D^-}")
    a = st.selectbox("Alternative évaluée", ss.alts, key="dist_alt")
    X = X_array()
    b = np.array([t == "+" for t in ss.types])
    base = pd.DataFrame({"Valeur x": X[ss.alts.index(a)],
                         "Idéal x+": np.where(b, X.max(axis=0), X.min(axis=0)),
                         "Anti-idéal x−": np.where(b, X.min(axis=0), X.max(axis=0)),
                         "Poids": ss.weights}, index=ss.crits)
    df = persistent_editor(f"dist|{'|'.join(ss.crits)}|{a}", base, width="stretch")
    norm = st.toggle("Normaliser (utilités 0–1) et pondérer", value=True,
                     help="Sans normalisation (formule brute du cours), les critères de grande amplitude dominent.")
    r = SA.distance_to_ideal(df["Valeur x"].values, df["Idéal x+"].values, df["Anti-idéal x−"].values,
                             df["Poids"].values, normalized=norm)
    c = st.columns(3)
    c[0].metric("D+", f"{r['D_plus']:.4f}")
    c[1].metric("D−", f"{r['D_minus']:.4f}")
    c[2].metric("Indice C", f"{r['C']:.4f}")


def _fuzzy_single() -> None:
    p = P.FUZZY_SINGLE
    st.markdown("Chaque évaluation est un nombre flou triangulaire x̃ = (a, b, c) : a pessimiste, "
                "b le plus probable, c optimiste (exemple p. 134).")
    base = pd.DataFrame({"Poids": p["weights"], "a": [t[0] for t in p["tfns"]], "b": [t[1] for t in p["tfns"]],
                         "c": [t[2] for t in p["tfns"]], "c_max": [t[2] for t in p["tfns"]]}, index=p["criteria"])
    df = persistent_editor("fsingle", base, num_rows="dynamic", width="stretch")
    df = df.dropna()
    T = df[["a", "b", "c"]].astype(float).values
    if not ((T[:, 0] <= T[:, 1]) & (T[:, 1] <= T[:, 2])).all():
        st.error("Chaque nombre flou doit vérifier a ≤ b ≤ c.")
        return
    r = SA.fuzzy_single(T, df["Poids"].astype(float).values, df["c_max"].astype(float).values)
    st.latex(r"\tilde r_j=\frac{(a,b,c)}{c_{max}},\quad \tilde S=\sum_j w_j\tilde r_j,\quad S=\frac{a+b+c}{3}")
    show_df(pd.DataFrame(r["R"], index=df.index, columns=["a", "b", "c"]).assign(Poids=df["Poids"].values), 3)
    c = st.columns(3)
    c[0].metric("S̃ (somme pondérée)", "(" + ", ".join(f"{v:.2f}" for v in r["S_fuzzy"]) + ")")
    c[1].metric("Score défuzzifié S", f"{r['S']:.4f}")
    c[2].metric("Largeur c − a (incertitude)", f"{r['width']:.4f}")


def _parse_frac(v) -> float:
    from fractions import Fraction
    try:
        return float(Fraction(str(v).strip().replace(",", ".")))
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _fmt_frac(x: float) -> str:
    from fractions import Fraction
    f = Fraction(float(x)).limit_denominator(9)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _pairwise(key: str, names, default) -> np.ndarray | None:
    """Éditeur de matrice de comparaison par paires (triangle supérieur, fractions acceptées)."""
    base = pd.DataFrame([[_fmt_frac(v) for v in row] for row in np.asarray(default, dtype=float)],
                        index=names, columns=names)
    df = persistent_editor(key, base, width="stretch",
                           column_config={c: st.column_config.TextColumn(c) for c in names})
    upper = df.map(_parse_frac).to_numpy()
    tri = upper[np.triu_indices(len(names), 1)]
    if np.isnan(tri).any() or (tri <= 0).any():
        st.error("Valeurs invalides dans le triangle supérieur (attendu : nombres positifs ou fractions comme 1/3).")
        return None
    return W.ahp_reciprocal_from_upper(upper)


def _ahp_ranking() -> None:
    st.markdown("AHP complet : une matrice de comparaison des **critères**, puis une matrice de comparaison des "
                "**alternatives pour chaque critère**. Aucune matrice de décision chiffrée n'est nécessaire. "
                "Saisissez le triangle supérieur (échelle de Saaty 1 à 9, fractions acceptées).")
    p = P.AHP_CAR
    c1, c2, c3 = st.columns([3, 3, 2])
    crit_txt = c1.text_input("Critères (séparés par des virgules)", ", ".join(p["criteria"]), key="ahpr_crits")
    alt_txt = c2.text_input("Alternatives (séparées par des virgules)", ", ".join(p["alternatives"]), key="ahpr_alts")
    method = c3.radio("Vecteur propre", ["approx", "eigen"], horizontal=True, key="ahpr_method",
                      format_func={"approx": "Approché (cours)", "eigen": "Exact"}.get)
    crits = [c.strip() for c in crit_txt.split(",") if c.strip()]
    alts = [a.strip() for a in alt_txt.split(",") if a.strip()]
    if len(crits) < 2 or len(alts) < 2 or len(set(crits)) != len(crits) or len(set(alts)) != len(alts):
        st.warning("Il faut au moins 2 critères et 2 alternatives, aux noms distincts.")
        return
    is_course = crits == p["criteria"] and alts == p["alternatives"]
    sig = "|".join(crits) + "#" + "|".join(alts)
    st.markdown("**Comparaison des critères**")
    A = _pairwise(f"ahpr_c|{sig}", crits, p["criteria_matrix"] if is_course else np.ones((len(crits),) * 2))
    mats = []
    tabs = st.tabs([f"Alternatives selon « {c} »" for c in crits])
    for j, (tab, c) in enumerate(zip(tabs, crits)):
        with tab:
            mats.append(_pairwise(f"ahpr_a{j}|{sig}", alts,
                                  p["alternative_matrices"][j] if is_course else np.ones((len(alts),) * 2)))
    if A is None or any(M is None for M in mats):
        return
    r = W.ahp_rank(A, mats, method)
    c = r["criteria"]
    best = alts[int(np.argmax(r["scores"]))]
    winner(best, f"priorité finale {r['scores'].max():.4f}")
    cons = pd.DataFrame({"λmax": [c["lambda_max"]] + [x["lambda_max"] for x in r["local"]],
                         "CI": [c["CI"]] + [x["CI"] for x in r["local"]],
                         "CR": [c["CR"]] + [x["CR"] for x in r["local"]],
                         "Consistante (CR < 0,1)": ["oui" if x["consistent"] else "non" for x in [c] + r["local"]]},
                        index=["Critères"] + [f"Alternatives / {k}" for k in crits])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test de consistance de chaque matrice**")
        show_df(cons)
        if not all(x["consistent"] for x in [c] + r["local"]):
            note("Au moins une matrice a un CR ≥ 0,1 : révisez les comparaisons correspondantes.", warn=True)
    with c2:
        st.markdown("**Priorités locales, poids des critères et priorités finales**")
        show_df(pd.DataFrame([c["weights"]], index=["Poids des critères wⱼ"], columns=crits))
        L = pd.DataFrame(r["local_matrix"], index=alts, columns=crits)
        L["Priorité finale"] = r["scores"]
        show_df(L, highlight_max="Priorité finale")
    st.latex(r"\text{Priorité finale}(A_i)=\sum_j w_j\,w_{ij}")
    c1, c2 = st.columns(2)
    with c1:
        plot(bar_scores(alts, r["scores"], "Priorités finales"))
    with c2:
        j = st.selectbox("Analyse de sensibilité : poids du critère", range(len(crits)), format_func=lambda i: crits[i],
                         key="ahpr_sens")
        grid = np.linspace(0, 1, 41)
        w0 = c["weights"]
        rest = np.delete(w0, j)
        rest = rest / rest.sum() if rest.sum() > 0 else np.full(len(rest), 1 / len(rest))
        S = np.array([r["local_matrix"] @ np.insert(rest * (1 - g), j, g) for g in grid])
        plot(line_chart(grid, {a: S[:, i] for i, a in enumerate(alts)}, f"Priorités finales selon le poids de « {crits[j]} »",
                        xtitle="Poids du critère", ytitle="Priorité finale", vline=float(w0[j]), height=300))


def render() -> None:
    page_header("Étape 3", "Évaluation",
                "Classer plusieurs alternatives (méthodes additives ou de surclassement), les comparer par paires "
                "avec AHP, ou évaluer une alternative isolée par rapport à un seuil.")
    errors, _ = V.check_decision_matrix(ss.X, ss.types)
    scope = st.radio("Portée de l'évaluation", ["multi", "ahp", "single"], horizontal=True,
                     format_func={"multi": "Plusieurs alternatives — classement et surclassement",
                                  "ahp": "AHP complet — comparaisons par paires",
                                  "single": "Une seule alternative — évaluation absolue"}.get)
    if scope == "multi":
        st.caption(f"Poids actifs : {ss.weights_source}")
        show_df(weights_df(ss.weights))
        if errors:
            for e in errors:
                st.error(e)
            return
        _multiple()
    elif scope == "ahp":
        _ahp_ranking()
    else:
        tabs = st.tabs(["MAUT + seuil", "Seuil par ROC / Youden", "Distance à l'idéal", "Évaluation floue (TFN)"])
        with tabs[0]:
            _maut()
        with tabs[1]:
            _roc()
        with tabs[2]:
            _distance()
        with tabs[3]:
            _fuzzy_single()
