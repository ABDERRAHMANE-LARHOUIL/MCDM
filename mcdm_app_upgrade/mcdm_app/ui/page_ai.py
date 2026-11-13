"""Page — MCDM et IA : classification supervisée à partir d'un score MCDM, pondération par
valeurs de Shapley (XAI) et MCDM dynamique par prévision des critères."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ai as AI
from ui.common import (PALETTE, X_array, bar_scores, grouped_bars, heatmap, line_chart, load_decision_problem, note,
                       page_header, persistent_editor, plot, set_weights, show_df, ss, stats, style_fig, winner)
from ui.page_evaluation import METHODS, method_fn

SCORERS = ["WSM", "WPM", "WASPAS", "TOPSIS"]

try:  # scikit-learn n'est requis que pour cette page
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

CLASSIFIERS = {
    "Forêt aléatoire": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "Régression logistique": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "k plus proches voisins": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Arbre de décision": lambda: DecisionTreeClassifier(max_depth=5, random_state=0),
} if HAS_SKLEARN else {}
REGRESSORS = {
    "Forêt aléatoire": lambda: RandomForestRegressor(n_estimators=120, max_depth=8, random_state=0),
    "Gradient boosting": lambda: GradientBoostingRegressor(random_state=0),
    "Régression linéaire (ridge)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1e-3)),
} if HAS_SKLEARN else {}


# ---------------------------------------------------------------------------
# Base d'apprentissage commune aux scénarios 1 et 2
# ---------------------------------------------------------------------------
def _dataset():
    """Renvoie (X_all, scores, n_real, méthode) ou None en cas d'erreur."""
    X = X_array()
    m = len(X)
    c1, c2, c3 = st.columns(3)
    scorer = c1.selectbox("Méthode MCDM additive qui produit le score Sᵢ", SCORERS, index=3, key="ai_scorer")
    n_syn = c2.slider("Alternatives synthétiques ajoutées", 0, 600, 0 if m >= 60 else 240, 20, key="ai_nsyn",
                      help="Un modèle de ML a besoin de plus d'exemples que les quelques alternatives d'un exercice. "
                           "Les alternatives synthétiques sont des mélanges bruités des alternatives réelles.")
    seed = int(c3.number_input("Graine aléatoire", 0, 9999, 0, key="ai_seed"))
    X_all = np.vstack([X, AI.augment(X, n_syn, seed=seed)])
    try:
        norm = "minmax" if (X_all <= 0).any() else "linear"
        scores = np.asarray(method_fn(scorer, {"norm": norm})(X_all, ss.weights, ss.types)["scores"], dtype=float)
    except ValueError as exc:
        st.error(f"{scorer} : {exc}")
        return None
    if not np.isfinite(scores).all():
        st.error(f"{scorer} produit des scores non définis sur ces données ; choisissez une autre méthode.")
        return None
    if len(X_all) < 12:
        note("Moins de 12 alternatives : ajoutez des alternatives synthétiques pour entraîner un modèle.", warn=True)
        return None
    return X_all, scores, m, scorer


# ---------------------------------------------------------------------------
# Scénario 1 — de MCDM à la classification supervisée
# ---------------------------------------------------------------------------
def _scenario1(data) -> None:
    X_all, scores, m, scorer = data
    st.markdown("**Principe.** ① Pondérer les critères par une méthode MCDM. ② Calculer le score Sᵢ de chaque "
                "alternative. ③ Étiqueter les alternatives selon Sᵢ. ④ Entraîner un modèle sur (critères → classe). "
                "⑤ Prédire la classe d'une **nouvelle** alternative sans refaire le classement.")
    c1, c2 = st.columns(2)
    k = c1.select_slider("Nombre de classes", [2, 3, 4], value=4, key="ai_k")
    model_name = c2.selectbox("Modèle de classification", list(CLASSIFIERS), key="ai_clf")
    labels = tuple("ABCD"[:k])
    y = AI.label_by_quantiles(scores, labels)
    counts = pd.Series(y).value_counts().reindex(labels).fillna(0).astype(int)
    folds = int(min(5, counts.min()))
    model = CLASSIFIERS[model_name]()
    if folds >= 2:
        pred = cross_val_predict(model, X_all, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=list(labels))
    else:
        acc, cm = None, None
        note("Trop peu d'exemples par classe pour une validation croisée.", warn=True)
    model.fit(X_all, y)
    stats([("Base d'apprentissage", str(len(X_all)), f"{m} réelles + {len(X_all) - m} synthétiques"),
           ("Score MCDM", scorer), ("Classes", " / ".join(labels), "quantiles de Sᵢ, A = meilleure"),
           ("Exactitude (validation croisée)", f"{acc:.1%}" if acc is not None else "—", f"{folds} plis" if folds >= 2 else "")])
    c1, c2 = st.columns(2)
    with c1:
        tbl = pd.DataFrame({"Score Sᵢ": scores[:m], "Classe MCDM": y[:m], "Classe prédite": model.predict(X_all[:m])},
                           index=ss.alts)
        st.markdown("**Alternatives réelles**")
        show_df(tbl.sort_values("Score Sᵢ", ascending=False), height=min(420, 38 * (m + 1)))
    with c2:
        if cm is not None:
            plot(heatmap(pd.DataFrame(cm, index=[f"Réelle {c}" for c in labels], columns=[f"Prédite {c}" for c in labels]),
                         "Matrice de confusion (validation croisée)", fmt=".0f"))
    st.subheader("Prédire la classe d'une nouvelle alternative")
    base = pd.DataFrame([np.round(np.median(X_array(), axis=0), 4)], index=["Nouvelle alternative"], columns=ss.crits)
    new = persistent_editor(f"ai_new|{'|'.join(ss.crits)}", base, width="stretch")
    x_new = new.astype(float).fillna(0).values
    cls = model.predict(x_new)[0]
    c1, c2 = st.columns([2, 3])
    with c1:
        winner(f"Classe {cls}", f"prédite par « {model_name} »", kicker="Nouvelle alternative")
        lo, hi = X_all.min(axis=0), X_all.max(axis=0)
        if ((x_new[0] < lo) | (x_new[0] > hi)).any():
            note("Certaines valeurs sortent de l'étendue des données d'apprentissage : la prédiction extrapole.", warn=True)
    with c2:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_new)[0]
            classes = list(model.classes_)
            fig = go.Figure(go.Bar(x=[f"Classe {c}" for c in classes], y=proba,
                                   marker=dict(color=[PALETTE[0] if c == cls else "#b7d0f0" for c in classes], cornerradius=4),
                                   text=[f"{p:.0%}" for p in proba], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x} : %{y:.1%}<extra></extra>"))
            fig = style_fig(fig, 260, "Probabilités d'appartenance")
            fig.update_yaxes(range=[0, 1.15], tickformat=".0%")
            plot(fig)
    st.caption("Limite : le modèle apprend à imiter la méthode MCDM ; il hérite de ses poids et de ses hypothèses.")


# ---------------------------------------------------------------------------
# Scénario 2 — XAI comme approche de pondération
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul des valeurs de Shapley…")
def _fit_and_explain(model_name: str, X_all: np.ndarray, scores: np.ndarray, m: int, seed: int):
    model = REGRESSORS[model_name]()
    n = len(X_all)
    r2 = None
    if n >= 10:
        pred = cross_val_predict(REGRESSORS[model_name](), X_all, scores, cv=KFold(5, shuffle=True, random_state=0))
        r2 = float(r2_score(scores, pred))
    model.fit(X_all, scores)
    rng = np.random.default_rng(seed)
    bg = X_all[rng.choice(n, size=min(30, n), replace=False)]
    explain = X_all[:min(m, 40)]
    phi0, phi = AI.shapley_values(model.predict, explain, bg)
    return r2, phi0, phi, model.predict(explain)


def _scenario2(data) -> None:
    X_all, scores, m, scorer = data
    st.markdown("**Principe.** ① Poids initiaux par MCDM. ② Scores Sᵢ par une méthode additive. ③ Apprendre "
                "Ŝ = f(x). ④ Expliquer f par les valeurs de Shapley : Ŝ = φ₀ + Σ φⱼ. ⑤ Mettre à jour les poids.")
    st.latex(r"w_j^{new}=\alpha\,w_j+(1-\alpha)\,\frac{|\phi_j|}{\sum_k|\phi_k|}")
    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Modèle de régression", list(REGRESSORS), key="ai_reg")
    alpha = c2.slider("α : part conservée des poids initiaux", 0.0, 1.0, 0.5, 0.05, key="ai_alpha")
    r2, phi0, phi, pred = _fit_and_explain(model_name, X_all, scores, m, int(ss.get("ai_seed", 0)))
    upd = AI.shap_weight_update(ss.weights, phi, alpha)
    exact = len(ss.crits) <= 10
    stats([("R² (validation croisée)", f"{r2:.3f}" if r2 is not None else "—", "qualité de Ŝ = f(x)"),
           ("Alternatives expliquées", str(len(phi))),
           ("Calcul de Shapley", "exact" if exact else "permutations", f"{2 ** len(ss.crits)} coalitions" if exact else "n > 10 critères"),
           ("Critère le plus influent", ss.crits[int(np.argmax(upd["importance"]))])])
    if r2 is not None and r2 < 0.8:
        note("R² inférieur à 0,8 : le modèle reproduit mal le score MCDM, les importances sont à interpréter avec prudence. "
             "Ajoutez des alternatives synthétiques ou changez de modèle.", warn=True)
    c1, c2 = st.columns(2)
    with c1:
        plot(grouped_bars(ss.crits, {"Poids initiaux": ss.weights, "Importance Shapley": upd["importance_norm"],
                                     "Nouveaux poids": upd["weights"]}, "Mise à jour des poids"))
    with c2:
        i = st.selectbox("Explication locale d'une alternative", range(len(phi)), format_func=lambda k: ss.alts[k], key="ai_local")
        order = np.argsort(np.abs(phi[i]))
        fig = go.Figure(go.Bar(x=phi[i][order], y=[ss.crits[j] for j in order], orientation="h",
                               marker=dict(color=[PALETTE[0] if v >= 0 else PALETTE[1] for v in phi[i][order]], cornerradius=4),
                               hovertemplate="%{y} : φ = %{x:+.4f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#5b6677", width=1))
        plot(style_fig(fig, max(260, 30 * len(ss.crits) + 90),
                       f"Contributions φⱼ — Ŝ = {pred[i]:.3f} = φ₀ ({phi0[i]:.3f}) + Σφⱼ ({phi[i].sum():+.3f})"))
        st.caption("Bleu : le critère pousse le score au-dessus de la moyenne ; orange : en dessous.")
    show_df(pd.DataFrame({"Poids initiaux": ss.weights, "Importance |φ| moyenne": upd["importance"],
                          "Importance normalisée": upd["importance_norm"], "Nouveaux poids": upd["weights"]},
                         index=ss.crits).T)
    if st.button("Appliquer les nouveaux poids", type="primary", icon=":material/check:", key="ai_apply"):
        set_weights(upd["weights"], f"MCDM + Shapley (α = {alpha:g})")
        st.rerun()


# ---------------------------------------------------------------------------
# MCDM dynamique
# ---------------------------------------------------------------------------
def _dynamic() -> None:
    st.markdown("Les critères évoluent dans le temps. Pour chaque couple (alternative, critère), un modèle de série "
                "temporelle prédit la valeur à l'horizon t + 1 ; la matrice prévue est ensuite classée par une méthode MCDM.")
    st.latex(r"\hat x_{ij}(t+1)=\text{Modèle}\big(x_{ij}(1),\dots,x_{ij}(t)\big)\ \longrightarrow\ \hat D(t+1)\ \longrightarrow\ \text{MCDM}")
    src = st.radio("Données", ["demo", "csv"], horizontal=True, key="dyn_src",
                   format_func={"demo": "Exemple du cours : 3 actions (séries simulées)", "csv": "Importer un CSV"}.get)
    if src == "demo":
        d = AI.demo_stock_series()
        t, series = d["t"], d["series"]
        default_types, default_w = ["+", "-", "+"], [0.4, 0.4, 0.2]
    else:
        up = st.file_uploader("CSV au format long : colonnes « t », « alternative », puis une colonne par critère.", type=["csv"], key="dyn_up")
        if up is None:
            st.info("Importez un fichier pour continuer, ou revenez à l'exemple du cours.")
            return
        try:
            raw = pd.read_csv(io.BytesIO(up.getvalue()), sep=None, engine="python")
            raw.columns = [str(c).strip() for c in raw.columns]
            tcol, acol = raw.columns[0], raw.columns[1]
            crit_cols = list(raw.columns[2:])
            raw = raw.sort_values(tcol)
            t = sorted(raw[tcol].unique())
            series = {str(a): {c: pd.to_numeric(g[c], errors="coerce").values for c in crit_cols} for a, g in raw.groupby(acol, sort=False)}
            if not crit_cols or len(series) < 2 or len({len(next(iter(v.values()))) for v in series.values()}) != 1:
                raise ValueError("il faut au moins 2 alternatives, 1 critère et le même nombre de dates par alternative")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Import impossible : {exc}")
            return
        default_types, default_w = ["+"] * len(crit_cols), [1 / len(crit_cols)] * len(crit_cols)
    alts = list(series)
    crits = list(next(iter(series.values())))
    T = len(next(iter(series[alts[0]].values())))
    c1, c2 = st.columns([2, 3])
    with c1:
        cfg = persistent_editor(f"dyn_cfg|{src}|{'|'.join(crits)}",
                                pd.DataFrame({"Type": default_types, "Poids": default_w}, index=crits), width="stretch",
                                column_config={"Type": st.column_config.SelectboxColumn(options=["+", "-"], required=True)})
    with c2:
        d1, d2 = st.columns(2)
        model = d1.selectbox("Modèle de prévision", list(AI.FORECASTERS), index=2, format_func=AI.FORECASTERS.get, key="dyn_model")
        method = d2.selectbox("Méthode MCDM", METHODS, index=3, key="dyn_method")
        window = d1.slider("Fenêtre (moyenne mobile)", 2, 20, 5, key="dyn_win", disabled=model != "ma")
        a_ses = d2.slider("α (lissage exponentiel)", 0.05, 1.0, 0.4, 0.05, key="dyn_a", disabled=model != "ses")
    types = list(cfg["Type"])
    w = cfg["Poids"].astype(float).clip(lower=0).values
    if w.sum() <= 0:
        st.error("La somme des poids doit être positive.")
        return
    w = w / w.sum()
    kw = {"window": window, "alpha": a_ses}
    D_hat = AI.forecast_matrix(series, model, **kw)
    mparams = {"norm": "minmax" if any((np.asarray(v) <= 0).any() for a in series.values() for v in a.values()) else "linear"}
    try:
        res = method_fn(method, mparams)(D_hat, w, types)
    except ValueError as exc:
        st.error(f"{method} : {exc}")
        return
    best = alts[int(np.argmin(res["ranks"]))]
    winner(best, f"{method} appliquée à la matrice prévue à t + 1 = {T + 1} ({AI.FORECASTERS[model]})")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matrice de décision prévue D̂(t + 1)**")
        show_df(pd.DataFrame(D_hat, index=alts, columns=[f"{c} ({s})" for c, s in zip(crits, types)]).assign(Rang=res["ranks"]))
        if st.button("Charger cette matrice dans l'espace de décision", icon=":material/arrow_forward:", key="dyn_load"):
            load_decision_problem(alts, crits, types, D_hat, w, label=f"MCDM dynamique — prévision t+1 ({AI.FORECASTERS[model]})")
            st.rerun()
    with c2:
        vals = res["Q"] if method == "VIKOR" else res.get("mean_rank", res["scores"]) if method == "ELECTRE III" else res["scores"]
        lower = method in ("VIKOR", "ELECTRE III")
        plot(bar_scores(alts, vals, f"{method} à t + 1" + (" (plus faible = meilleur)" if lower else ""), higher_better=not lower))
    st.subheader("Séries et prévisions")
    cols = st.columns(min(len(crits), 3))
    for j, c in enumerate(crits):
        with cols[j % len(cols)]:
            fig = go.Figure()
            for i, a in enumerate(alts):
                col = PALETTE[i % 8] if i < 8 else "#c5cad3"
                fig.add_scatter(x=list(t), y=series[a][c], mode="lines", name=a, line=dict(color=col, width=2), legendgroup=a,
                                hovertemplate=f"{a} : %{{y:.4g}}<extra></extra>")
                fig.add_scatter(x=[t[-1], T + 1 if isinstance(t[-1], (int, np.integer)) else "t+1"], y=[series[a][c][-1], D_hat[i, j]],
                                mode="lines+markers", line=dict(color=col, width=2, dash="dot"),
                                marker=dict(size=[0, 9], color=col, line=dict(color="#fff", width=2)), legendgroup=a,
                                showlegend=False, hovertemplate=f"{a} · prévision : %{{y:.4g}}<extra></extra>")
            fig = style_fig(fig, 300, f"{c} ({types[j]})")
            fig.update_layout(showlegend=j == 0)
            plot(fig)
    start = max(8, T // 3)
    if T - start >= 3:
        hist = []
        for u in range(start, T + 1):
            try:
                hist.append(method_fn(method, mparams)(AI.forecast_matrix(series, model, upto=u, **kw), w, types)["ranks"])
            except ValueError:
                hist.append(np.full(len(alts), np.nan))
        hist = np.array(hist)
        plot(line_chart([u + 1 for u in range(start, T + 1)], {a: hist[:, i] for i, a in enumerate(alts)},
                        "Classement dynamique : rang obtenu en refaisant la prévision à chaque date",
                        xtitle="Date de la prévision (t + 1)", ytitle="Rang (1 = meilleure)", reverse_y=True, step=True,
                        fmt=".0f", height=300))
        changes = int((np.argmin(hist, axis=1)[1:] != np.argmin(hist, axis=1)[:-1]).sum())
        st.caption(f"L'alternative classée première change {changes} fois sur la période.")
    note("Limites signalées par le cours : <b>propagation des erreurs</b> (une erreur de prévision biaise le classement) "
         "et <b>dépendance au modèle</b> de prévision. Comparez plusieurs modèles : si le classement change, il n'est pas robuste. "
         "Les modèles proposés ici sont volontairement simples ; ARIMA, GARCH ou LSTM peuvent leur être substitués.", warn=True)


def render() -> None:
    page_header("Aller plus loin", "MCDM et IA",
                "Trois couplages présentés dans le cours : apprendre un classifieur à partir d'un score MCDM, "
                "réviser les poids à l'aide de l'explicabilité (Shapley), et classer des alternatives dont les "
                "critères évoluent dans le temps.")
    tabs = st.tabs(["Scénario 1 — Classification supervisée", "Scénario 2 — Pondération par XAI", "MCDM dynamique"])
    if not HAS_SKLEARN:
        for tab in tabs[:2]:
            with tab:
                st.error("Cette partie nécessite scikit-learn (`pip install scikit-learn`).")
    else:
        with tabs[0]:
            st.caption(f"Problème courant : {ss.get('problem_label') or 'personnalisé'} · poids : {ss.weights_source}")
            data = _dataset()
            if data is not None:
                _scenario1(data)
        with tabs[1]:
            if data is not None:
                _scenario2(data)
            else:
                st.info("Réglez d'abord la base d'apprentissage dans l'onglet « Scénario 1 ».")
    with tabs[2]:
        _dynamic()
