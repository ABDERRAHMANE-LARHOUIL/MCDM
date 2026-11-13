"""Page — Logique floue : fonctions d'appartenance, arithmétique floue et extensions
(intuitionniste, pythagoricienne, q-ROF, picturale, hésitante)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import fuzzy_logic as FL
from ui.common import INK, MUTED, PALETTE, note, page_header, persistent_editor, plot, show_df, stats, style_fig


def _fmt(t, d: int = 3) -> str:
    return "(" + " ; ".join(f"{v:.{d}f}".rstrip("0").rstrip(".") if d else f"{v:g}" for v in t) + ")"


def _floats(text: str) -> list[float]:
    out = []
    for tok in str(text).replace(";", ",").split(","):
        try:
            out.append(float(tok.strip()))
        except ValueError:
            pass
    return out


def _curve_fig(x, curves: dict, title: str, alpha: float | None = None, xmark: float | None = None,
               height: int = 330) -> go.Figure:
    fig = go.Figure()
    for k, (name, mu) in enumerate(curves.items()):
        fig.add_scatter(x=x, y=mu, name=name, mode="lines", line=dict(color=PALETTE[k % 8], width=2),
                        hovertemplate=name + " · x = %{x:.3g} · μ = %{y:.3f}<extra></extra>")
    fig = style_fig(fig, height, title)
    fig.update_layout(showlegend=len(curves) > 1)
    fig.update_yaxes(title="μ(x)", range=[-0.03, 1.1])
    fig.update_xaxes(title="x")
    if alpha is not None:
        fig.add_hline(y=alpha, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=f"α = {alpha:.2f}",
                      annotation_font=dict(color=MUTED, size=11))
    if xmark is not None:
        fig.add_vline(x=xmark, line=dict(color=MUTED, width=1, dash="dot"))
    return fig


# ---------------------------------------------------------------------------
def _membership() -> None:
    c1, c2 = st.columns([2, 3])
    with c1:
        kind = st.radio("Fonction d'appartenance", ["trap", "tri", "gauss"], horizontal=True, key="fl_kind",
                        format_func={"trap": "Trapézoïdale", "tri": "Triangulaire", "gauss": "Gaussienne"}.get)
        if kind == "gauss":
            m = st.number_input("Centre m", value=45.0, key="fl_m")
            sg = st.number_input("Écart-type σ", value=10.0, min_value=0.01, key="fl_s")
            lo, hi = m - 4 * sg, m + 4 * sg
            params = None
        else:
            default = "25, 35, 55, 65" if kind == "trap" else "25, 45, 65"
            vals = sorted(_floats(st.text_input("Paramètres " + ("a, b, c, d" if kind == "trap" else "a, b, c"),
                                                default, key=f"fl_p_{kind}")))
            need = 4 if kind == "trap" else 3
            if len(vals) != need:
                st.error(f"Saisissez {need} valeurs séparées par des virgules.")
                return
            params = vals if kind == "trap" else [vals[0], vals[1], vals[1], vals[2]]
            span = max(params[3] - params[0], 1e-9)
            lo, hi = params[0] - 0.25 * span, params[3] + 0.25 * span
        x0 = st.slider("Valeur x à évaluer", float(lo), float(hi), float((lo + hi) / 2 if kind == "gauss" else 30.0 if lo <= 30 <= hi else (lo + hi) / 2),
                       key=f"fl_x_{kind}")
        alpha = st.slider("Niveau α de la coupe", 0.0, 1.0, 0.5, 0.05, key="fl_alpha")
    x = np.linspace(lo, hi, 600)
    if kind == "gauss":
        mu, mu0 = FL.mu_gauss(x, m, sg), float(FL.mu_gauss(x0, m, sg))
        cut = FL.alpha_cut_gauss(m, sg, alpha) if alpha > 0 else (-np.inf, np.inf)
        support, core = "ℝ (μ > 0 partout)", f"{{{m:g}}}"
    else:
        mu, mu0 = FL.mu_trapezoid(x, *params), float(FL.mu_trapezoid(x0, *params))
        cut = FL.alpha_cut_trapezoid(*params, alpha)
        support, core = f"]{params[0]:g} ; {params[3]:g}[", f"[{params[1]:g} ; {params[2]:g}]"
    with c2:
        fig = _curve_fig(x, {"μ(x)": mu}, "Fonction d'appartenance et coupe-α", alpha, x0)
        if np.isfinite(cut[0]):
            fig.add_scatter(x=[cut[0], cut[1]], y=[alpha, alpha], mode="lines+markers", name="coupe-α",
                            line=dict(color=PALETTE[1], width=4), marker=dict(size=8), showlegend=False,
                            hovertemplate="coupe-α : %{x:.3g}<extra></extra>")
        fig.add_scatter(x=[x0], y=[mu0], mode="markers", marker=dict(size=11, color=INK, line=dict(color="#fff", width=2)),
                        showlegend=False, hovertemplate=f"μ({x0:.3g}) = {mu0:.3f}<extra></extra>")
        plot(fig)
    stats([(f"μ({x0:.4g})", f"{mu0:.3f}", f"x appartient à l'ensemble à {mu0:.0%}"),
           (f"Coupe-α (α = {alpha:.2f})", f"[{cut[0]:.4g} ; {cut[1]:.4g}]" if np.isfinite(cut[0]) else "ℝ"),
           ("Support", support), ("Noyau", core), ("Hauteur h(A)", "1", "ensemble normalisé")])
    st.latex(r"\text{supp}(A)=\{x\mid\mu_A(x)>0\},\quad \text{noy}(A)=\{x\mid\mu_A(x)=1\},\quad "
             r"\alpha\text{-cut}(A)=\{x\mid\mu_A(x)\ge\alpha\},\quad h(A)=\sup_x\mu_A(x)")

    st.subheader("Variable linguistique : l'exemple « Âge » du cours")
    age = st.slider("Âge", 0, 100, 30, key="fl_age")
    xa = np.linspace(0, 100, 501)
    sets = {"Jeune": (0, 0, 25, 35), "Âge moyen": (25, 35, 55, 65), "Âgé": (55, 65, 100, 100)}
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(_curve_fig(xa, {k: FL.mu_trapezoid(xa, *v) for k, v in sets.items()},
                        "Trois ensembles flous sur l'univers [0, 100]", xmark=age, height=300))
    with c2:
        show_df(pd.DataFrame({"Degré d'appartenance μ": {k: float(FL.mu_trapezoid(age, *v)) for k, v in sets.items()}}), 2)
        st.caption("À 30 ans, une personne est « jeune » et d'« âge moyen » à 0,5 chacun : la logique floue capture "
                   "la transition graduelle que la logique booléenne ignore.")


# ---------------------------------------------------------------------------
def _arithmetic() -> None:
    st.markdown("Nombres flous trapézoïdaux positifs A(a₁, a₂, a₃, a₄) et B(b₁, b₂, b₃, b₄). "
                "Avec a₂ = a₃ le nombre est triangulaire ; avec quatre valeurs égales, c'est un nombre réel.")
    c1, c2, c3 = st.columns([3, 3, 1])
    A = sorted(_floats(c1.text_input("A = (a₁, a₂, a₃, a₄)", "4, 5, 6, 7", key="fa_A")))
    B = sorted(_floats(c2.text_input("B = (b₁, b₂, b₃, b₄)", "1, 2, 3, 4", key="fa_B")))
    lam = c3.number_input("λ", value=2.0, min_value=0.0, key="fa_lam")
    if len(A) == 3:
        A = [A[0], A[1], A[1], A[2]]
    if len(B) == 3:
        B = [B[0], B[1], B[1], B[2]]
    if len(A) != 4 or len(B) != 4:
        st.error("Saisissez 3 valeurs (triangulaire) ou 4 valeurs (trapézoïdal) pour A et pour B.")
        return
    res = {"A ⊕ B": FL.fz_add(A, B), "A ⊖ B": FL.fz_sub(A, B), "A ⊗ B": FL.fz_mul(A, B), f"{lam:g}·A": FL.fz_scale(lam, A)}
    if B[0] > 0:
        res["A ⊘ B"] = FL.fz_div(A, B)
    else:
        note("La division A ⊘ B exige B strictement positif.", warn=True)
    st.latex(r"A\oplus B=(a_1+b_1,\dots,a_4+b_4),\qquad A\ominus B=(a_1-b_4,\,a_2-b_3,\,a_3-b_2,\,a_4-b_1)")
    st.latex(r"A\otimes B=(a_1b_1,\dots,a_4b_4),\qquad A\oslash B=\Big(\frac{a_1}{b_4},\frac{a_2}{b_3},\frac{a_3}{b_2},\frac{a_4}{b_1}\Big)")
    c1, c2 = st.columns([2, 3])
    with c1:
        show_df(pd.DataFrame(res, index=["1", "2", "3", "4"]).T, 4)
    with c2:
        pick = st.multiselect("Résultats à tracer", list(res), default=["A ⊕ B", "A ⊖ B"], key="fa_pick")
        shown = {"A": tuple(A), "B": tuple(B), **{k: res[k] for k in pick}}
        lo = min(v[0] for v in shown.values())
        hi = max(v[3] for v in shown.values())
        pad = 0.08 * max(hi - lo, 1e-9)
        x = np.linspace(lo - pad, hi + pad, 800)
        plot(_curve_fig(x, {k: FL.mu_trapezoid(x, *v) for k, v in shown.items()}, "Nombres flous", height=320))
    with st.expander("Arithmétique des intervalles"):
        d1, d2 = st.columns(2)
        Ia = sorted(_floats(d1.text_input("A = [a₁, a₂]", "2, 5", key="fa_Ia")))
        Ib = sorted(_floats(d2.text_input("B = [b₁, b₂]", "1, 3", key="fa_Ib")))
        if len(Ia) == 2 and len(Ib) == 2:
            ops = FL.interval_ops(Ia, Ib)
            st.markdown(" · ".join(f"**{k}** = {'[' + ' ; '.join(f'{v:.4g}' for v in val) + ']' if len(val) == 2 else f'{val[0]:.4g}'}"
                                   for k, val in ops.items()))
            st.caption("Ordre : A ≤ B ⟺ m(A) ≤ m(B). Un intervalle est un nombre flou dont l'appartenance vaut 1 "
                       "dans l'intervalle et 0 ailleurs.")
        else:
            st.error("Deux bornes par intervalle.")


# ---------------------------------------------------------------------------
def _orthopair() -> None:
    st.markdown("Un nombre flou **intuitionniste** (q = 1), **pythagoricien** (q = 2) ou **q-Rung Orthopair** "
                "associe un degré d'appartenance μ et de non-appartenance ν tels que μ^q + ν^q ≤ 1.")
    c1, c2, c3 = st.columns(3)
    mu = c1.slider("Appartenance μ", 0.0, 1.0, 0.6, 0.01, key="op_mu")
    nu = c2.slider("Non-appartenance ν", 0.0, 1.0, 0.3, 0.01, key="op_nu")
    q = c3.slider("Rang q", 1, 6, 1, key="op_q")
    pi = FL.orthopair_hesitation(mu, nu, q)
    qmin = FL.min_valid_q(mu, nu)
    kinds = [("Intuitionniste (q = 1)", mu + nu <= 1 + 1e-12), ("Pythagoricien (q = 2)", mu ** 2 + nu ** 2 <= 1 + 1e-12),
             (f"q-ROF (q = {q})", pi is not None)]
    c1, c2 = st.columns([2, 3])
    with c1:
        stats([(f"Hésitation π (q = {q})", f"{pi:.4f}" if pi is not None else "invalide",
                "plus π est élevé, plus l'incertitude est grande" if pi is not None else f"μ^{q} + ν^{q} > 1"),
               ("Plus petit q valide", str(qmin) if qmin else "aucun")])
        st.markdown("  \n".join(f"{':green[:material/check_circle:]' if ok else ':red[:material/cancel:]'} {name}"
                                for name, ok in kinds))
        st.latex(r"\pi=\sqrt[q]{1-\mu^{q}-\nu^{q}}")
        if abs(mu - 0.6) < 1e-9 and abs(nu - 0.3) < 1e-9:
            st.caption("Lecture du cours : 60 % votent pour, 30 % contre, 10 % ne se prononcent pas.")
    with c2:
        t = np.linspace(0, 1, 200)
        fig = go.Figure()
        for k, qq in enumerate(sorted({1, 2, 3, q})):
            fig.add_scatter(x=t, y=(1 - t ** qq) ** (1 / qq), mode="lines", name=f"q = {qq}",
                            line=dict(color=PALETTE[k], width=3 if qq == q else 1.5),
                            hovertemplate=f"q = {qq} · μ = %{{x:.2f}} · ν max = %{{y:.2f}}<extra></extra>")
        fig.add_scatter(x=[mu], y=[nu], mode="markers", name="(μ, ν)", marker=dict(size=13, color=INK, line=dict(color="#fff", width=2)),
                        hovertemplate=f"μ = {mu:.2f} · ν = {nu:.2f}<extra></extra>")
        fig = style_fig(fig, 400, "Espace admissible")
        fig.update_xaxes(title="μ", range=[0, 1.02], constrain="domain")
        fig.update_yaxes(title="ν", range=[0, 1.02], scaleanchor="x", constrain="domain")
        plot(fig)

    st.subheader("Opérations sur les nombres flous intuitionnistes")
    c1, c2, c3 = st.columns(3)
    a = _floats(c1.text_input("α₁ = (μ, ν)", "0.6, 0.3", key="op_a"))
    b = _floats(c2.text_input("α₂ = (μ, ν)", "0.5, 0.2", key="op_b"))
    lam = c3.number_input("λ > 0", value=2.0, min_value=0.01, key="op_lam")
    if len(a) != 2 or len(b) != 2 or a[0] + a[1] > 1 + 1e-12 or b[0] + b[1] > 1 + 1e-12 or min(a + b) < 0:
        st.error("Chaque nombre doit vérifier μ, ν ≥ 0 et μ + ν ≤ 1.")
    else:
        ops = {"α₁ ⊕ α₂": FL.ifn_add(a, b), "α₁ ⊗ α₂": FL.ifn_mul(a, b), f"{lam:g}·α₁": FL.ifn_scale(lam, a),
               f"α₁^{lam:g}": FL.ifn_pow(a, lam)}
        st.latex(r"\alpha_1\oplus\alpha_2=(\mu_1+\mu_2-\mu_1\mu_2,\ \nu_1\nu_2),\qquad "
                 r"\alpha_1\otimes\alpha_2=(\mu_1\mu_2,\ \nu_1+\nu_2-\nu_1\nu_2)")
        st.latex(r"\lambda\alpha=(1-(1-\mu)^\lambda,\ \nu^\lambda),\qquad \alpha^\lambda=(\mu^\lambda,\ 1-(1-\nu)^\lambda)")
        c1, c2 = st.columns(2)
        with c1:
            show_df(pd.DataFrame({k: {"μ": v[0], "ν": v[1], "π": 1 - v[0] - v[1]} for k, v in ops.items()}).T)
        with c2:
            show_df(pd.DataFrame({"Score S = μ − ν": [FL.ifn_score(a), FL.ifn_score(b)],
                                  "Précision H = μ + ν": [FL.ifn_accuracy(a), FL.ifn_accuracy(b)]}, index=["α₁", "α₂"]))
            cmp = FL.ifn_compare(a, b)
            st.markdown("Comparaison (score, puis précision) : **" + {1: "α₁ > α₂", -1: "α₁ < α₂", 0: "α₁ = α₂"}[cmp] + "**")

    st.subheader("Agrégation IFWA")
    base = pd.DataFrame({"μ": [0.6, 0.7, 0.5], "ν": [0.3, 0.2, 0.4], "Poids σ": [0.4, 0.35, 0.25]},
                        index=["α₁", "α₂", "α₃"])
    df = persistent_editor("ifwa_list", base, num_rows="dynamic", width="stretch").dropna()
    st.latex(r"IFWA_\sigma(\alpha_1,\dots,\alpha_k)=\Big(1-\prod_j(1-\mu_j)^{\sigma_j},\ \prod_j\nu_j^{\sigma_j}\Big)")
    P = df[["μ", "ν"]].astype(float).values
    if len(df) and (P >= 0).all() and (P.sum(axis=1) <= 1 + 1e-12).all() and df["Poids σ"].astype(float).sum() > 0:
        m_, n_ = FL.ifwa_list(P, df["Poids σ"].astype(float).values)
        stats([("μ agrégé", f"{m_:.4f}"), ("ν agrégé", f"{n_:.4f}"), ("π agrégé", f"{1 - m_ - n_:.4f}"),
               ("Score S", f"{m_ - n_:.4f}")])
    else:
        st.error("Chaque ligne doit vérifier μ, ν ≥ 0 et μ + ν ≤ 1, avec des poids de somme positive.")

    with st.expander("q-ROF à valeurs d'intervalle"):
        d = st.columns(5)
        ml = d[0].number_input("μ_L", 0.0, 1.0, 0.4, 0.05, key="iv_ml")
        mu_ = d[1].number_input("μ_U", 0.0, 1.0, 0.6, 0.05, key="iv_mu")
        nl = d[2].number_input("ν_L", 0.0, 1.0, 0.2, 0.05, key="iv_nl")
        nu_ = d[3].number_input("ν_U", 0.0, 1.0, 0.3, 0.05, key="iv_nu")
        qi = d[4].number_input("q", 1, 10, 2, key="iv_q")
        st.latex(r"\pi_L=\sqrt[q]{1-\mu_U^{q}-\nu_U^{q}},\qquad \pi_U=\sqrt[q]{1-\mu_L^{q}-\nu_L^{q}}")
        if ml > mu_ or nl > nu_:
            st.error("Il faut μ_L ≤ μ_U et ν_L ≤ ν_U.")
        else:
            r = FL.interval_qrof_hesitation(ml, mu_, nl, nu_, qi)
            if r is None:
                st.error(f"Contrainte violée : μ_U^{qi} + ν_U^{qi} > 1.")
            else:
                st.markdown(f"Degré d'hésitation : **π = [{r[0]:.4f} ; {r[1]:.4f}]**")


# ---------------------------------------------------------------------------
def _picture() -> None:
    st.markdown("Un nombre flou **pictural** distingue quatre réponses : pour (μ), contre (ν), abstention ou neutre (π), "
                "et refus de répondre, avec μ + ν + π ≤ 1. Exemple typique : un vote.")
    c1, c2, c3 = st.columns(3)
    a = _floats(c1.text_input("A = (μ, ν, π)", "0.5, 0.3, 0.1", key="pf_a"))
    b = _floats(c2.text_input("B = (μ, ν, π)", "0.4, 0.2, 0.3", key="pf_b"))
    lam = c3.number_input("λ > 0", value=2.0, min_value=0.01, key="pf_lam")
    if len(a) != 3 or len(b) != 3 or FL.picture_refusal(a) is None or FL.picture_refusal(b) is None or min(a + b) < 0:
        st.error("Chaque nombre doit avoir trois composantes positives de somme ≤ 1.")
        return
    c1, c2 = st.columns([2, 3])
    with c1:
        stats([("Refus (A)", f"{FL.picture_refusal(a):.3f}", "1 − (μ + ν + π)"), ("Refus (B)", f"{FL.picture_refusal(b):.3f}")])
        ops = {"A ⊕ B": FL.picture_add(a, b), f"{lam:g}·A": FL.picture_scale(lam, a), f"A^{lam:g}": FL.picture_pow(a, lam)}
        show_df(pd.DataFrame({k: {"μ (pour)": v[0], "ν (contre)": v[1], "π (neutre)": v[2]} for k, v in ops.items()}).T)
    with c2:
        fig = go.Figure()
        parts = [("Pour μ", 0, PALETTE[0]), ("Contre ν", 1, PALETTE[1]), ("Neutre π", 2, PALETTE[3])]
        for name, k, col in parts:
            fig.add_bar(y=["A", "B"], x=[a[k], b[k]], name=name, orientation="h", marker=dict(color=col, line=dict(color="#fff", width=2)),
                        hovertemplate=name + " : %{x:.2f}<extra></extra>")
        fig.add_bar(y=["A", "B"], x=[FL.picture_refusal(a), FL.picture_refusal(b)], name="Refus", orientation="h",
                    marker=dict(color="#c5cad3", line=dict(color="#fff", width=2)), hovertemplate="Refus : %{x:.2f}<extra></extra>")
        fig.update_layout(barmode="stack")
        fig = style_fig(fig, 240, "Répartition des quatre réponses")
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        plot(fig)
    st.latex(r"A\oplus B=(\mu_A+\mu_B-\mu_A\mu_B,\ \nu_A\nu_B,\ \pi_A\pi_B),\quad "
             r"\lambda A=(1-(1-\mu_A)^\lambda,\ \nu_A^\lambda,\ \pi_A^\lambda),\quad "
             r"A^\lambda=(\mu_A^\lambda,\ 1-(1-\nu_A)^\lambda,\ 1-(1-\pi_A)^\lambda)")


# ---------------------------------------------------------------------------
def _hesitant() -> None:
    st.markdown("Un élément flou **hésitant** h = {γ₁, …, γₖ} regroupe plusieurs degrés d'appartenance possibles, "
                "par exemple ceux proposés par différents décideurs. Les valeurs par défaut sont celles de l'exemple du cours.")
    base = pd.DataFrame({"Valeurs (séparées par des virgules)": ["0.2, 0.3, 0.5, 0.8", "0.4, 0.6, 0.8", "0.3, 0.45, 0.6"]},
                        index=["h1", "h2", "h3"])
    df = persistent_editor("hfe", base, num_rows="dynamic", width="stretch").dropna()
    H = {}
    for name, row in df.iterrows():
        vals = _floats(row.iloc[0])
        if vals and all(0 <= v <= 1 for v in vals):
            H[str(name)] = vals
        else:
            st.error(f"{name} : valeurs attendues dans [0, 1].")
    if len(H) < 2:
        st.info("Saisissez au moins deux éléments hésitants.")
        return
    st.latex(r"s(h)=\frac{1}{l}\sum_{\gamma\in h}\gamma,\qquad \sigma(h)=\sqrt{\frac{1}{l}\sum_{\gamma\in h}(\gamma-s(h))^2}")
    import functools
    order = sorted(H, key=functools.cmp_to_key(lambda p, q_: -FL.hfe_compare(H[p], H[q_])))
    tbl = pd.DataFrame({"Score s(h)": {k: FL.hfe_score(v) for k, v in H.items()},
                        "Déviation σ(h)": {k: FL.hfe_deviation(v) for k, v in H.items()},
                        "Rang": {k: order.index(k) + 1 for k in H}})
    c1, c2 = st.columns(2)
    with c1:
        show_df(tbl.sort_values("Rang"))
        chain = order[0]
        for prev, cur in zip(order, order[1:]):
            chain += (" = " if FL.hfe_compare(H[prev], H[cur]) == 0 else " > ") + cur
        st.markdown(f"Ordre : **{chain}**")
        st.caption("À score égal, l'élément le moins dispersé (σ le plus faible) est préféré.")
    with c2:
        opt = st.toggle("Compléter les éléments courts par leur plus grande valeur (optimiste)", value=False, key="hfe_opt",
                        help="Par défaut, l'élément le plus court est complété par sa plus petite valeur (pessimiste).")
        names = list(H)
        D = pd.DataFrame([[FL.hfe_hamming(H[a], H[b], opt) for b in names] for a in names], index=names, columns=names)
        st.markdown("**Distance de Hamming hésitante**")
        show_df(D)
        st.latex(r"\|h_1-h_2\|=\frac{1}{l}\sum_{k=1}^{l}\big|h_1^{\sigma(k)}-h_2^{\sigma(k)}\big|")
    with st.expander("Opérations sur les deux premiers éléments"):
        al = st.number_input("α > 0", value=2.0, min_value=0.01, key="hfe_alpha")
        n1, n2 = list(H)[:2]
        ops = FL.hfe_ops(H[n1], H[n2], al)
        for k, v in ops.items():
            st.markdown(f"**{k.replace('h1', n1).replace('h2', n2)}** = {{{', '.join(f'{x:g}' for x in v)}}}")


def render() -> None:
    page_header("Incertitude épistémique", "Logique floue",
                "Quand les jugements sont linguistiques ou imprécis, le degré d'appartenance μ(x) ∈ [0, 1] remplace "
                "le vrai / faux. Cette page couvre les ensembles flous du cours et leurs extensions.")
    tabs = st.tabs(["Fonctions d'appartenance", "Arithmétique floue", "Intuitionniste, pythagoricien, q-ROF",
                    "Pictural", "Hésitant"])
    for tab, fn in zip(tabs, [_membership, _arithmetic, _orthopair, _picture, _hesitant]):
        with tab:
            fn()
