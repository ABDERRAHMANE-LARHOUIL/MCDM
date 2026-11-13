"""État de session, éditeurs persistants, mise en forme et graphiques partagés."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import presets as P

ss = st.session_state

# Palette catégorielle validée (ordre fixe, jamais recyclée) + rampes
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#898781"
SEQ_BLUE = [[0.0, "#f1f6fd"], [0.5, "#86b3ea"], [1.0, "#17539f"]]
DIVERGING = [[0.0, "#eb6834"], [0.5, "#efefec"], [1.0, "#2a78d6"]]
GRID = "#e6e8ee"
INK, MUTED, ACCENT, ACCENT_SOFT = "#0f1b2d", "#5b6677", "#2a78d6", "#b7d0f0"
FONT = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select, [data-testid="stMetricValue"] {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container { max-width: 1300px; padding-top: 2.6rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -0.015em; color: #0f1b2d; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; padding-top: .6rem !important; }
.pg-head { margin: 0 0 1.3rem 0; padding-bottom: 1.05rem; border-bottom: 1px solid #dfe3ea; }
.pg-kicker { font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: #2a78d6; }
.pg-title { font-size: 2.05rem; font-weight: 700; line-height: 1.15; margin: .45rem 0 .35rem; color: #0f1b2d; letter-spacing: -0.02em; }
.pg-sub { color: #5b6677; font-size: 1rem; max-width: 64rem; line-height: 1.5; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .75rem; margin: .2rem 0 1.1rem; }
.stat { background: #fff; border: 1px solid #dfe3ea; border-radius: 12px; padding: .85rem 1rem; }
.stat .l { font-size: .7rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: #5b6677; }
.stat .v { font: 500 1.5rem/1.25 'IBM Plex Mono', ui-monospace, monospace; color: #0f1b2d; margin-top: .25rem; overflow-wrap: anywhere; }
.stat .v.txt { font: 600 1.05rem/1.35 'IBM Plex Sans', system-ui, sans-serif; }
.stat .h { font-size: .8rem; color: #5b6677; margin-top: .15rem; }
.win { background: #0f1b2d; color: #fff; border-radius: 14px; padding: 1rem 1.3rem; margin: .3rem 0 1.1rem;
       display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
.win .k { font-size: .7rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: #8fb8ee; }
.win .n { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.01em; }
.win .d { color: #c3ccd9; font-size: .92rem; }
.note { border-left: 3px solid #2a78d6; background: #eaf2fc; color: #1d3557; padding: .65rem .9rem; border-radius: 0 8px 8px 0;
        font-size: .92rem; margin: .4rem 0 1rem; line-height: 1.5; }
.note.warn { border-color: #eda100; background: #fdf5df; color: #5a4300; }
.badge { display: inline-block; padding: .14rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600; white-space: nowrap; }
.badge.ok { background: #e1f3ea; color: #0b6b47; } .badge.ko { background: #fde7e6; color: #a32423; }
.badge.mid { background: #fdf0d2; color: #7a5600; }
.step { background: #fff; border: 1px solid #dfe3ea; border-radius: 12px; padding: 1rem 1.1rem; min-height: 9.5rem; }
.step .i { font: 500 .8rem 'IBM Plex Mono', ui-monospace, monospace; color: #2a78d6; }
.step .t { font-weight: 600; font-size: 1.05rem; margin: .25rem 0 .3rem; color: #0f1b2d; }
.step .x { color: #5b6677; font-size: .9rem; line-height: 1.45; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #dfe3ea; border-radius: 12px; padding: .75rem 1rem; }
div[data-testid="stMetricLabel"] p { font-size: .78rem; font-weight: 600; color: #5b6677; }
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 1.45rem; font-weight: 500; }
div[data-testid="stTabs"] button[role="tab"] p { font-size: .95rem; font-weight: 500; }
[data-testid="stMain"] div[data-testid="stExpander"] details { background: #fff; border-radius: 10px; }
[data-testid="stSidebarNav"]::before { content: "MCDM · EMI"; display: block; font-weight: 700; font-size: 1.25rem;
        letter-spacing: -0.01em; color: #fff; padding: 0 0 .1rem 1.9rem; }
[data-testid="stSidebarNav"]::after { content: "Aide à la décision multicritère"; display: block; position: absolute; top: 1.75rem;
        left: 1.9rem; font-size: .78rem; color: #9fb0c8; }
[data-testid="stSidebarNav"] { position: relative; }
[data-testid="stSidebar"] { background-color: #0f1b2d; }
[data-testid="stSidebarNav"] a span, [data-testid="stSidebarNav"] header, [data-testid="stSidebarNavSeparator"],
[data-testid="stSidebar"] summary, [data-testid="stSidebar"] summary p { color: #e6ebf2; }
[data-testid="stSidebarNav"] > ul { margin-top: 1.9rem; }
[data-testid="stSidebar"] .side-brand { font-weight: 700; font-size: 1.2rem; letter-spacing: -0.01em; color: #fff; }
[data-testid="stSidebar"] .side-sub { font-size: .8rem; color: #9fb0c8; margin-bottom: .4rem; }
[data-testid="stSidebar"] .side-box { border: 1px solid #2a3c57; border-radius: 10px; padding: .7rem .8rem; font-size: .83rem;
        color: #c8d3e3; line-height: 1.5; margin-top: .6rem; }
[data-testid="stSidebar"] .side-box b { color: #fff; font-weight: 600; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Blocs de mise en page
# ---------------------------------------------------------------------------
def page_header(kicker: str, title: str, sub: str = "") -> None:
    st.markdown(f'<div class="pg-head"><div class="pg-kicker">{kicker}</div>'
                f'<div class="pg-title">{title}</div><div class="pg-sub">{sub}</div></div>',
                unsafe_allow_html=True)


def stats(items) -> None:
    """Cartes de chiffres clés : liste de (libellé, valeur[, aide]). Valeur str → affichée telle quelle."""
    html = ""
    for label, value, *hint in items:
        if isinstance(value, str):
            numeric = value.replace(".", "").replace(",", "").replace("−", "").replace("-", "").replace("%", "").replace(" ", "").isdigit()
            val = value
        else:
            numeric, val = True, f"{value:.4g}"
        h = f'<div class="h">{hint[0]}</div>' if hint and hint[0] else ""
        html += (f'<div class="stat"><div class="l">{label}</div>'
                 f'<div class="v{"" if numeric else " txt"}">{val}</div>{h}</div>')
    st.markdown(f'<div class="stats">{html}</div>', unsafe_allow_html=True)


def winner(name: str, detail: str = "", kicker: str = "Meilleure alternative") -> None:
    st.markdown(f'<div class="win"><span class="k">{kicker}</span><span class="n">{name}</span>'
                f'<span class="d">{detail}</span></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "ok") -> str:
    """Pastille HTML (ok / ko / mid) à insérer dans un st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="badge {kind}">{text}</span>'


def yes_no(ok: bool) -> str:
    """Icône Material colorée utilisable dans tout texte Markdown Streamlit."""
    return ":green[:material/check_circle:]" if ok else ":red[:material/cancel:]"


def color_for(i: int) -> str:
    return PALETTE[i] if i < len(PALETTE) else OTHER


# ---------------------------------------------------------------------------
# État global
# ---------------------------------------------------------------------------
def init_state() -> None:
    if ss.get("_initialised"):
        return
    ss._initialised = True
    ss.weight_sets = {}
    ss.results = {}
    load_decision_problem(**_preset_kwargs(next(iter(P.DECISION_PRESETS))))


def _preset_kwargs(name: str) -> dict:
    p = P.DECISION_PRESETS[name]
    return {"alternatives": p["alternatives"], "criteria": p["criteria"], "types": p["types"],
            "matrix": p["matrix"], "weights": p["weights"], "label": name}


def load_preset(name: str) -> None:
    load_decision_problem(**_preset_kwargs(name))


def load_decision_problem(alternatives, criteria, types, matrix, weights=None, label="") -> None:
    ss.alts = list(alternatives)
    ss.crits = list(criteria)
    ss.types = list(types)
    ss.X = pd.DataFrame(np.asarray(matrix, dtype=float), index=ss.alts, columns=ss.crits)
    n = len(ss.crits)
    if weights is None:
        ss.weights = np.full(n, 1.0 / n)
        ss.weights_source = "Équipondération (par défaut)"
    else:
        w = np.asarray(weights, dtype=float)
        ss.weights = w / w.sum()
        ss.weights_source = "Poids donnés par l'énoncé"
    ss.weight_sets = {}
    ss.results = {}
    ss.problem_label = label
    reset_editor("crit", pd.DataFrame({"Critère": ss.crits, "Type": ss.types}))
    reset_editor("alt", pd.DataFrame({"Alternative": ss.alts}))
    reset_editor("matrix", ss.X.copy())


def set_weights(w, source: str) -> None:
    w = np.asarray(w, dtype=float)
    ss.weights = w / w.sum()
    ss.weights_source = source
    ss.results = {}


def X_array() -> np.ndarray:
    return ss.X.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Éditeurs persistants (survivent aux changements de page)
# ---------------------------------------------------------------------------
def persistent_editor(name: str, default: pd.DataFrame, **kwargs) -> pd.DataFrame:
    store, base = f"_val_{name}", f"_base_{name}"
    key = f"_ed_{name}_{ss.get(f'_v_{name}', 0)}"
    if store not in ss:
        ss[store] = default.copy()
    if key not in ss:
        ss[base] = ss[store].copy()
    edited = st.data_editor(ss[base], key=key, **kwargs)
    ss[store] = edited
    return edited


def reset_editor(name: str, df: pd.DataFrame) -> None:
    ss[f"_val_{name}"] = df.copy()
    ss[f"_v_{name}"] = ss.get(f"_v_{name}", 0) + 1


def editor_value(name: str, default: pd.DataFrame) -> pd.DataFrame:
    return ss.get(f"_val_{name}", default)


# ---------------------------------------------------------------------------
# Mise en forme
# ---------------------------------------------------------------------------
def show_df(df: pd.DataFrame, decimals: int = 4, highlight_max: str | None = None,
            highlight_min: str | None = None, **kwargs) -> None:
    sty = df.style.format(precision=decimals, na_rep="—")
    if highlight_max and highlight_max in df.columns:
        sty = sty.highlight_max(subset=[highlight_max], color="rgba(42,120,214,0.25)")
    if highlight_min and highlight_min in df.columns:
        sty = sty.highlight_min(subset=[highlight_min], color="rgba(42,120,214,0.25)")
    st.dataframe(sty, width="stretch", **kwargs)


def weights_df(w, crits=None) -> pd.DataFrame:
    crits = crits or ss.crits
    return pd.DataFrame({"Poids": np.asarray(w, dtype=float)}, index=crits).T


def tfn_str(t, d: int = 3) -> str:
    return "(" + ", ".join(f"{v:.{d}f}" for v in t) + ")"


def tfn_frame(T, index, columns, d: int = 3) -> pd.DataFrame:
    T = np.asarray(T)
    return pd.DataFrame([[tfn_str(T[i, j], d) for j in range(T.shape[1])] for i in range(T.shape[0])],
                        index=index, columns=columns)


def check_badge(value: float, expected: float, tol: float = 0.01) -> str:
    return ":green[:material/check_circle:]" if abs(value - expected) <= tol else ":orange[:material/warning:]"


# ---------------------------------------------------------------------------
# Graphiques Plotly
# ---------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=78 if title else 34, b=10),
        title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top", pad=dict(t=6, l=2),
                   font=dict(size=15, color=INK)) if title else None,
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", colorway=PALETTE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", font=dict(size=12, color=MUTED)),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color="#ffffff", family=FONT, size=12)),
        bargap=0.35,
    )
    axis = dict(gridcolor=GRID, zeroline=False, linecolor="#cfd4dc", tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=12), automargin=True, title_standoff=10)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def bar_scores(names, scores, title: str, higher_better: bool = True, fmt: str = ".4f",
               color: str = PALETTE[0]) -> go.Figure:
    """Barres horizontales triées ; la meilleure alternative (en haut) est mise en avant."""
    order = np.argsort(scores)
    if not higher_better:
        order = order[::-1]
    names = [names[i] for i in order]
    vals = [float(scores[i]) for i in order]
    colors = [ACCENT_SOFT] * len(vals)
    colors[-1] = color
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{v:{fmt}}" for v in vals], textposition="outside", textfont=dict(color=INK, size=12),
        cliponaxis=False, hovertemplate="%{y} : %{x:" + fmt + "}<extra></extra>"))
    fig.update_xaxes(title=None, range=[min(0, min(vals) * 1.15), max(vals) * 1.15 if max(vals) > 0 else 1])
    fig = style_fig(fig, height=max(220, 40 * len(names) + 90), title=title)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return fig


def grouped_bars(categories, series: dict[str, np.ndarray], title: str, fmt: str = ".3f") -> go.Figure:
    fig = go.Figure()
    for k, (name, vals) in enumerate(series.items()):
        fig.add_bar(name=name, x=list(categories), y=list(vals), marker_color=color_for(k),
                    marker=dict(cornerradius=4),
                    hovertemplate=f"{name}<br>%{{x}} : %{{y:{fmt}}}<extra></extra>")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    return style_fig(fig, height=380, title=title)


def radar(X_norm: np.ndarray, alts, crits, title: str) -> go.Figure:
    fig = go.Figure()
    for i, a in enumerate(alts):
        r = list(X_norm[i]) + [X_norm[i][0]]
        fig.add_trace(go.Scatterpolar(r=r, theta=list(crits) + [crits[0]], name=a, fill="toself", fillcolor="rgba(0,0,0,0)",
                                      line=dict(color=color_for(i), width=2),
                                      hovertemplate=f"{a}<br>%{{theta}} : %{{r:.3f}}<extra></extra>"))
    fig = style_fig(fig, height=430, title=title)
    fig.update_layout(margin=dict(l=60, r=60, t=70, b=30),
                      polar=dict(bgcolor="#ffffff",
                                 radialaxis=dict(range=[0, 1], gridcolor=GRID, linecolor=GRID, tickfont=dict(size=10, color=MUTED)),
                                 angularaxis=dict(gridcolor=GRID, linecolor="#cfd4dc", tickfont=dict(size=11, color=INK))))
    return fig


def heatmap(M: pd.DataFrame, title: str, diverging: bool = False, zmin=None, zmax=None,
            fmt: str = ".2f") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=M.values, x=list(map(str, M.columns)), y=list(map(str, M.index)),
        colorscale=DIVERGING if diverging else SEQ_BLUE, zmin=zmin, zmax=zmax,
        text=[[f"{v:{fmt}}" for v in row] for row in M.values],
        texttemplate="%{text}" if M.size <= 400 else None, textfont=dict(size=11),
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=MUTED)),
        xgap=2, ygap=2, hovertemplate="%{y} × %{x} : %{z:" + fmt + "}<extra></extra>"))
    fig = style_fig(fig, height=max(280, min(50, 900 // max(len(M), 1)) * len(M) + 120), title=title)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=INK))
    return fig


def series_colors(names, focus=None) -> dict:
    """Couleur stable par entité. Au-delà de 8 séries, seules celles de `focus` sont colorées."""
    names = list(names)
    if len(names) <= len(PALETTE):
        return {n: PALETTE[i] for i, n in enumerate(names)}
    focus = [n for n in (focus or names[:6]) if n in names][:len(PALETTE)]
    return {n: (PALETTE[focus.index(n)] if n in focus else "#c5cad3") for n in names}


def line_chart(x, series: dict, title: str | None = None, xtitle: str = "", ytitle: str = "", colors: dict | None = None,
               height: int = 380, reverse_y: bool = False, markers: bool = False, vline=None, fmt: str = ".3f",
               step: bool = False) -> go.Figure:
    """Courbes multi-séries ; les séries neutres (grises) passent dessous et sortent de la légende."""
    colors = colors or series_colors(series)
    fig = go.Figure()
    for name in sorted(series, key=lambda n: colors[n] != "#c5cad3"):
        c = colors[name]
        muted = c == "#c5cad3"
        fig.add_scatter(x=list(x), y=list(series[name]), name=str(name), mode="lines+markers" if markers else "lines",
                        line=dict(color=c, width=1.2 if muted else 2, shape="hv" if step else "linear"),
                        marker=dict(size=8, line=dict(color="#ffffff", width=2)), showlegend=not muted,
                        hovertemplate=f"{name} : %{{y:{fmt}}}<extra></extra>")
    fig = style_fig(fig, height, title)
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle, autorange="reversed" if reverse_y else True)
    if reverse_y:
        fig.update_yaxes(dtick=1)
    if vline is not None:
        fig.add_vline(x=vline, line=dict(color=MUTED, width=1, dash="dot"))
    return fig
