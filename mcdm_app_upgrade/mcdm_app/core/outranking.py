"""Méthodes de surclassement (outranking) : PROMETHEE I / II (Brans & Vincke) et ELECTRE III (Roy).

Le cours introduit PROMETHEE comme méthode non compensatoire ; la
formulation implémentée est la formulation standard de Brans (1985) avec les
six fonctions de préférence généralisées.
"""
from __future__ import annotations

import numpy as np

from .crisp_ranking import is_benefit, rank_desc

PREFERENCE_FUNCTIONS = {
    "usual": "Usuel (type I)",
    "u-shape": "En U (type II, seuil q)",
    "v-shape": "En V (type III, seuil p)",
    "level": "Par paliers (type IV, q, p)",
    "linear": "Linéaire avec indifférence (type V, q, p)",
    "gaussian": "Gaussien (type VI, s)",
}


def preference(d: np.ndarray, kind: str = "usual", q: float = 0.0, p: float = 1.0,
               s: float = 1.0) -> np.ndarray:
    """Degré de préférence P(d) ∈ [0, 1] pour un écart d (déjà orienté)."""
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    pos = d > 0
    if kind == "usual":
        out[pos] = 1.0
    elif kind == "u-shape":
        out[d > q] = 1.0
    elif kind == "v-shape":
        p = max(p, 1e-12)
        out = np.where(d <= 0, 0.0, np.where(d >= p, 1.0, d / p))
    elif kind == "level":
        out = np.where(d <= q, 0.0, np.where(d <= p, 0.5, 1.0))
    elif kind == "linear":
        width = max(p - q, 1e-12)
        out = np.where(d <= q, 0.0, np.where(d >= p, 1.0, (d - q) / width))
    elif kind == "gaussian":
        s = max(s, 1e-12)
        out = np.where(d <= 0, 0.0, 1.0 - np.exp(-(d ** 2) / (2 * s ** 2)))
    else:
        raise ValueError(f"Fonction de préférence inconnue : {kind}")
    return out


def promethee(X, w, types, functions=None) -> dict:
    """PROMETHEE II.

    functions : liste (une par critère) de dicts {"kind", "q", "p", "s"} ;
    par défaut fonction usuelle pour tous les critères.
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m, n = X.shape
    b = is_benefit(types)
    if functions is None:
        functions = [{"kind": "usual"}] * n
    Pi = np.zeros((m, m))
    per_criterion = []
    for j in range(n):
        col = X[:, j]
        d = col[:, None] - col[None, :]
        if not b[j]:
            d = -d
        f = functions[j]
        Pj = preference(d, f.get("kind", "usual"), f.get("q", 0.0), f.get("p", 1.0), f.get("s", 1.0))
        np.fill_diagonal(Pj, 0.0)
        per_criterion.append(Pj)
        Pi += w[j] * Pj
    phi_plus = Pi.sum(axis=1) / (m - 1)
    phi_minus = Pi.sum(axis=0) / (m - 1)
    phi = phi_plus - phi_minus
    return {"Pi": Pi, "P_criteria": per_criterion, "phi_plus": phi_plus,
            "phi_minus": phi_minus, "scores": phi, "ranks": rank_desc(phi)}


def promethee_i_relations(phi_plus, phi_minus, names) -> list[tuple[str, str, str]]:
    """Relations du préordre partiel PROMETHEE I : P (surclasse), I (indifférent), R (incomparable)."""
    rel = []
    m = len(names)
    for a in range(m):
        for c in range(a + 1, m):
            dp, dm = phi_plus[a] - phi_plus[c], phi_minus[c] - phi_minus[a]
            if abs(dp) < 1e-12 and abs(dm) < 1e-12:
                rel.append((names[a], "I", names[c]))
            elif dp >= -1e-12 and dm >= -1e-12:
                rel.append((names[a], "P", names[c]))
            elif dp <= 1e-12 and dm <= 1e-12:
                rel.append((names[c], "P", names[a]))
            else:
                rel.append((names[a], "R", names[c]))
    return rel


# ---------------------------------------------------------------------------
# ELECTRE III (Roy, 1978) — formulation standard : pseudo-critères (q, p, v),
# concordance, discordance, crédibilité, distillations descendante et ascendante.
# ---------------------------------------------------------------------------
_ALPHA, _BETA = 0.30, -0.15  # seuil de discrimination s(λ) = α + β·λ


def _distill(S: np.ndarray, idx: list[int], descending: bool) -> list[list[int]]:
    """Une distillation sur le sous-ensemble idx → liste de classes (ordre d'extraction)."""
    remaining = list(idx)
    classes = []
    while remaining:
        D = list(remaining)
        lam0 = None
        while True:
            sub = S[np.ix_(D, D)].copy()
            np.fill_diagonal(sub, 0.0)
            if lam0 is None:
                lam0 = sub.max()
            if lam0 <= 1e-12 or len(D) == 1:
                break
            below = sub[sub < lam0 - (_ALPHA + _BETA * lam0)]
            lam1 = below.max() if below.size else 0.0
            T = (sub > lam1) & ((sub - sub.T) > (_ALPHA + _BETA * sub))
            qual = T.sum(axis=1) - T.sum(axis=0)  # puissance − faiblesse
            target = qual.max() if descending else qual.min()
            D_new = [D[i] for i in range(len(D)) if qual[i] == target]
            if len(D_new) == 1 or lam1 <= 1e-12:
                D = D_new
                break
            D, lam0 = D_new, lam1
        classes.append(D)
        remaining = [a for a in remaining if a not in D]
    return classes


def electre3(X, w, types, q=None, p=None, v=None) -> dict:
    """ELECTRE III.

    q, p, v : seuils d'indifférence, de préférence et de veto par critère (mêmes unités que le
    critère ; v[j] = NaN → pas de veto). Par défaut q = 5 %, p = 25 % de l'étendue, sans veto.
    Renvoie les matrices de concordance et de crédibilité, les deux préordres (distillations),
    les relations du préordre final (P, P⁻, I, R) et un rang final.
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m, n = X.shape
    b = is_benefit(types)
    spread = X.max(axis=0) - X.min(axis=0)
    q = 0.05 * spread if q is None else np.asarray(q, dtype=float)
    p = 0.25 * spread if p is None else np.asarray(p, dtype=float)
    v = np.full(n, np.nan) if v is None else np.asarray(v, dtype=float)
    C = np.zeros((m, m))
    discordances = []
    for j in range(n):
        # diff[a, b] = avantage de b sur a dans le sens de préférence
        diff = X[:, j][None, :] - X[:, j][:, None]
        if not b[j]:
            diff = -diff
        pj = max(p[j], q[j] + 1e-12)
        C += w[j] * np.clip((pj - diff) / (pj - q[j]), 0, 1)
        if np.isnan(v[j]):
            discordances.append(np.zeros((m, m)))
        else:
            vj = max(v[j], pj + 1e-12)
            discordances.append(np.clip((diff - pj) / (vj - pj), 0, 1))
    S = C.copy()
    for Dj in discordances:
        S = np.where(Dj > C, S * (1 - Dj) / np.where(1 - C < 1e-12, 1.0, 1 - C), S)
    np.fill_diagonal(S, 1.0)
    idx = list(range(m))
    desc = _distill(S, idx, True)
    asc = _distill(S, idx, False)[::-1]
    r_desc, r_asc = np.zeros(m, dtype=int), np.zeros(m, dtype=int)
    for r, cls in enumerate(desc, 1):
        r_desc[cls] = r
    for r, cls in enumerate(asc, 1):
        r_asc[cls] = r
    rel = np.full((m, m), "I", dtype=object)
    for a in range(m):
        for c in range(m):
            if a == c:
                rel[a, c] = "·"
                continue
            d1, d2 = r_desc[a] - r_desc[c], r_asc[a] - r_asc[c]
            if d1 == 0 and d2 == 0:
                rel[a, c] = "I"
            elif d1 <= 0 and d2 <= 0:
                rel[a, c] = "P"
            elif d1 >= 0 and d2 >= 0:
                rel[a, c] = "P⁻"
            else:
                rel[a, c] = "R"
    mean_rank = (r_desc + r_asc) / 2.0
    dominated_by = (rel == "P⁻").sum(axis=1)
    ranks = rank_desc(-(dominated_by * 1000 + mean_rank))
    return {"C": C, "S": S, "q": q, "p": p, "v": v, "desc": desc, "asc": asc,
            "rank_desc": r_desc, "rank_asc": r_asc, "relations": rel,
            "scores": -mean_rank, "mean_rank": mean_rank, "ranks": ranks}
