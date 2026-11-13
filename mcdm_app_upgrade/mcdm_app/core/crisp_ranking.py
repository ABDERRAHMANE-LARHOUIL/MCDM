"""Classement des alternatives — méthodes additives / compensatoires du cours :
WSM, WPM, WASPAS, TOPSIS, VIKOR.

Convention : `types` est une liste de "+" (critère positif / benefit) ou
"−"/"-" (critère négatif / cost). Les rangs vont de 1 (meilleure) à m.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def is_benefit(types) -> np.ndarray:
    return np.array([t == "+" for t in types])


def rank_desc(scores) -> np.ndarray:
    """Rang 1 = score le plus élevé (ex aequo : rang minimal)."""
    return rankdata(-np.asarray(scores, dtype=float), method="min").astype(int)


def rank_asc(scores) -> np.ndarray:
    """Rang 1 = score le plus faible (VIKOR)."""
    return rankdata(np.asarray(scores, dtype=float), method="min").astype(int)


def normalize_linear(X, types) -> np.ndarray:
    """Normalisation linéaire du cours (méthodes de pondération simple) :
    critère positif r_ij = x_ij / max_i x_ij ; critère négatif r_ij = min_i x_ij / x_ij.
    """
    X = np.asarray(X, dtype=float)
    b = is_benefit(types)
    if (X[:, ~b] <= 0).any():
        raise ValueError("La normalisation min/x exige des valeurs strictement positives pour les critères négatifs.")
    if (X[:, b] < 0).any() or (X[:, b].max(axis=0) <= 0).any():
        raise ValueError("La normalisation x/max exige des valeurs positives pour les critères positifs "
                         "(données négatives : utilisez TOPSIS, VIKOR, PROMETHEE II ou ELECTRE III).")
    return np.where(b, X / X.max(axis=0), X.min(axis=0) / X)


def normalize_minmax(X, types) -> np.ndarray:
    """Normalisation min–max (celle de CRITIC) : (x − min)/(max − min) pour un critère positif,
    (max − x)/(max − min) pour un critère négatif. Accepte les valeurs négatives ou nulles."""
    X = np.asarray(X, dtype=float)
    b = is_benefit(types)
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    R = np.where(b, (X - lo) / rng, (hi - X) / rng)
    R[:, hi - lo == 0] = 1.0
    return R


def _normalize(X, types, norm: str) -> np.ndarray:
    return normalize_minmax(X, types) if norm == "minmax" else normalize_linear(X, types)


def ideal_solutions(X, types):
    """Solutions idéales positive A+ et négative A− (valeurs brutes)."""
    X = np.asarray(X, dtype=float)
    b = is_benefit(types)
    a_plus = np.where(b, X.max(axis=0), X.min(axis=0))
    a_minus = np.where(b, X.min(axis=0), X.max(axis=0))
    return a_plus, a_minus


# ---------------------------------------------------------------------------
# WSM / WPM / WASPAS
# ---------------------------------------------------------------------------
def wsm(X, w, types, norm: str = "linear") -> dict:
    R = _normalize(X, types, norm)
    q = R @ np.asarray(w, dtype=float)
    return {"R": R, "scores": q, "ranks": rank_desc(q)}


def wpm(X, w, types, norm: str = "linear") -> dict:
    R = _normalize(X, types, norm)
    q = np.prod(R ** np.asarray(w, dtype=float), axis=1)
    return {"R": R, "scores": q, "ranks": rank_desc(q)}


def waspas(X, w, types, lam: float = 0.5, norm: str = "linear") -> dict:
    """WASPAS (Zavadskas et al., 2012) : Q_i = λ·Q_i(1) + (1 − λ)·Q_i(2)."""
    R = _normalize(X, types, norm)
    w = np.asarray(w, dtype=float)
    q1 = R @ w
    q2 = np.prod(R ** w, axis=1)
    q = lam * q1 + (1 - lam) * q2
    return {"R": R, "Q1": q1, "Q2": q2, "scores": q, "ranks": rank_desc(q), "lambda": lam}


# ---------------------------------------------------------------------------
# TOPSIS (Hwang & Yoon, 1981)
# ---------------------------------------------------------------------------
def topsis(X, w, types) -> dict:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1.0, norms)
    R = X / norms                                   # Étape 1
    V = R * w                                       # Étape 2
    b = is_benefit(types)
    v_plus = np.where(b, V.max(axis=0), V.min(axis=0))   # Étape 3
    v_minus = np.where(b, V.min(axis=0), V.max(axis=0))
    s_plus = np.sqrt(((V - v_plus) ** 2).sum(axis=1))    # Étape 4
    s_minus = np.sqrt(((V - v_minus) ** 2).sum(axis=1))
    denom = s_plus + s_minus
    rc = np.divide(s_minus, denom, out=np.zeros_like(s_minus), where=denom != 0)  # Étape 5
    return {"R": R, "V": V, "I_plus": v_plus, "I_minus": v_minus,
            "S_plus": s_plus, "S_minus": s_minus, "scores": rc, "ranks": rank_desc(rc)}


# ---------------------------------------------------------------------------
# VIKOR (Opricovic)
# ---------------------------------------------------------------------------
def vikor(X, w, types, v: float = 0.5) -> dict:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    m = X.shape[0]
    x_best, x_worst = ideal_solutions(X, types)            # Étape 1
    span = x_best - x_worst
    gap = np.divide(x_best - X, span, out=np.zeros_like(X), where=span != 0)
    weighted = w * gap
    S = weighted.sum(axis=1)                               # Étape 2
    R = weighted.max(axis=1)
    s_star, s_minus = S.min(), S.max()
    r_star, r_minus = R.min(), R.max()
    qs = (S - s_star) / (s_minus - s_star) if s_minus != s_star else np.zeros(m)
    qr = (R - r_star) / (r_minus - r_star) if r_minus != r_star else np.zeros(m)
    Q = v * qs + (1 - v) * qr                              # Étape 3

    # Étapes 4–5 : solution de compromis
    order = np.argsort(Q, kind="stable")
    a1, a2 = order[0], order[1] if m > 1 else order[0]
    dq = 1.0 / (m - 1) if m > 1 else 0.0
    c1 = bool(Q[a2] - Q[a1] >= dq - 1e-12)
    c2 = bool(S[a1] == S.min() and R[a1] == R.min())
    if c1 and c2:
        compromise = [int(a1)]
    elif c1 and not c2:
        compromise = [int(a1), int(a2)]
    else:
        compromise = [int(k) for k in order if Q[k] - Q[a1] < dq - 1e-12]
    return {"x_best": x_best, "x_worst": x_worst, "gap": gap, "S": S, "R": R, "Q": Q,
            "rank_S": rank_asc(S), "rank_R": rank_asc(R), "ranks": rank_asc(Q),
            "scores": 1 - Q,  # score « plus grand = meilleur » pour les comparaisons
            "DQ": dq, "C1_advantage": c1, "C2_stability": c2, "compromise": compromise}
