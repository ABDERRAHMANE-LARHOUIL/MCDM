"""MCDM floue :
- TOPSIS flou à nombres triangulaires (Chen, 2000)
- TOPSIS flou intuitionniste (agrégation IFWA, distance euclidienne normalisée)
"""
from __future__ import annotations

import numpy as np

from .crisp_ranking import is_benefit, rank_desc

# ---------------------------------------------------------------------------
# Échelles linguistiques (Chen, 2000 — tableau 1 du cours)
# ---------------------------------------------------------------------------
CHEN_RATING_SCALE = {
    "VP": (0, 0, 1),     # Very poor
    "P": (0, 1, 3),      # Poor
    "MP": (1, 3, 5),     # Medium poor
    "F": (3, 5, 7),      # Fair
    "MG": (5, 7, 9),     # Medium good
    "G": (7, 9, 10),     # Good
    "VG": (9, 10, 10),   # Very good
}
CHEN_WEIGHT_SCALE = {
    "VL": (0, 0, 0.1),       # Very low
    "L": (0, 0.1, 0.3),      # Low
    "ML": (0.1, 0.3, 0.5),   # Medium low
    "M": (0.3, 0.5, 0.7),    # Medium
    "MH": (0.5, 0.7, 0.9),   # Medium high
    "H": (0.7, 0.9, 1.0),    # High
    "VH": (0.9, 1.0, 1.0),   # Very high
}

# Échelles intuitionnistes usuelles (Boran et al., 2009)
IFN_RATING_SCALE = {
    "EG": (1.00, 0.00),   # Extremely good
    "VVG": (0.90, 0.10),  # Very very good
    "VG": (0.80, 0.10),   # Very good
    "G": (0.70, 0.20),    # Good
    "MG": (0.60, 0.30),   # Medium good
    "F": (0.50, 0.40),    # Fair
    "MB": (0.40, 0.50),   # Medium bad
    "B": (0.25, 0.60),    # Bad
    "VB": (0.10, 0.75),   # Very bad
    "VVB": (0.10, 0.90),  # Very very bad
}
IFN_WEIGHT_SCALE = {
    "VI": (0.90, 0.10),   # Very important
    "I": (0.75, 0.20),    # Important
    "M": (0.50, 0.45),    # Medium
    "U": (0.35, 0.60),    # Unimportant
    "VU": (0.10, 0.90),   # Very unimportant
}


# ---------------------------------------------------------------------------
# Nombres flous triangulaires
# ---------------------------------------------------------------------------
def linguistic_to_tfn(labels, scale) -> np.ndarray:
    """Tableau de libellés (forme quelconque) → tableau (…, 3) de TFN."""
    arr = np.asarray(labels, dtype=object)
    out = np.zeros(arr.shape + (3,))
    for idx, lab in np.ndenumerate(arr):
        out[idx] = scale[str(lab).strip()]
    return out


def aggregate_tfn_mean(tfns) -> np.ndarray:
    """x̃_ij = (1/K)·(x̃_ij¹ ⊕ … ⊕ x̃_ijᴷ) — tfns de forme (K, …, 3)."""
    return np.asarray(tfns, dtype=float).mean(axis=0)


def vertex_distance(a, b) -> np.ndarray:
    """d(ã, b̃) = √[(1/3)·Σ (a_k − b_k)²] (méthode du sommet)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return np.sqrt(((a - b) ** 2).mean(axis=-1))


def centroid(tfn) -> np.ndarray:
    """Défuzzification par centre de gravité : (a + b + c) / 3."""
    return np.asarray(tfn, dtype=float).mean(axis=-1)


def fuzzy_topsis_chen(X, W, types) -> dict:
    """TOPSIS flou de Chen (2000).

    X : (m, n, 3) évaluations collectives ; W : (n, 3) poids collectifs.
    Normalisation : benefit (a/c*, b/c*, c/c*) avec c* = max_i c_ij ;
                    cost (a⁻/c, a⁻/b, a⁻/a) avec a⁻ = min_i a_ij.
    FPIS = (1,1,1), FNIS = (0,0,0) ; distance du sommet ; CC = D⁻/(D⁻ + D*).
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = is_benefit(types)
    m, n, _ = X.shape
    R = np.zeros_like(X)
    for j in range(n):
        if b[j]:
            c_star = X[:, j, 2].max()
            R[:, j, :] = X[:, j, :] / (c_star if c_star else 1.0)
        else:
            a_minus = X[:, j, 0].min()
            rev = X[:, j, ::-1]  # (c, b, a)
            with np.errstate(divide="ignore", invalid="ignore"):
                R[:, j, :] = np.where(rev > 0, a_minus / rev, 1.0)
    V = R * W[None, :, :]
    d_star = vertex_distance(V, np.ones(3))
    d_minus = vertex_distance(V, np.zeros(3))
    D_star = d_star.sum(axis=1)
    D_minus = d_minus.sum(axis=1)
    CC = D_minus / (D_minus + D_star)
    return {"R": R, "V": V, "d_star": d_star, "d_minus": d_minus,
            "D_star": D_star, "D_minus": D_minus, "scores": CC, "ranks": rank_desc(CC)}


# ---------------------------------------------------------------------------
# Nombres flous intuitionnistes
# ---------------------------------------------------------------------------
def ifn_add(a, b):
    """α1 ⊕ α2 = (μ1 + μ2 − μ1μ2, ν1ν2)."""
    return (a[0] + b[0] - a[0] * b[0], a[1] * b[1])


def ifn_mul(a, b):
    """α1 ⊗ α2 = (μ1μ2, ν1 + ν2 − ν1ν2)."""
    return (a[0] * b[0], a[1] + b[1] - a[1] * b[1])


def ifn_score(mu, nu):
    """S(α) = μ − ν ; précision H(α) = μ + ν."""
    return np.asarray(mu) - np.asarray(nu), np.asarray(mu) + np.asarray(nu)


def ifwa(mu, nu, sigma) -> tuple[np.ndarray, np.ndarray]:
    """IFWA_σ(α1, …, αK) = (1 − Π(1 − μ_k)^σk, Π ν_k^σk) — axe 0 = décideurs."""
    mu = np.asarray(mu, dtype=float); nu = np.asarray(nu, dtype=float)
    s = np.asarray(sigma, dtype=float).reshape((-1,) + (1,) * (mu.ndim - 1))
    agg_mu = 1.0 - np.prod((1.0 - mu) ** s, axis=0)
    with np.errstate(divide="ignore"):
        agg_nu = np.prod(np.where(nu > 0, nu, 0.0) ** s, axis=0)
    return agg_mu, agg_nu


def if_topsis(mu, nu, w_mu, w_nu, dm_weights, types) -> dict:
    """TOPSIS flou intuitionniste.

    mu, nu       : (K, m, n) évaluations de chaque décideur
    w_mu, w_nu   : (K, n) importance des critères selon chaque décideur
    dm_weights   : (K,) poids des décideurs σ, Σσ = 1
    """
    sigma = np.asarray(dm_weights, dtype=float)
    sigma = sigma / sigma.sum()
    A_mu, A_nu = ifwa(mu, nu, sigma)                 # étape 3 : (m, n)
    Wj_mu, Wj_nu = ifwa(w_mu, w_nu, sigma)           # étape 4 : (n,)
    P_mu = A_mu * Wj_mu                              # étape 5 : a_ij ⊗ w_j
    P_nu = A_nu + Wj_nu - A_nu * Wj_nu
    P_pi = 1.0 - P_mu - P_nu
    b = is_benefit(types)
    Ip_mu = np.where(b, P_mu.max(axis=0), P_mu.min(axis=0))   # étape 6
    Ip_nu = np.where(b, P_nu.min(axis=0), P_nu.max(axis=0))
    Im_mu = np.where(b, P_mu.min(axis=0), P_mu.max(axis=0))
    Im_nu = np.where(b, P_nu.max(axis=0), P_nu.min(axis=0))
    Ip_pi = 1.0 - Ip_mu - Ip_nu
    Im_pi = 1.0 - Im_mu - Im_nu
    n = P_mu.shape[1]
    S_plus = np.sqrt(((P_mu - Ip_mu) ** 2 + (P_nu - Ip_nu) ** 2 + (P_pi - Ip_pi) ** 2).sum(axis=1) / (2 * n))
    S_minus = np.sqrt(((P_mu - Im_mu) ** 2 + (P_nu - Im_nu) ** 2 + (P_pi - Im_pi) ** 2).sum(axis=1) / (2 * n))
    denom = S_plus + S_minus
    C = np.divide(S_minus, denom, out=np.zeros_like(S_minus), where=denom != 0)
    return {"A_mu": A_mu, "A_nu": A_nu, "W_mu": Wj_mu, "W_nu": Wj_nu,
            "P_mu": P_mu, "P_nu": P_nu, "P_pi": P_pi,
            "I_plus": (Ip_mu, Ip_nu), "I_minus": (Im_mu, Im_nu),
            "S_plus": S_plus, "S_minus": S_minus, "scores": C, "ranks": rank_desc(C)}
