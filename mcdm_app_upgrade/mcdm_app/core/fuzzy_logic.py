"""Logique floue (chapitre « Introduction à la logique floue » du cours) :
fonctions d'appartenance, coupe-α, arithmétique des nombres flous et des intervalles,
nombres flous intuitionnistes, pythagoriciens, q-Rung Orthopair, picturaux et hésitants.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Fonctions d'appartenance
# ---------------------------------------------------------------------------
def mu_trapezoid(x, a, b, c, d) -> np.ndarray:
    """μ(x) trapézoïdale N(a, b, c, d) ; b = c donne un nombre triangulaire."""
    x = np.asarray(x, dtype=float)
    up = (x >= a).astype(float) if b - a < EPS else (x - a) / (b - a)
    down = (x <= d).astype(float) if d - c < EPS else (d - x) / (d - c)
    return np.clip(np.minimum(np.minimum(up, 1.0), down), 0.0, 1.0)


def mu_triangle(x, a, b, c) -> np.ndarray:
    return mu_trapezoid(x, a, b, b, c)


def mu_gauss(x, m, sigma) -> np.ndarray:
    return np.exp(-((np.asarray(x, dtype=float) - m) ** 2) / (2 * max(sigma, EPS) ** 2))


def alpha_cut_trapezoid(a, b, c, d, alpha: float) -> tuple[float, float]:
    """Coupe-α {x | μ(x) ≥ α} d'un nombre trapézoïdal : intervalle [gauche, droite]."""
    return a + alpha * (b - a), d - alpha * (d - c)


def alpha_cut_gauss(m, sigma, alpha: float) -> tuple[float, float]:
    half = sigma * np.sqrt(-2 * np.log(max(alpha, EPS)))
    return m - half, m + half


# ---------------------------------------------------------------------------
# Arithmétique des nombres flous (a1, a2, a3, a4) positifs et des intervalles
# ---------------------------------------------------------------------------
def fz_add(A, B):
    return tuple(float(x) for x in np.add(A, B))


def fz_sub(A, B):
    return tuple(float(x) for x in np.subtract(A, B[::-1]))


def fz_mul(A, B):
    return tuple(float(x) for x in np.multiply(A, B))


def fz_div(A, B):
    return tuple(float(x) for x in np.divide(A, B[::-1]))


def fz_scale(lam: float, A):
    return tuple(float(lam * x) for x in A)


def interval_ops(A, B) -> dict:
    """Opérations sur deux intervalles A = [a1, a2], B = [b1, b2]."""
    a1, a2 = A
    b1, b2 = B
    prods = [a1 * b1, a1 * b2, a2 * b1, a2 * b2]
    out = {"A + B": (a1 + b1, a2 + b2), "A − B": (a1 - b2, a2 - b1), "A × B": (min(prods), max(prods))}
    if not (b1 <= 0 <= b2):
        out["A ÷ B"] = (a1 / b2, a2 / b1)
    out["centre m(A)"] = ((a1 + a2) / 2,)
    out["largeur w(A)"] = ((a2 - a1) / 2,)
    return out


# ---------------------------------------------------------------------------
# Nombres flous intuitionnistes (q = 1), pythagoriciens (q = 2), q-ROF
# ---------------------------------------------------------------------------
def orthopair_hesitation(mu: float, nu: float, q: float = 1.0):
    """π = (1 − μ^q − ν^q)^(1/q). Renvoie None si la contrainte μ^q + ν^q ≤ 1 est violée."""
    t = mu ** q + nu ** q
    if t > 1 + 1e-12:
        return None
    return float(max(1 - t, 0.0) ** (1.0 / q))


def min_valid_q(mu: float, nu: float, q_max: int = 50):
    """Plus petit entier q ≥ 1 tel que μ^q + ν^q ≤ 1 (None si μ = ν = 1)."""
    for q in range(1, q_max + 1):
        if mu ** q + nu ** q <= 1 + 1e-12:
            return q
    return None


def interval_qrof_hesitation(mu_l, mu_u, nu_l, nu_u, q: float = 1.0):
    """Degré d'hésitation [π_L, π_U] d'un nombre q-ROF à valeurs d'intervalle."""
    if mu_u ** q + nu_u ** q > 1 + 1e-12:
        return None
    lo = max(1 - mu_u ** q - nu_u ** q, 0.0) ** (1.0 / q)
    hi = max(1 - mu_l ** q - nu_l ** q, 0.0) ** (1.0 / q)
    return float(lo), float(hi)


def ifn_add(a, b):
    return (a[0] + b[0] - a[0] * b[0], a[1] * b[1])


def ifn_mul(a, b):
    return (a[0] * b[0], a[1] + b[1] - a[1] * b[1])


def ifn_scale(lam: float, a):
    return (1 - (1 - a[0]) ** lam, a[1] ** lam)


def ifn_pow(a, lam: float):
    return (a[0] ** lam, 1 - (1 - a[1]) ** lam)


def ifn_score(a) -> float:
    return a[0] - a[1]


def ifn_accuracy(a) -> float:
    return a[0] + a[1]


def ifn_compare(a, b) -> int:
    """+1 si a > b, −1 si a < b, 0 si a = b (score S puis précision H)."""
    sa, sb = ifn_score(a), ifn_score(b)
    if not np.isclose(sa, sb):
        return 1 if sa > sb else -1
    ha, hb = ifn_accuracy(a), ifn_accuracy(b)
    if np.isclose(ha, hb):
        return 0
    return 1 if ha > hb else -1


def ifwa_list(pairs, sigma):
    """IFWA de k nombres (μ_j, ν_j) pondérés par σ : (1 − Π(1 − μ_j)^σ_j , Π ν_j^σ_j)."""
    P = np.asarray(pairs, dtype=float)
    s = np.asarray(sigma, dtype=float)
    s = s / s.sum()
    return float(1 - np.prod((1 - P[:, 0]) ** s)), float(np.prod(P[:, 1] ** s))


# ---------------------------------------------------------------------------
# Nombres flous picturaux (μ positif, ν négatif, π neutre)
# ---------------------------------------------------------------------------
def picture_refusal(a):
    """Degré de refus 1 − (μ + ν + π) ; None si μ + ν + π > 1."""
    t = a[0] + a[1] + a[2]
    return None if t > 1 + 1e-12 else float(1 - t)


def picture_add(a, b):
    return (a[0] + b[0] - a[0] * b[0], a[1] * b[1], a[2] * b[2])


def picture_scale(lam: float, a):
    return (1 - (1 - a[0]) ** lam, a[1] ** lam, a[2] ** lam)


def picture_pow(a, lam: float):
    return (a[0] ** lam, 1 - (1 - a[1]) ** lam, 1 - (1 - a[2]) ** lam)


# ---------------------------------------------------------------------------
# Nombres flous hésitants
# ---------------------------------------------------------------------------
def hfe_score(h) -> float:
    return float(np.mean(h))


def hfe_deviation(h) -> float:
    h = np.asarray(h, dtype=float)
    return float(np.sqrt(np.mean((h - h.mean()) ** 2)))


def hfe_compare(h1, h2) -> int:
    """+1 si h1 > h2, −1 si h1 < h2, 0 si égaux (score, puis déviation la plus faible)."""
    s1, s2 = hfe_score(h1), hfe_score(h2)
    if not np.isclose(s1, s2):
        return 1 if s1 > s2 else -1
    d1, d2 = hfe_deviation(h1), hfe_deviation(h2)
    if np.isclose(d1, d2):
        return 0
    return 1 if d1 < d2 else -1


def hfe_extend(h, length: int, optimistic: bool = False) -> list[float]:
    """Complète h jusqu'à `length` éléments (valeur min : pessimiste, max : optimiste), trié."""
    h = sorted(float(x) for x in h)
    fill = max(h) if optimistic else min(h)
    return sorted(h + [fill] * (length - len(h)))


def hfe_hamming(h1, h2, optimistic: bool = False) -> float:
    """Distance de Hamming hésitante : (1/l) Σ |h1^σ(k) − h2^σ(k)|, l = max(l(h1), l(h2))."""
    length = max(len(h1), len(h2))
    a, b = hfe_extend(h1, length, optimistic), hfe_extend(h2, length, optimistic)
    return float(np.mean(np.abs(np.subtract(a, b))))


def hfe_ops(h1, h2, alpha: float = 2.0) -> dict:
    def u(values):
        return sorted({round(float(x), 6) for x in values})
    return {
        f"h1^{alpha:g}": u(g ** alpha for g in h1),
        f"{alpha:g}·h1": u(1 - (1 - g) ** alpha for g in h1),
        "h1 ⊕ h2": u(a + b - a * b for a in h1 for b in h2),
        "h1 ⊗ h2": u(a * b for a in h1 for b in h2),
    }
