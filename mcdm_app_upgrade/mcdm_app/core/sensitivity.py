"""Analyse de sensibilité / robustesse et comparaison des méthodes
(corrélation de rang de Spearman)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .crisp_ranking import rank_desc


def perturb_weight(w, j: int, new_value: float) -> np.ndarray:
    """Fixe w_j = new_value et redistribue (1 − new_value) sur les autres
    critères proportionnellement à leurs poids initiaux."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    others = np.delete(np.arange(len(w)), j)
    rest = w[others].sum()
    out = np.empty_like(w)
    out[j] = new_value
    if rest > 0:
        out[others] = w[others] / rest * (1 - new_value)
    else:
        out[others] = (1 - new_value) / len(others)
    return out


def one_at_a_time(method: Callable, X, w, types, j: int, grid) -> dict:
    """Fait varier le poids du critère j sur `grid` ; renvoie scores et rangs."""
    scores, ranks, weights = [], [], []
    for v in grid:
        wv = perturb_weight(w, j, float(v))
        res = method(X, wv, types)
        scores.append(res["scores"]); ranks.append(res["ranks"]); weights.append(wv)
    return {"grid": np.asarray(grid), "scores": np.array(scores),
            "ranks": np.array(ranks), "weights": np.array(weights)}


def rank_reversal_points(grid, ranks) -> list[float]:
    """Valeurs du poids pour lesquelles l'alternative classée 1re change."""
    firsts = np.argmin(ranks, axis=1)
    return [float(grid[k]) for k in range(1, len(grid)) if firsts[k] != firsts[k - 1]]


def monte_carlo(method: Callable, X, w, types, n_sims: int = 1000,
                concentration: float = 50.0, seed: int = 0) -> dict:
    """Tire des vecteurs de poids w' ~ Dirichlet(κ·w) autour de w et compte la
    fréquence de chaque rang par alternative."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    alpha = np.maximum(concentration * w, 1e-3)
    m = np.asarray(X).shape[0]
    freq = np.zeros((m, m))
    base = method(X, w, types)["ranks"]
    rhos = []
    for _ in range(n_sims):
        ws = rng.dirichlet(alpha)
        r = method(X, ws, types)["ranks"]
        for i, k in enumerate(r):
            freq[i, min(k, m) - 1] += 1
        rhos.append(spearman(base, r))
    return {"rank_freq": freq / n_sims, "rho": np.array(rhos), "base_ranks": base}


def spearman(r1, r2) -> float:
    """ρ = 1 − 6Σd²/(n(n² − 1)) ; scipy (correction des ex aequo) si rangs liés."""
    r1 = np.asarray(r1, dtype=float); r2 = np.asarray(r2, dtype=float)
    n = len(r1)
    if n < 2:
        return 1.0
    if len(set(r1)) == n and len(set(r2)) == n:
        d2 = ((r1 - r2) ** 2).sum()
        return float(1 - 6 * d2 / (n * (n ** 2 - 1)))
    rho = spearmanr(r1, r2).statistic
    return float(rho) if np.isfinite(rho) else 1.0


def spearman_matrix(rankings: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(rankings)
    M = np.array([[spearman(rankings[a], rankings[b]) for b in names] for a in names])
    return pd.DataFrame(M, index=names, columns=names)


# Valeurs critiques bilatérales de ρ de Spearman à 5 % (petits n)
SPEARMAN_CRITICAL_5 = {4: 1.000, 5: 1.000, 6: 0.886, 7: 0.786, 8: 0.738, 9: 0.700,
                       10: 0.648, 11: 0.618, 12: 0.587, 13: 0.560, 14: 0.538,
                       15: 0.521, 16: 0.503, 17: 0.485, 18: 0.472, 19: 0.460, 20: 0.447}


def borda_consensus(rankings: dict[str, np.ndarray]) -> np.ndarray:
    """Classement de consensus par comptage de Borda (somme des rangs)."""
    total = np.sum([np.asarray(r, dtype=float) for r in rankings.values()], axis=0)
    return rank_desc(-total)
