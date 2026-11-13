"""MCDM et IA (chapitre « Techniques MCDM et IA » du cours).

Scénario 1 : MCDM → étiquettes → classification supervisée.
Scénario 2 : XAI comme approche de pondération (valeurs de Shapley, mise à jour des poids).
MCDM dynamique : prévision des critères à t+1 puis classement.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

EPS = 1e-12


def label_by_quantiles(scores, labels=("A", "B", "C", "D")) -> np.ndarray:
    """Étiquette les alternatives par quantiles du score MCDM (première étiquette = meilleure classe)."""
    s = np.asarray(scores, dtype=float)
    k = len(labels)
    cuts = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    idx = k - 1 - np.searchsorted(cuts, s, side="right")
    return np.array(labels)[idx]


def augment(X, n_new: int, noise: float = 0.15, seed: int = 0) -> np.ndarray:
    """Alternatives synthétiques : rééchantillonnage des lignes + bruit gaussien (en % de l'écart-type),
    borné à l'étendue observée. Sert à constituer une base d'apprentissage quand m est petit."""
    X = np.asarray(X, dtype=float)
    if n_new <= 0:
        return np.empty((0, X.shape[1]))
    rng = np.random.default_rng(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    # mélange convexe de deux alternatives tirées au hasard, puis bruit
    i, j = rng.integers(0, len(X), n_new), rng.integers(0, len(X), n_new)
    t = rng.random((n_new, X.shape[1]))
    Z = X[i] * t + X[j] * (1 - t) + rng.normal(0, 1, (n_new, X.shape[1])) * X.std(axis=0) * noise
    return np.clip(Z, lo, hi)


def shapley_values(predict, X, background=None, max_exact: int = 10, n_perm: int = 80, seed: int = 0):
    """Valeurs de Shapley interventionnelles (équivalent SHAP) : Ŝ(x) = φ₀ + Σ φⱼ.

    Calcul exact par énumération des 2ⁿ coalitions si n ≤ max_exact, sinon estimation par
    permutations aléatoires. Les variables hors coalition prennent les valeurs du jeu de référence.
    Renvoie (φ₀ de forme (m,), φ de forme (m, n)).
    """
    X = np.asarray(X, dtype=float)
    B = X if background is None else np.asarray(background, dtype=float)
    m, n = X.shape
    nb = len(B)

    def value(mask):
        Z = np.repeat(B[None, :, :], m, axis=0)
        Z[:, :, mask] = X[:, None, mask]
        return np.asarray(predict(Z.reshape(m * nb, n))).reshape(m, nb).mean(axis=1)

    phi = np.zeros((m, n))
    phi0 = value(np.zeros(n, dtype=bool))
    feats = list(range(n))
    if n <= max_exact:
        cache = {}
        for r in range(n + 1):
            for S in combinations(feats, r):
                mask = np.zeros(n, dtype=bool)
                mask[list(S)] = True
                cache[S] = value(mask)
        for j in feats:
            rest = [f for f in feats if f != j]
            for r in range(n):
                wgt = factorial(r) * factorial(n - r - 1) / factorial(n)
                for S in combinations(rest, r):
                    phi[:, j] += wgt * (cache[tuple(sorted(S + (j,)))] - cache[S])
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_perm):
            mask = np.zeros(n, dtype=bool)
            prev = phi0
            for j in rng.permutation(n):
                mask[j] = True
                cur = value(mask)
                phi[:, j] += cur - prev
                prev = cur
        phi /= n_perm
    return phi0, phi


def shap_weight_update(w, phi, alpha: float = 0.5) -> dict:
    """w_new = α·w + (1 − α)·|φⱼ| / Σ|φⱼ|, avec |φⱼ| = importance globale (moyenne des |φ| locaux)."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    imp = np.abs(np.asarray(phi, dtype=float)).mean(axis=0)
    imp_n = imp / imp.sum() if imp.sum() > EPS else np.full(len(imp), 1.0 / len(imp))
    new = alpha * w + (1 - alpha) * imp_n
    return {"importance": imp, "importance_norm": imp_n, "weights": new / new.sum()}


FORECASTERS = {
    "naive": "Naïf (dernière valeur)",
    "ma": "Moyenne mobile",
    "ses": "Lissage exponentiel simple",
    "trend": "Tendance linéaire",
    "ar1": "AR(1)",
}


def forecast_next(y, model: str = "ses", window: int = 5, alpha: float = 0.4) -> float:
    """Prévision x̂(t+1) d'une série par un modèle simple (le cours cite ARIMA, LSTM, GARCH… :
    tout modèle de prévision peut être substitué)."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return float("nan")
    if model == "naive" or len(y) < 3:
        return float(y[-1])
    if model == "ma":
        return float(y[-window:].mean())
    if model == "ses":
        level = y[0]
        for v in y[1:]:
            level = alpha * v + (1 - alpha) * level
        return float(level)
    if model == "trend":
        slope, intercept = np.polyfit(np.arange(len(y)), y, 1)
        return float(intercept + slope * len(y))
    if model == "ar1":
        x0, x1 = y[:-1], y[1:]
        var = ((x0 - x0.mean()) ** 2).sum()
        phi = 0.0 if var < EPS else float(((x0 - x0.mean()) * (x1 - x1.mean())).sum() / var)
        phi = float(np.clip(phi, -0.99, 0.99))
        return float(x1.mean() - phi * x0.mean() + phi * y[-1])
    raise ValueError(f"Modèle de prévision inconnu : {model}")


def forecast_matrix(series: dict, model: str = "ses", upto: int | None = None, **kw) -> np.ndarray:
    """series[alternative][critère] = liste de valeurs ; renvoie la matrice de décision prévue (m × n)."""
    return np.array([[forecast_next(np.asarray(v)[:upto], model, **kw) for v in crits.values()]
                     for crits in series.values()])


def demo_stock_series(T: int = 60, seed: int = 7) -> dict:
    """Séries synthétiques pour l'exemple du cours (3 actions × rendement, volatilité, liquidité).
    Renvoie {"t": [...], "series": {alternative: {critère: ndarray}}}."""
    rng = np.random.default_rng(seed)
    spec = {"A1 Tech": (8.0, 0.25, 100.0, 0.03), "A2 Énergie": (6.0, 0.15, 80.0, -0.01),
            "A3 Bancaire": (7.0, 0.20, 120.0, 0.01)}
    out = {}
    for name, (r, s, liq, drift) in spec.items():
        rr, ss_, ll = r - drift * T, s, liq
        R, S, L = [], [], []
        for t in range(1, T + 1):
            rr += drift + 0.35 * (r - drift * (T - t) - rr) + rng.normal(0, 0.45)
            ss_ = max(0.05, ss_ + 0.3 * (s - ss_) + rng.normal(0, 0.012))
            ll = max(10.0, ll + 0.25 * (liq - ll) + rng.normal(0, 5))
            R.append(rr); S.append(ss_); L.append(ll)
        out[name] = {"Rendement r (%)": np.array(R), "Volatilité σ": np.array(S), "Liquidité L": np.array(L)}
    return {"t": list(range(1, T + 1)), "series": out}
