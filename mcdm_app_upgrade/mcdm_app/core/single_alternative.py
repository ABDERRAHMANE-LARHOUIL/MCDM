"""Techniques MCDM pour évaluer une alternative isolée (évaluation absolue) :
MAUT, seuil de décision, courbe ROC / indice de Youden, distance à l'idéal,
évaluation floue d'une seule alternative.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# MAUT
# ---------------------------------------------------------------------------
def utilities(x, x_min, x_max, types) -> np.ndarray:
    """Utilités linéaires du cours :
    benefit u = (x − x_min) / (x_max − x_min) ; cost u = (x − x_max) / (x_min − x_max).
    Les valeurs sont bornées à [0, 1].
    """
    x, lo, hi = (np.asarray(a, dtype=float) for a in (x, x_min, x_max))
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    b = np.array([t == "+" for t in types])
    u = np.where(b, (x - lo) / span, (x - hi) / (-span))
    return np.clip(u, 0.0, 1.0)


def maut(x, w, x_min, x_max, types) -> dict:
    u = utilities(x, x_min, x_max, types)
    w = np.asarray(w, dtype=float)
    contrib = w * u
    return {"utilities": u, "contributions": contrib, "score": float(contrib.sum())}


def decide(score: float, tau: float) -> str:
    return "Accepter" if score >= tau else "Rejeter"


def classify(score: float, bounds: dict[str, float]) -> str:
    """Classe (A, B, C…) : bounds = {"A": 0.8, "B": 0.6, "C": 0.0} (seuils bas)."""
    for label, lo in sorted(bounds.items(), key=lambda kv: -kv[1]):
        if score >= lo:
            return label
    return list(bounds)[-1]


# ---------------------------------------------------------------------------
# Seuil issu des données : ROC, Youden, distance au point idéal, coûts
# ---------------------------------------------------------------------------
def confusion_at(scores, labels, tau: float) -> dict:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    yhat = (s >= tau).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tau": tau, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "TPR": tpr, "FPR": fpr,
            "d": float(np.hypot(fpr, 1 - tpr)), "J": tpr - fpr,
            "accuracy": (tp + tn) / len(y) if len(y) else 0.0}


def roc_table(scores, labels, thresholds=None, cost_fp: float = 1.0, cost_fn: float = 1.0) -> list[dict]:
    """Évalue chaque seuil τ : TPR, FPR, d = √(FPR² + (1 − TPR)²), J = TPR − FPR,
    exactitude et coût attendu C = c_FP·FP + c_FN·FN.

    Par défaut, les seuils candidats sont les scores observés (plus 0 et 1.01).
    """
    s = np.asarray(scores, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.concatenate([s, [0.0, s.max() + 0.01]]))[::-1]
    rows = []
    for t in thresholds:
        r = confusion_at(s, labels, float(t))
        r["cost"] = cost_fp * r["FP"] + cost_fn * r["FN"]
        rows.append(r)
    return rows


def auc(rows: list[dict]) -> float:
    pts = sorted({(r["FPR"], r["TPR"]) for r in rows} | {(0.0, 0.0), (1.0, 1.0)})
    fpr = np.array([p[0] for p in pts]); tpr = np.array([p[1] for p in pts])
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def best_thresholds(rows: list[dict]) -> dict:
    """Seuils optimaux selon chaque critère (en cas d'égalité, le premier — le plus élevé)."""
    return {
        "youden": max(rows, key=lambda r: r["J"]),
        "distance": min(rows, key=lambda r: r["d"]),
        "accuracy": max(rows, key=lambda r: r["accuracy"]),
        "cost": min(rows, key=lambda r: r["cost"]),
    }


# ---------------------------------------------------------------------------
# Distance à l'idéal (logique TOPSIS pour une seule alternative)
# ---------------------------------------------------------------------------
def distance_to_ideal(x, ideal, anti_ideal, w=None, normalized: bool = False, types=None) -> dict:
    """C = D− / (D+ + D−) avec D+ = √Σ(x_j − x_j+)², D− = √Σ(x_j − x_j−)².

    normalized=True : les valeurs sont d'abord ramenées à des utilités [0, 1]
    (idéal → 1, anti-idéal → 0) et pondérées par w, ce qui évite que les
    critères de grande amplitude dominent la distance.
    """
    x, ip, im = (np.asarray(a, dtype=float) for a in (x, ideal, anti_ideal))
    if normalized:
        span = np.where(ip - im == 0, 1.0, ip - im)
        u = np.clip((x - im) / span, 0, 1)
        wv = np.ones_like(u) if w is None else np.asarray(w, dtype=float)
        d_plus = float(np.sqrt((wv * (1 - u) ** 2).sum()))
        d_minus = float(np.sqrt((wv * u ** 2).sum()))
    else:
        d_plus = float(np.sqrt(((x - ip) ** 2).sum()))
        d_minus = float(np.sqrt(((x - im) ** 2).sum()))
    c = d_minus / (d_plus + d_minus) if d_plus + d_minus > 0 else 0.0
    return {"D_plus": d_plus, "D_minus": d_minus, "C": c}


# ---------------------------------------------------------------------------
# Évaluation floue d'une seule alternative (nombres triangulaires)
# ---------------------------------------------------------------------------
def fuzzy_single(tfns, w, c_max=None) -> dict:
    """1) normalisation (a, b, c) ÷ c_max ; 2) S̃ = Σ w_j·x̃_j ; 3) S = (a + b + c)/3.

    c_max : par critère, le plus grand c parmi les alternatives (par défaut
    celui de l'alternative évaluée).
    """
    T = np.asarray(tfns, dtype=float)          # n × 3
    w = np.asarray(w, dtype=float)
    cm = T[:, 2] if c_max is None else np.asarray(c_max, dtype=float)
    R = T / cm[:, None]
    S = (w[:, None] * R).sum(axis=0)
    return {"R": R, "S_fuzzy": S, "S": float(S.mean()), "width": float(S[2] - S[0])}
