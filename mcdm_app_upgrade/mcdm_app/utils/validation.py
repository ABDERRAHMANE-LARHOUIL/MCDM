"""Contrôles de cohérence : dimensions, valeurs, poids, matrices AHP, nombres flous."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_decision_matrix(X: pd.DataFrame, types: list[str]) -> tuple[list[str], list[str]]:
    """Renvoie (erreurs bloquantes, avertissements)."""
    errors, warnings = [], []
    if X.shape[0] < 2:
        errors.append("Il faut au moins 2 alternatives pour un classement.")
    if X.shape[1] < 1:
        errors.append("Il faut au moins 1 critère.")
    if len(types) != X.shape[1]:
        errors.append("Le nombre de types (+/−) ne correspond pas au nombre de critères.")
    vals = X.apply(pd.to_numeric, errors="coerce")
    if vals.isna().any().any():
        bad = [f"{r} / {c}" for r, c in zip(*np.where(vals.isna()))]
        errors.append(f"Valeurs manquantes ou non numériques : {len(bad)} cellule(s).")
        return errors, warnings
    arr = vals.to_numpy(dtype=float)
    if (arr <= 0).any():
        warnings.append("Valeurs ≤ 0 détectées : la normalisation linéaire du cours (x/max, min/x) et l'entropie "
                        "exigent des valeurs positives. WSM, WPM et WASPAS basculent sur la normalisation min–max ; "
                        "TOPSIS, VIKOR, PROMETHEE et ELECTRE III ne sont pas concernées.")
    const = [c for c, col in zip(X.columns, arr.T) if np.ptp(col) == 0]
    if const:
        warnings.append("Critère(s) constant(s) (aucun pouvoir discriminant) : " + ", ".join(map(str, const)))
    if len(set(X.columns)) != len(X.columns):
        errors.append("Les noms de critères doivent être uniques.")
    if len(set(X.index)) != len(X.index):
        errors.append("Les noms d'alternatives doivent être uniques.")
    return errors, warnings


def normalize_weights(w) -> tuple[np.ndarray, str | None]:
    w = np.asarray(w, dtype=float)
    if (w < 0).any():
        raise ValueError("Les poids doivent être positifs ou nuls.")
    s = w.sum()
    if s <= 0:
        raise ValueError("La somme des poids doit être strictement positive.")
    msg = None if abs(s - 1) < 1e-6 else f"Somme des poids = {s:.4f} → renormalisée à 1."
    return w / s, msg


def check_ahp_matrix(A) -> list[str]:
    A = np.asarray(A, dtype=float)
    issues = []
    if A.shape[0] != A.shape[1]:
        return ["La matrice de comparaison doit être carrée."]
    if not np.allclose(np.diag(A), 1):
        issues.append("La diagonale doit valoir 1.")
    if (A <= 0).any():
        issues.append("Tous les jugements doivent être strictement positifs.")
    elif not np.allclose(A * A.T, 1, atol=1e-2):
        issues.append("Réciprocité non respectée : a_ji doit valoir 1/a_ij.")
    if (A > 9 + 1e-9).any() or (A < 1 / 9 - 1e-9).any():
        issues.append("Les jugements doivent rester dans l'échelle de Saaty [1/9, 9].")
    return issues


def check_tfn(T) -> list[str]:
    T = np.asarray(T, dtype=float)
    if not ((T[..., 0] <= T[..., 1]) & (T[..., 1] <= T[..., 2])).all():
        return ["Chaque nombre flou triangulaire doit vérifier a ≤ b ≤ c."]
    return []


def check_ifn(mu, nu) -> list[str]:
    mu, nu = np.asarray(mu, dtype=float), np.asarray(nu, dtype=float)
    issues = []
    if ((mu < 0) | (mu > 1) | (nu < 0) | (nu > 1)).any():
        issues.append("μ et ν doivent appartenir à [0, 1].")
    if (mu + nu > 1 + 1e-9).any():
        issues.append("Chaque nombre intuitionniste doit vérifier μ + ν ≤ 1.")
    return issues
