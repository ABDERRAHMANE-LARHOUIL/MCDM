"""Pondération des critères — méthodes subjectives (AHP, BWM, SWARA), objectives
(Entropie, CRITIC) et hybrides, selon les formulations du cours
« Aide à la décision » (Prof. Lamrani Alaoui, EMI).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

# Indice de consistance aléatoire (table du cours pour n = 3..10, Saaty au-delà)
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
            8: 1.41, 9: 1.45, 10: 1.56, 11: 1.51, 12: 1.48, 13: 1.56,
            14: 1.57, 15: 1.59}


# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------
def ahp(A, method: str = "approx") -> dict:
    """Poids AHP à partir d'une matrice de comparaisons par paires.

    method="approx" : méthode approximative du cours (somme des colonnes,
    normalisation, moyenne des lignes, λmax = Σ s_i·w_i).
    method="eigen"  : vecteur propre principal exact (A·W = λmax·W).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    col_sums = A.sum(axis=0)
    normalized = A / col_sums
    if method == "eigen":
        vals, vecs = np.linalg.eig(A)
        k = int(np.argmax(vals.real))
        w = np.abs(vecs[:, k].real)
        w = w / w.sum()
        lambda_max = float(vals[k].real)
    else:
        w = normalized.mean(axis=1)
        lambda_max = float(col_sums @ w)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.59)
    cr = ci / ri if ri > 0 else 0.0
    return {
        "weights": w,
        "col_sums": col_sums,
        "normalized": normalized,
        "lambda_max": lambda_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "consistent": cr < 0.1,
    }


def ahp_rank(criteria_matrix, alternative_matrices, method: str = "approx") -> dict:
    """AHP complet : poids des critères + priorités locales → priorités finales.

    alternative_matrices : liste (une par critère) de matrices m×m de
    comparaison des alternatives.
    """
    crit = ahp(criteria_matrix, method)
    locals_ = [ahp(M, method) for M in alternative_matrices]
    L = np.column_stack([r["weights"] for r in locals_])  # m × n
    scores = L @ crit["weights"]
    return {"criteria": crit, "local": locals_, "local_matrix": L, "scores": scores}


def ahp_reciprocal_from_upper(upper: np.ndarray) -> np.ndarray:
    """Complète une matrice AHP à partir de son triangle supérieur (a_ji = 1/a_ij)."""
    A = np.array(upper, dtype=float)
    n = A.shape[0]
    for i in range(n):
        A[i, i] = 1.0
        for j in range(i + 1, n):
            A[j, i] = 1.0 / A[i, j]
    return A


# ---------------------------------------------------------------------------
# BWM (Rezaei) — modèle linéaire du cours
# ---------------------------------------------------------------------------
def bwm(best: int, worst: int, a_BO, a_OW) -> dict:
    """Best-Worst Method, forme linéaire :

        min ξ  s.c.  |w_B − a_Bj·w_j| ≤ ξ,  |w_j − a_jW·w_W| ≤ ξ,
                     Σ w_j = 1,  w_j ≥ 0,  ξ ≥ 0

    Variables x = (w_1, …, w_n, ξ). Résolu avec scipy.optimize.linprog.
    """
    a_BO = np.asarray(a_BO, dtype=float)
    a_OW = np.asarray(a_OW, dtype=float)
    n = len(a_BO)
    c = np.zeros(n + 1)
    c[-1] = 1.0

    rows, labels = [], []
    for j in range(n):
        # w_B − a_Bj w_j − ξ ≤ 0   et   −w_B + a_Bj w_j − ξ ≤ 0
        r = np.zeros(n + 1); r[best] += 1; r[j] -= a_BO[j]; r[-1] = -1
        rows.append(r); labels.append(f"w_B − {a_BO[j]:g}·w_{j+1} − ξ ≤ 0")
        r = np.zeros(n + 1); r[best] -= 1; r[j] += a_BO[j]; r[-1] = -1
        rows.append(r); labels.append(f"−w_B + {a_BO[j]:g}·w_{j+1} − ξ ≤ 0")
        # w_j − a_jW w_W − ξ ≤ 0   et   −w_j + a_jW w_W − ξ ≤ 0
        r = np.zeros(n + 1); r[j] += 1; r[worst] -= a_OW[j]; r[-1] = -1
        rows.append(r); labels.append(f"w_{j+1} − {a_OW[j]:g}·w_W − ξ ≤ 0")
        r = np.zeros(n + 1); r[j] -= 1; r[worst] += a_OW[j]; r[-1] = -1
        rows.append(r); labels.append(f"−w_{j+1} + {a_OW[j]:g}·w_W − ξ ≤ 0")
    A_ub = np.array(rows)
    b_ub = np.zeros(len(rows))
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise ValueError(f"BWM : le programme linéaire n'a pas convergé ({res.message})")
    w = res.x[:n]
    return {
        "weights": w / w.sum(),
        "xi": float(res.x[-1]),
        "A_ub": A_ub,
        "constraint_labels": labels,
        "n_comparisons": 2 * n - 3,
        "n_comparisons_ahp": n * (n - 1) // 2,
    }


# Indice de consistance de Rezaei (2015) selon a_BW
BWM_CI = {1: 0.0, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30, 6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23}


def bwm_nonlinear(best: int, worst: int, a_BO, a_OW, n_starts: int = 20, seed: int = 0) -> dict:
    """Modèle original de Rezaei (2015), non linéaire :

        min ξ  s.c.  |w_B / w_j − a_Bj| ≤ ξ,  |w_j / w_W − a_jW| ≤ ξ,  Σ w_j = 1.

    Résolu par SLSQP avec plusieurs points de départ ; CR = ξ* / CI(a_BW).
    """
    from scipy.optimize import minimize

    a_BO = np.asarray(a_BO, dtype=float)
    a_OW = np.asarray(a_OW, dtype=float)
    n = len(a_BO)

    def cons(x):
        w, xi = x[:n], x[n]
        return np.concatenate([xi - np.abs(w[best] / w - a_BO), xi - np.abs(w / w[worst] - a_OW)])

    rng = np.random.default_rng(seed)
    lin = bwm(best, worst, a_BO, a_OW)
    starts = [np.append(lin["weights"], lin["xi"] * 10 + 1)]
    starts += [np.append(rng.dirichlet(np.ones(n)), 5.0) for _ in range(n_starts)]
    best_res = None
    for x0 in starts:
        res = minimize(lambda x: x[n], x0=x0, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons},
                                    {"type": "eq", "fun": lambda x: x[:n].sum() - 1}],
                       bounds=[(1e-6, 1)] * n + [(0, None)])
        if res.success and (best_res is None or res.x[n] < best_res.x[n]):
            best_res = res
    if best_res is None:
        raise ValueError("BWM non linéaire : aucune solution trouvée.")
    xi = float(best_res.x[n])
    ci = BWM_CI.get(int(round(a_BO[worst])), 5.23)
    return {"weights": best_res.x[:n] / best_res.x[:n].sum(), "xi": xi,
            "CI": ci, "CR": xi / ci if ci > 0 else 0.0,
            "n_comparisons": 2 * n - 3, "n_comparisons_ahp": n * (n - 1) // 2}


def bwm_sanity(best: int, worst: int, a_BO, a_OW) -> list[str]:
    """Contrôles de bon sens du cours : a_BB = a_WW = 1, max(BO) porte sur W."""
    issues = []
    a_BO = np.asarray(a_BO, dtype=float)
    a_OW = np.asarray(a_OW, dtype=float)
    if best == worst:
        issues.append("Le critère Best et le critère Worst doivent être différents.")
    if a_BO[best] != 1:
        issues.append("a_BB doit valoir 1 (le Best comparé à lui-même).")
    if a_OW[worst] != 1:
        issues.append("a_WW doit valoir 1 (le Worst comparé à lui-même).")
    if a_BO[worst] != a_BO.max():
        issues.append("La note la plus élevée du vecteur BO devrait porter sur le critère Worst.")
    if a_BO[worst] != a_OW[best]:
        issues.append(f"Incohérence : a_BW (BO) = {a_BO[worst]:g} mais a_BW (OW) = {a_OW[best]:g}.")
    if ((a_BO < 1) | (a_BO > 9) | (a_OW < 1) | (a_OW > 9)).any():
        issues.append("Les notes doivent appartenir à l'échelle 1 à 9.")
    return issues


# ---------------------------------------------------------------------------
# Entropie
# ---------------------------------------------------------------------------
def entropy(X) -> dict:
    """Poids par l'entropie de Shannon.

    p_ij = d_ij / Σ_i d_ij,  E_j = −(1/ln m) Σ_i p_ij ln p_ij,
    w_j = (1 − E_j) / (n − Σ_k E_k).
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    if (X < 0).any():
        raise ValueError("L'entropie exige une matrice à valeurs positives ou nulles.")
    col = X.sum(axis=0)
    P = np.divide(X, col, out=np.full_like(X, 1.0 / m), where=col != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    k = 1.0 / np.log(m)
    E = -k * plogp.sum(axis=0)
    d = 1.0 - E
    denom = n - E.sum()
    w = d / denom if denom > 0 else np.full(n, 1.0 / n)
    return {"P": P, "E": E, "diversification": d, "weights": w, "k": k}


# ---------------------------------------------------------------------------
# CRITIC
# ---------------------------------------------------------------------------
def critic(X, types) -> dict:
    """CRITIC (Diakoulaki) selon le cours.

    1) Normalisation min–max par type (+ / −)
    2) Corrélation de Pearson ρ_jk
    3) σ_j (écart-type, dénominateur m−1),  C_j = σ_j Σ_k (1 − |ρ_jk|)
    4) w_j = C_j / Σ C_j
    """
    X = np.asarray(X, dtype=float)
    benefit = np.array([t == "+" for t in types])
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    R = np.where(benefit, (X - mn) / rng, (mx - X) / rng)
    sigma = R.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.corrcoef(R, rowvar=False)
    rho = np.nan_to_num(np.atleast_2d(rho), nan=0.0)
    np.fill_diagonal(rho, 1.0)
    conflict = (1.0 - np.abs(rho)).sum(axis=1)
    C = sigma * conflict
    w = C / C.sum() if C.sum() > 0 else np.full(len(C), 1.0 / len(C))
    return {"R": R, "sigma": sigma, "rho": rho, "conflict": conflict, "C": C, "weights": w}


# ---------------------------------------------------------------------------
# Hybride
# ---------------------------------------------------------------------------
def hybrid(w_subjective, w_objective, alpha: float = 0.5, mode: str = "linear") -> np.ndarray:
    """Combinaison de poids subjectifs et objectifs.

    linear         : w = α·w_subj + (1 − α)·w_obj
    multiplicative : w_j = w_subj,j · w_obj,j / Σ_k w_subj,k · w_obj,k
    """
    ws = np.asarray(w_subjective, dtype=float); ws = ws / ws.sum()
    wo = np.asarray(w_objective, dtype=float); wo = wo / wo.sum()
    if mode == "multiplicative":
        p = ws * wo
        return p / p.sum()
    return alpha * ws + (1 - alpha) * wo


# ---------------------------------------------------------------------------
# SWARA (Keršulienė et al., 2010) — citée par le cours parmi les méthodes subjectives
# ---------------------------------------------------------------------------
def swara(order, s) -> dict:
    """SWARA (Step-wise Weight Assessment Ratio Analysis).

    order : indices des critères, du plus au moins important.
    s     : importance comparative s_j de chaque critère par rapport au précédent dans cet
            ordre (s[0] est ignoré). k_j = 1 + s_j ; q_j = q_{j−1} / k_j ; w_j = q_j / Σ q.
    Renvoie les poids dans l'ordre d'origine des critères.
    """
    order = list(order)
    s = np.asarray(s, dtype=float).copy()
    if len(set(order)) != len(order) or len(s) != len(order):
        raise ValueError("SWARA : l'ordre doit contenir chaque critère une seule fois.")
    if (s[1:] < 0).any():
        raise ValueError("SWARA : les importances comparatives s_j doivent être positives.")
    s[0] = 0.0
    k = 1.0 + s
    q = np.ones(len(s))
    for j in range(1, len(s)):
        q[j] = q[j - 1] / k[j]
    w_sorted = q / q.sum()
    w = np.empty(len(s))
    w[order] = w_sorted
    return {"order": order, "s": s, "k": k, "q": q, "weights_sorted": w_sorted, "weights": w}
