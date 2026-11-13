"""Vérifie que les méthodes reproduisent les résultats publiés dans le cours.

Exécution :  python -m pytest tests   (ou)   python tests/test_course_examples.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import crisp_ranking as CR  # noqa: E402
from core import fuzzy_mcdm as F  # noqa: E402
from core import outranking as O  # noqa: E402
from core import sensitivity as SE  # noqa: E402
from core import single_alternative as SA  # noqa: E402
from core import weighting as W  # noqa: E402
from utils import presets as P  # noqa: E402

D = P.DECISION_PRESETS


def test_ahp_car():
    r = W.ahp_rank(P.AHP_CAR["criteria_matrix"], P.AHP_CAR["alternative_matrices"])
    assert r["criteria"]["CR"] < 0.1
    assert np.argmax(r["criteria"]["weights"]) == 0          # le coût domine
    assert np.argmax(r["scores"]) == 0                       # voiture 1
    assert np.isclose(r["scores"].sum(), 1)


def test_ahp_eigen_matches_consistent_matrix():
    w = np.array([0.5, 0.3, 0.2])
    A = w[:, None] / w[None, :]
    for m in ("approx", "eigen"):
        r = W.ahp(A, m)
        assert np.allclose(r["weights"], w) and abs(r["CR"]) < 1e-9


def test_bwm_supplier_xi():
    p = P.BWM_SUPPLIER
    r = W.bwm(p["best"], p["worst"], p["a_BO"], p["a_OW"])
    assert abs(r["xi"] - 0.08) < 0.005                       # ξ* ≈ 0,08 (p. 42)
    assert list(np.argsort(-r["weights"])) == [0, 1, 2, 3]   # Qualité > Prix > Délai > RSE
    assert r["n_comparisons"] == 5


def test_entropy_and_critic_sum_to_one():
    p = D["CRITIC — Machines de découpage (p. 51)"]
    assert np.isclose(W.entropy(p["matrix"])["weights"].sum(), 1)
    c = W.critic(p["matrix"], p["types"])
    assert np.isclose(c["weights"].sum(), 1)
    assert np.argmax(c["weights"]) == 3


def test_waspas_machine():
    p = D["WSM / WPM / WASPAS — Choix d'une machine (p. 58)"]
    for f in (CR.wsm, CR.wpm, CR.waspas):
        assert f(p["matrix"], p["weights"], p["types"])["ranks"][3] == 1   # A4


def test_topsis_car():
    p = D["TOPSIS — Choix d'une voiture (p. 65)"]
    r = CR.topsis(p["matrix"], p["weights"], p["types"])
    assert list(r["ranks"]) == [3, 1, 4, 2]


def test_vikor_machine():
    p = D["VIKOR — Achat d'une machine (p. 72)"]
    r = CR.vikor(p["matrix"], p["weights"], p["types"])
    assert np.allclose(r["S"], [0.5, 0.375, 0.5]) and np.allclose(r["Q"], [1, 0, 1])
    assert r["compromise"] == [1] and r["C1_advantage"] and r["C2_stability"]


def test_promethee_net_flows_sum_to_zero():
    p = D["TOPSIS — Choix d'une voiture (p. 65)"]
    r = O.promethee(p["matrix"], p["weights"], p["types"])
    assert abs(r["scores"].sum()) < 1e-12


def test_spearman_course():
    M = SE.spearman_matrix({k: np.array(v) for k, v in P.SPEARMAN_EXAMPLE["rankings"].items()})
    for (a, b), e in P.SPEARMAN_EXAMPLE["expected"].items():
        assert abs(M.loc[a, b] - e) < 0.01


def test_roc_course_cases():
    p = P.ROC_EXAMPLE
    rows = SA.roc_table(p["scores"], p["labels"], p["thresholds"])
    got = [(round(r["TPR"], 2), round(r["FPR"], 2), round(r["d"], 2), round(r["J"], 2)) for r in rows]
    assert got == [(0.67, 0.0, 0.33, 0.67), (1.0, 0.33, 0.33, 0.67), (1.0, 0.67, 0.67, 0.33)]


def test_fuzzy_single_course():
    p = P.FUZZY_SINGLE
    r = SA.fuzzy_single(p["tfns"], p["weights"])
    assert np.allclose(r["S_fuzzy"], p["expected_S_fuzzy"], atol=0.01)
    assert abs(r["S"] - p["expected_S"]) < 0.01


def test_maut_utilities():
    u = SA.utilities([5, 5], [0, 0], [10, 10], ["+", "-"])
    assert np.allclose(u, [0.5, 0.5])


def test_chen_recruitment():
    p = P.CHEN_RECRUITMENT
    X = F.aggregate_tfn_mean([F.linguistic_to_tfn(m, F.CHEN_RATING_SCALE) for m in p["ratings"]])
    Wt = F.aggregate_tfn_mean([F.linguistic_to_tfn(m, F.CHEN_WEIGHT_SCALE) for m in p["weights"]])
    r = F.fuzzy_topsis_chen(X, Wt, p["types"])
    assert list(r["ranks"]) == [3, 1, 2]                     # A2 > A3 > A1


def test_ifwa_identity():
    mu, nu = F.ifwa([[0.6], [0.6]], [[0.3], [0.3]], [0.5, 0.5])
    assert np.allclose(mu, 0.6) and np.allclose(nu, 0.3)


def test_perturb_weight_keeps_ratios():
    w = SE.perturb_weight([0.2, 0.3, 0.5], 0, 0.6)
    assert np.isclose(w.sum(), 1) and np.isclose(w[0], 0.6) and np.isclose(w[1] / w[2], 0.6)


if __name__ == "__main__":
    tests = [(n, f) for n, f in dict(globals()).items() if n.startswith("test_")]
    for n, f in tests:
        f()
        print(f"OK  {n}")
    print(f"{len(tests)} tests réussis")


# ---------------------------------------------------------------------------
# Méthodes ajoutées : ELECTRE III, SWARA, logique floue, MCDM & IA
# ---------------------------------------------------------------------------
from core import ai as AI  # noqa: E402
from core import fuzzy_logic as FL  # noqa: E402


def test_electre3_dominance_and_veto():
    X = np.array([[9, 9, 1], [5, 5, 5], [1, 1, 9.0]])
    r = O.electre3(X, [1 / 3] * 3, ["+", "+", "-"])
    assert list(r["ranks"]) == [1, 2, 3]
    assert np.allclose(np.diag(r["S"]), 1) and (r["S"] >= 0).all() and (r["S"] <= 1).all()
    # un veto franchi annule la crédibilité du surclassement
    X2 = np.array([[10, 0.0], [9, 10.0]])
    r2 = O.electre3(X2, [0.9, 0.1], ["+", "+"], q=[0, 0], p=[1, 1], v=[np.nan, 5])
    assert r2["C"][0, 1] > 0.8 and r2["S"][0, 1] == 0


def test_electre3_on_course_problems():
    for p in D.values():
        r = O.electre3(p["matrix"], p["weights"] or [1] * len(p["criteria"]), p["types"])
        assert sorted(r["ranks"])[0] == 1 and len(r["ranks"]) == len(p["alternatives"])


def test_swara():
    r = W.swara([2, 0, 1], [0, 0.25, 0.5])
    assert abs(r["weights"].sum() - 1) < 1e-12
    assert r["weights"][2] > r["weights"][0] > r["weights"][1]
    assert np.isclose(r["weights"][2] / r["weights"][0], 1.25)


def test_fuzzy_logic_course_values():
    # âge moyen (p. 104) : μ(45) = 1, μ(30) = 0,5
    assert FL.mu_trapezoid(45, 25, 35, 55, 65) == 1 and np.isclose(FL.mu_trapezoid(30, 25, 35, 55, 65), 0.5)
    assert FL.fz_sub((4, 5, 6, 7), (1, 2, 3, 4)) == (0, 2, 4, 6)
    # nombre intuitionniste (0,6 ; 0,3) → π = 0,1 (p. 114)
    assert np.isclose(FL.orthopair_hesitation(0.6, 0.3, 1), 0.1)
    assert FL.orthopair_hesitation(0.8, 0.5, 1) is None and FL.min_valid_q(0.8, 0.5) == 2
    assert np.allclose(FL.ifwa_list([(0.6, 0.3), (0.6, 0.3)], [0.5, 0.5]), (0.6, 0.3))
    # nombres hésitants (p. 128) : h2 > h3 > h1
    h1, h2, h3 = [0.2, 0.3, 0.5, 0.8], [0.4, 0.6, 0.8], [0.3, 0.45, 0.6]
    assert FL.hfe_compare(h2, h3) == 1 and FL.hfe_compare(h3, h1) == 1
    assert np.isclose(FL.picture_refusal((0.5, 0.3, 0.1)), 0.1)


def test_shapley_linear_model():
    rng = np.random.default_rng(0)
    X = rng.random((12, 4))
    beta = np.array([0.4, 0.3, 0.2, 0.1])
    phi0, phi = AI.shapley_values(lambda Z: Z @ beta, X)
    assert np.allclose(phi0 + phi.sum(axis=1), X @ beta)
    assert np.allclose(phi, (X - X.mean(axis=0)) * beta)
    upd = AI.shap_weight_update(beta, phi, 0.5)
    assert abs(upd["weights"].sum() - 1) < 1e-12


def test_forecast_and_labels():
    y = np.arange(10.0)
    assert np.isclose(AI.forecast_next(y, "trend"), 10) and AI.forecast_next(y, "naive") == 9
    lab = AI.label_by_quantiles(np.linspace(0, 1, 20))
    assert lab[-1] == "A" and lab[0] == "D" and len(set(lab)) == 4


def test_minmax_normalisation_handles_negative_data():
    p = [v for k, v in D.items() if "Finance" in k][0]
    X, w, t = np.array(p["matrix"]), np.array(p["weights"]), p["types"]
    try:
        CR.wsm(X, w, t)
        raise AssertionError("la normalisation linéaire devrait refuser des valeurs négatives")
    except ValueError:
        pass
    r = CR.wsm(X, w, t, norm="minmax")
    assert np.isfinite(r["scores"]).all() and r["R"].min() >= 0 and r["R"].max() <= 1
    assert sorted(r["ranks"])[0] == 1
    # sur des données positives, le comportement par défaut (cours) est inchangé
    q = D["TOPSIS — Choix d'une voiture (p. 65)"]
    assert np.allclose(CR.wsm(q["matrix"], q["weights"], q["types"])["scores"],
                       CR.wsm(q["matrix"], q["weights"], q["types"], "linear")["scores"])
