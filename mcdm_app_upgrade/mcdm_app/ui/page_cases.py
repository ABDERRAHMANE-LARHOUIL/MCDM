"""Page 6 — Études de cas pédagogiques reprises du cours, avec vérification
des résultats attendus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import crisp_ranking as CR
from core import fuzzy_mcdm as F
from core import sensitivity as SE
from core import single_alternative as SA
from core import weighting as W
from utils import presets as P
from ui.common import bar_scores, check_badge, load_preset, page_header, plot, show_df, tfn_frame, yes_no


def _goto_space(preset: str) -> None:
    if st.button("Charger ce cas dans l'espace de décision", key=f"go_{preset}", icon=":material/arrow_forward:"):
        load_preset(preset)
        st.success("Cas chargé : poursuivez dans les pages 2 à 5.")


def _matrix(p) -> pd.DataFrame:
    return pd.DataFrame(p["matrix"], index=p["alternatives"],
                        columns=[f"{c} ({t})" for c, t in zip(p["criteria"], p["types"])])


def case_ahp() -> None:
    p = P.AHP_CAR
    st.markdown("> Vous voulez acheter une voiture ; les critères importants sont le **coût**, le **confort** "
                "et la **sécurité**. Choisir entre voiture 1 et voiture 2.")
    r = W.ahp_rank(p["criteria_matrix"], p["alternative_matrices"])
    c = r["criteria"]
    st.markdown("**1) Poids des critères et test de consistance**")
    df = pd.DataFrame(c["normalized"], index=p["criteria"], columns=p["criteria"])
    df["Priorité w"] = c["weights"]
    show_df(df)
    m = st.columns(4)
    m[0].metric("λmax", f"{c['lambda_max']:.4f}")
    m[1].metric("CI", f"{c['CI']:.4f}")
    m[2].metric("RI (n = 3)", f"{c['RI']:.2f}")
    m[3].metric("CR", f"{c['CR']:.4f}", "consistante" if c["consistent"] else "incohérente",
                delta_color="normal" if c["consistent"] else "inverse")
    st.markdown("**2) Priorités locales et finales des voitures**")
    L = pd.DataFrame(r["local_matrix"], index=p["alternatives"], columns=p["criteria"])
    L["Priorité finale Σ wⱼ·wⱼᵢ"] = r["scores"]
    show_df(L, highlight_max="Priorité finale Σ wⱼ·wⱼᵢ")
    best = p["alternatives"][int(np.argmax(r["scores"]))]
    st.success(f"Meilleur choix : **{best}** (le coût, critère dominant avec w ≈ {c['weights'][0]:.2f}, "
               "l'emporte sur le confort et la sécurité).")


def case_bwm() -> None:
    p = P.BWM_SUPPLIER
    st.markdown("> Choisir un fournisseur selon 4 critères. Best = C1 (Qualité), Worst = C4 (RSE).")
    show_df(pd.DataFrame([p["a_BO"], p["a_OW"]], index=["BO (Best → autres)", "OW (autres → Worst)"],
                         columns=p["criteria"]), 0)
    lin = W.bwm(p["best"], p["worst"], p["a_BO"], p["a_OW"])
    nl = W.bwm_nonlinear(p["best"], p["worst"], p["a_BO"], p["a_OW"])
    tbl = pd.DataFrame([lin["weights"], nl["weights"], p["expected_weights"]],
                       index=["Modèle linéaire (cours) — calculé", "Modèle non linéaire (Rezaei 2015)",
                              "Valeurs affichées sur la diapositive (p. 42)"], columns=p["criteria"])
    tbl["ξ*"] = [lin["xi"], nl["xi"], np.nan]
    show_df(tbl)
    exp = np.array(p["expected_weights"])
    xi_needed = max(np.abs(exp[0] - np.array(p["a_BO"]) * exp).max(),
                    np.abs(exp - np.array(p["a_OW"]) * exp[3]).max())
    st.markdown(f"{check_badge(lin['xi'], 0.08)} ξ* = {lin['xi']:.4f} ≈ 0,08 comme dans le cours.")
    st.warning(
        ":orange[:material/warning:] Les poids de la diapositive (0,552 ; 0,207 ; 0,155 ; 0,086) ne vérifient pas les contraintes du "
        f"modèle linéaire pour ξ = 0,08 : ils exigeraient ξ ≥ {xi_needed:.3f} (par exemple "
        f"|w₁ − 8·w₄| = {abs(exp[0] - 8 * exp[3]):.3f}). L'optimum du programme linéaire est unique : "
        f"w = ({', '.join(f'{v:.3f}' for v in lin['weights'])}). La plateforme retient la solution du modèle "
        "du cours ; le classement des critères (Qualité > Prix > Délai > RSE) est identique.")


def case_critic() -> None:
    name = "CRITIC — Machines de découpage (p. 51)"
    p = P.DECISION_PRESETS[name]
    st.markdown("> Évaluer 4 machines de découpage ; déterminer l'importance des critères par CRITIC.")
    show_df(_matrix(p), 3)
    r = W.critic(p["matrix"], p["types"])
    st.markdown("**Matrice normalisée et corrélations**")
    c1, c2 = st.columns(2)
    with c1:
        show_df(pd.DataFrame(r["R"], index=p["alternatives"], columns=p["criteria"]))
    with c2:
        show_df(pd.DataFrame(r["rho"], index=p["criteria"], columns=p["criteria"]))
    show_df(pd.DataFrame({"σⱼ": r["sigma"], "Σ(1 − |ρ|)": r["conflict"], "Cⱼ": r["C"], "wⱼ": r["weights"]},
                         index=p["criteria"]).T)
    t = CR.topsis(p["matrix"], r["weights"], p["types"])
    st.markdown("**Prolongement : classement TOPSIS avec les poids CRITIC**")
    show_df(pd.DataFrame({"RC*": t["scores"], "Rang": t["ranks"]}, index=p["alternatives"]), highlight_max="RC*")
    _goto_space(name)


def case_waspas() -> None:
    name = "WSM / WPM / WASPAS — Choix d'une machine (p. 58)"
    p = P.DECISION_PRESETS[name]
    st.markdown("> 4 machines, 5 critères, poids donnés (0,331 ; 0,181 ; 0,369 ; 0,072 ; 0,047).")
    show_df(_matrix(p), 3)
    r = CR.waspas(p["matrix"], p["weights"], p["types"])
    st.markdown("**Matrice normalisée** (x/max pour +, min/x pour −)")
    show_df(pd.DataFrame(r["R"], index=p["alternatives"], columns=p["criteria"]))
    df = pd.DataFrame({"Q(1) WSM": r["Q1"], "Q(2) WPM": r["Q2"], "Q WASPAS (λ = 0,5)": r["scores"],
                       "Rang": r["ranks"]}, index=p["alternatives"])
    show_df(df, highlight_max="Q WASPAS (λ = 0,5)")
    st.success(f"Meilleure machine : **{p['alternatives'][int(np.argmax(r['scores']))]}** "
               "(identique pour WSM, WPM et WASPAS).")
    _goto_space(name)


def case_topsis() -> None:
    name = "TOPSIS — Choix d'une voiture (p. 65)"
    p = P.DECISION_PRESETS[name]
    st.markdown("> Sélectionner une voiture parmi 4 marques ; poids (0,2 ; 0,1 ; 0,4 ; 0,3).")
    show_df(_matrix(p), 0)
    r = CR.topsis(p["matrix"], p["weights"], p["types"])
    st.markdown("**Étapes 1–3**")
    show_df(pd.concat([pd.DataFrame(r["V"], index=p["alternatives"], columns=p["criteria"]),
                       pd.DataFrame([r["I_plus"], r["I_minus"]], index=["I+", "I−"], columns=p["criteria"])]))
    st.markdown("**Étapes 4–5**")
    show_df(pd.DataFrame({"S+": r["S_plus"], "S−": r["S_minus"], "RC*": r["scores"], "Rang": r["ranks"]},
                         index=p["alternatives"]), highlight_max="RC*")
    plot(bar_scores(p["alternatives"], r["scores"], "Coefficient de proximité RC*"))
    _goto_space(name)


def case_vikor() -> None:
    name = "VIKOR — Achat d'une machine (p. 72)"
    p = P.DECISION_PRESETS[name]
    st.markdown("> Acheter une machine parmi 3 ; deux critères de même poids.")
    show_df(_matrix(p), 0)
    r = CR.vikor(p["matrix"], p["weights"], p["types"])
    show_df(pd.DataFrame({"S": r["S"], "R": r["R"], "Q": r["Q"], "Rang Q": r["ranks"]},
                         index=p["alternatives"]), highlight_min="Q")
    st.markdown(f"{yes_no(r['C1_advantage'])} Avantage acceptable (DQ = {r['DQ']:.2f}) · "
                f"{yes_no(r['C2_stability'])} Stabilité acceptable")
    st.success("Solution de compromis : " + ", ".join(p["alternatives"][k] for k in r["compromise"]))
    _goto_space(name)


def case_spearman() -> None:
    p = P.SPEARMAN_EXAMPLE
    st.markdown("> Rangs de A1 à A6 selon TOPSIS, VIKOR et PROMETHEE II.")
    show_df(pd.DataFrame(p["rankings"], index=p["alternatives"]), 0)
    M = SE.spearman_matrix({k: np.array(v) for k, v in p["rankings"].items()})
    show_df(M, 2)
    for (a, b), e in p["expected"].items():
        st.markdown(f"{check_badge(M.loc[a, b], e)} ρ({a}, {b}) = {M.loc[a, b]:.3f} — cours : {e:.2f}")
    st.info("Avec n = 6, seule ρ ≥ 0,886 est significative à 5 % : uniquement TOPSIS – PROMETHEE II. "
            "Le choix de A1 et A2 est robuste ; le rang de A3 à A6 dépend de la méthode.")


def case_roc() -> None:
    p = P.ROC_EXAMPLE
    st.markdown("> 6 décisions passées ; choisir le seuil τ qui sépare le mieux Accepter / Rejeter.")
    show_df(pd.DataFrame({"Score S": p["scores"],
                          "Décision": ["Accepter" if y else "Rejeter" for y in p["labels"]]},
                         index=p["alternatives"]), 2)
    rows = SA.roc_table(p["scores"], p["labels"], p["thresholds"])
    df = pd.DataFrame(rows)[["tau", "TPR", "FPR", "d", "J"]]
    df.index = [f"Cas {i+1}" for i in range(len(df))]
    show_df(df.rename(columns={"tau": "Seuil τ"}), 2)
    st.info("Cas 1 (τ = 0,75) et cas 2 (τ = 0,65) sont à égalité (d = 0,33 ; J = 0,67) : "
            "départage selon le coût d'un faux positif ou d'un faux négatif.")


def case_fuzzy_single() -> None:
    p = P.FUZZY_SINGLE
    st.markdown("> Évaluer une seule alternative avec des nombres flous triangulaires (ROA, liquidité).")
    r = SA.fuzzy_single(p["tfns"], p["weights"])
    show_df(pd.DataFrame(r["R"], index=p["criteria"], columns=["a", "b", "c"]).assign(Poids=p["weights"]), 3)
    st.markdown(f"S̃ = ({', '.join(f'{v:.2f}' for v in r['S_fuzzy'])}) — cours : (0,74 ; 0,85 ; 1,00)  "
                f"{check_badge(r['S_fuzzy'][0], 0.74)}")
    st.markdown(f"S = {r['S']:.2f} — cours : 0,86 {check_badge(r['S'], 0.86)} · largeur c − a = {r['width']:.2f} "
                "(incertitude, à surveiller en plus du score).")


def case_chen() -> None:
    p = P.CHEN_RECRUITMENT
    st.markdown("> Recruter un ingénieur d'analyse de systèmes parmi 3 candidats ; 3 décideurs, 5 critères.")
    X = F.aggregate_tfn_mean([F.linguistic_to_tfn(m, F.CHEN_RATING_SCALE) for m in p["ratings"]])
    Wt = F.aggregate_tfn_mean([F.linguistic_to_tfn(m, F.CHEN_WEIGHT_SCALE) for m in p["weights"]])
    r = F.fuzzy_topsis_chen(X, Wt, p["types"])
    st.markdown("**1) Matrice de décision collective**")
    st.dataframe(tfn_frame(X, p["alternatives"], p["criteria"], 2), width="stretch")
    st.markdown("**2) Matrice collective normalisée**")
    st.dataframe(tfn_frame(r["R"], p["alternatives"], p["criteria"], 3), width="stretch")
    st.markdown("**3) Poids collectifs**")
    st.dataframe(tfn_frame(Wt[None], ["w̃ⱼ"], p["criteria"], 3), width="stretch")
    st.markdown("**4) Matrice normalisée pondérée**")
    st.dataframe(tfn_frame(r["V"], p["alternatives"], p["criteria"], 3), width="stretch")
    st.markdown("**5) Résolution**")
    show_df(pd.DataFrame({"D*": r["D_star"], "D−": r["D_minus"], "CC": r["scores"], "Rang": r["ranks"]},
                         index=p["alternatives"]), highlight_max="CC")
    order = " > ".join(p["alternatives"][i] for i in np.argsort(r["ranks"]))
    st.success(f"Classement : **{order}** — le candidat A2 est retenu.")


CASES = {
    "AHP — Achat d'une voiture (p. 32–34)": case_ahp,
    "BWM — Choix d'un fournisseur (p. 37–43)": case_bwm,
    "CRITIC — Machines de découpage (p. 51)": case_critic,
    "WSM / WPM / WASPAS — Choix d'une machine (p. 58)": case_waspas,
    "TOPSIS — Choix d'une voiture (p. 65)": case_topsis,
    "VIKOR — Achat d'une machine (p. 72)": case_vikor,
    "Spearman — Comparaison de méthodes (p. 78–79)": case_spearman,
    "Seuil de décision — ROC / Youden (p. 85–88)": case_roc,
    "Floue — Une seule alternative (p. 134)": case_fuzzy_single,
    "TOPSIS flou (Chen 2000) — Recrutement (p. 140–142)": case_chen,
}


def render() -> None:
    page_header("Apprendre", "Études de cas du cours",
                "Exercices repris des diapositives et résolus pas à pas. Une coche verte signale un résultat "
                "conforme à la valeur publiée dans le cours ; un triangle orange, un écart commenté.")
    name = st.selectbox("Étude de cas", list(CASES))
    st.subheader(name)
    CASES[name]()
