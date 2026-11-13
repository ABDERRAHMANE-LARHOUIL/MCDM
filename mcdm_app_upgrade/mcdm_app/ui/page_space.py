"""Page 1 — Définition de l'espace de décision."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from utils import presets as P
from utils import validation as V
from ui.common import (X_array, heatmap, load_decision_problem, load_preset, page_header, persistent_editor,
                       plot, radar, reset_editor, ss, stats)


def _clean_names(series) -> list[str]:
    return [str(v).strip() for v in series if v is not None and str(v).strip() not in ("", "None", "nan")]


def _source_index(new_list, old_list, k, name):
    if name in old_list:
        return old_list.index(name)
    if k < len(old_list) and old_list[k] not in new_list:
        return k  # renommé sur place
    return None


def _remap_matrix(oldX: pd.DataFrame, alts, crits) -> pd.DataFrame:
    out = pd.DataFrame(1.0, index=alts, columns=crits)
    oa, oc = list(oldX.index), list(oldX.columns)
    for i, a in enumerate(alts):
        si = _source_index(alts, oa, i, a)
        for j, c in enumerate(crits):
            sj = _source_index(crits, oc, j, c)
            if si is not None and sj is not None:
                out.iloc[i, j] = oldX.iloc[si, sj]
    return out


def _remap_weights(old_w, old_crits, crits) -> np.ndarray:
    w = []
    for j, c in enumerate(crits):
        sj = _source_index(crits, old_crits, j, c)
        w.append(old_w[sj] if sj is not None else np.nan)
    w = np.array(w, dtype=float)
    w[np.isnan(w)] = np.nanmean(w) if np.isfinite(w).any() else 1.0
    return w / w.sum()


def render() -> None:
    page_header("Étape 1", "Espace de décision",
                "Alternatives A₁…Aₘ, critères C₁…Cₙ, sens de préférence (+ positif / benefit, "
                "− négatif / cost) et matrice de décision D = (xᵢⱼ).")

    with st.expander("Charger un exercice du cours ou importer un CSV", expanded=False, icon=":material/folder_open:"):
        c1, c2 = st.columns([3, 1])
        name = c1.selectbox("Exercice", list(P.DECISION_PRESETS), key="space_preset")
        if c2.button("Charger", width="stretch"):
            load_preset(name)
            st.rerun()
        up = st.file_uploader("CSV : 1re colonne = alternatives, en-têtes = critères. "
                              "Option : une ligne nommée « type » contenant + / −.", type=["csv"])
        if up is not None and st.button("Importer le CSV"):
            try:
                raw = pd.read_csv(io.BytesIO(up.getvalue()), index_col=0, sep=None, engine="python")
                types = ["+"] * raw.shape[1]
                if "type" in [str(i).lower() for i in raw.index]:
                    row = [i for i in raw.index if str(i).lower() == "type"][0]
                    types = ["-" if str(t).strip() in ("-", "−", "cost", "coût") else "+" for t in raw.loc[row]]
                    raw = raw.drop(index=row)
                load_decision_problem(list(map(str, raw.index)), list(map(str, raw.columns)), types,
                                      raw.apply(pd.to_numeric, errors="coerce").fillna(0).values,
                                      label=f"Import : {up.name}")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Import impossible : {exc}")


    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Critères")
        crit_df = persistent_editor(
            "crit", pd.DataFrame({"Critère": ss.crits, "Type": ss.types}),
            num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "Critère": st.column_config.TextColumn(required=True),
                "Type": st.column_config.SelectboxColumn(
                    options=["+", "-"], required=True, default="+",
                    help="+ : critère positif (à maximiser) · − : critère négatif (à minimiser)"),
            })
    with c2:
        st.subheader("Alternatives")
        alt_df = persistent_editor("alt", pd.DataFrame({"Alternative": ss.alts}),
                                   num_rows="dynamic", hide_index=True, width="stretch")

    valid_rows = crit_df.dropna(subset=["Critère"])
    valid_rows = valid_rows[valid_rows["Critère"].astype(str).str.strip() != ""]
    new_crits = _clean_names(valid_rows["Critère"])
    new_types = ["-" if str(t).strip() in ("-", "−") else "+" for t in valid_rows["Type"]]
    new_alts = _clean_names(alt_df["Alternative"])

    if len(set(new_crits)) != len(new_crits) or len(set(new_alts)) != len(new_alts):
        st.error("Les noms de critères et d'alternatives doivent être uniques.")
    elif new_crits and new_alts and (new_crits != ss.crits or new_alts != ss.alts):
        ss.weights = _remap_weights(ss.weights, ss.crits, new_crits)
        ss.X = _remap_matrix(ss.X, new_alts, new_crits)
        ss.crits, ss.alts = new_crits, new_alts
        ss.types = new_types
        ss.results = {}
        reset_editor("matrix", ss.X.copy())
        st.rerun()
    elif new_types != ss.types and len(new_types) == len(ss.crits):
        ss.types = new_types
        ss.results = {}

    st.subheader("Matrice de décision")
    st.caption("Les en-têtes indiquent le sens de préférence. Les cellules sont éditables.")
    edited = persistent_editor(
        "matrix", ss.X.copy(), width="stretch",
        column_config={c: st.column_config.NumberColumn(f"{c} ({t})", format="%.4g")
                       for c, t in zip(ss.crits, ss.types)})
    if list(edited.columns) == ss.crits and list(edited.index) == ss.alts:
        newX = edited.apply(pd.to_numeric, errors="coerce")
        if not newX.equals(ss.X):
            ss.X = newX
            ss.results = {}

    errors, warnings = V.check_decision_matrix(ss.X, ss.types)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)

    stats([("Problème courant", ss.get("problem_label") or "Personnalisé"), ("Alternatives (m)", str(len(ss.alts))),
           ("Critères (n)", str(len(ss.crits))),
           ("Positifs / négatifs", f"{ss.types.count('+')} / {ss.types.count('-')}")])

    st.download_button("Exporter la matrice (CSV)", icon=":material/download:", data=
                       pd.concat([ss.X, pd.DataFrame([ss.types], index=["type"], columns=ss.crits)])
                       .to_csv().encode("utf-8"), file_name="matrice_decision.csv", mime="text/csv")

    if not errors and len(ss.crits) >= 3:
        X = X_array()
        mn, mx = X.min(axis=0), X.max(axis=0)
        rng = np.where(mx - mn == 0, 1, mx - mn)
        b = np.array([t == "+" for t in ss.types])
        R = np.where(b, (X - mn) / rng, (mx - X) / rng)
        st.subheader("Profils des alternatives")
        if len(ss.alts) <= 8:
            c1, c2 = st.columns(2)
            with c1:
                plot(radar(R, ss.alts, ss.crits, "Profils normalisés min–max (1 = meilleure valeur)"))
            with c2:
                plot(heatmap(pd.DataFrame(R, index=ss.alts, columns=ss.crits),
                             "Performance normalisée par critère", zmin=0, zmax=1))
        else:
            plot(heatmap(pd.DataFrame(R, index=ss.alts, columns=ss.crits),
                         "Performance normalisée par critère (1 = meilleure valeur)", zmin=0, zmax=1))
