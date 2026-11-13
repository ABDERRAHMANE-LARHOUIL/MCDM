"""Études de cas pré-chargées, reprises des exercices du cours
« Aide à la décision » (Prof. Lamrani Alaoui, EMI Rabat).

Chaque preset indique la page du support de cours d'où il est tiré. Les cas
marqués `illustrative=True` ne figurent pas dans le cours (données d'exemple
construites pour illustrer une méthode dont le cours ne donne pas de chiffres).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Matrices de décision « crisp » (espace de décision)
# ---------------------------------------------------------------------------
DECISION_PRESETS = {
    "TOPSIS — Choix d'une voiture (p. 65)": {
        "alternatives": ["M1", "M2", "M3", "M4"],
        "criteria": ["Style", "Fiabilité", "Économie carburant", "Coût"],
        "types": ["+", "+", "+", "-"],
        "matrix": [[7, 9, 9, 8],
                   [8, 7, 8, 7],
                   [9, 6, 8, 9],
                   [6, 7, 8, 6]],
        "weights": [0.2, 0.1, 0.4, 0.3],
        "method": "TOPSIS",
        "page": 65,
    },
    "WSM / WPM / WASPAS — Choix d'une machine (p. 58)": {
        "alternatives": ["A1", "A2", "A3", "A4"],
        "criteria": ["C1", "C2", "C3", "C4", "C5"],
        "types": ["-", "-", "+", "-", "+"],
        "matrix": [[0.035, 847, 0.335, 1.760, 0.590],
                   [0.027, 834, 0.335, 1.680, 0.665],
                   [0.037, 808, 0.590, 2.400, 0.500],
                   [0.028, 821, 0.500, 1.590, 0.410]],
        "weights": [0.331, 0.181, 0.369, 0.072, 0.047],
        "method": "WASPAS",
        "page": 58,
    },
    "CRITIC — Machines de découpage (p. 51)": {
        "alternatives": ["A1", "A2", "A3", "A4"],
        "criteria": ["C1 Épaisseur max", "C2 Largeur coupe min", "C3 Qualité découpe", "C4 Coût maintenance"],
        "types": ["-", "-", "+", "-"],
        "matrix": [[30, 0.100, 1, 20],
                   [100, 0.700, 1, 40],
                   [50, 1, 2, 10],
                   [300, 2, 3, 35]],
        "weights": None,
        "method": "CRITIC",
        "page": 51,
    },
    "VIKOR — Achat d'une machine (p. 72)": {
        "alternatives": ["A1", "A2", "A3"],
        "criteria": ["C1", "C2"],
        "types": ["-", "+"],
        "matrix": [[1, 3000],
                   [2, 3750],
                   [5, 4500]],
        "weights": [0.5, 0.5],
        "method": "VIKOR",
        "page": 72,
    },
    "Finance — Sélection d'actions, Vuković et al. 2020 (p. 18)": {
        "alternatives": ["ADPL", "ADRS2", "ARNT", "ATGR", "ATPL", "DDJH", "DLKV", "ERNT", "HT", "INGR", "KOEI",
                         "KRAS", "MAIS", "OPTE", "PODR", "RIVP", "VLEN", "ZABA"],
        "criteria": ["C1 Rendement moyen (%)", "C2 Écart-type (%)", "C3 Volume", "C4 Bêta", "C5 EPS",
                     "C6 ROE (%)", "C7 P/B", "C8 P/S", "C9 ROA (%)"],
        "types": ["+", "-", "+", "-", "+", "+", "-", "-", "+"],
        "matrix": [[0.2137, 2.19, 1382.93, 0.51, 16.72, 9.36, 0.95, 0.66, 5.20],
                   [-0.1037, 1.94, 2302.18, 0.68, 55.08, 3.63, 0.28, 0.67, 1.90],
                   [-0.3592, 2.67, 864.13, 0.37, 17.17, 5.63, 1.47, 3.24, 3.29],
                   [0.2753, 3.06, 420.82, 1.12, 82.84, 12.28, 1.28, 0.54, 5.39],
                   [0.1222, 5.90, 762.22, 0.06, 58.38, 14.42, 1.58, 2.44, 5.70],
                   [-1.2446, 7.06, 4504.48, 0.68, -3.29, -38.62, 2.29, 0.38, -5.35],
                   [-0.5443, 5.65, 10656.87, 1.42, 0.50, 5.23, 1.98, 0.32, 0.94],
                   [-0.1648, 2.69, 162.01, 0.70, 50.98, 28.82, 6.81, 1.08, 8.26],
                   [-0.1194, 1.60, 7220.56, 0.46, 9.85, 6.40, 1.07, 1.74, 5.11],
                   [0.1794, 5.22, 13159.67, 1.77, 0.95, 12.02, 0.46, 0.72, 1.50],
                   [-0.2807, 2.34, 374.91, 0.81, 42.03, 4.35, 0.74, 0.65, 2.91],
                   [-0.3532, 2.44, 194.47, 0.79, 22.17, 4.57, 0.95, 0.62, 2.51],
                   [0.1381, 2.79, 311.65, 1.33, 14.82, 10.33, 1.95, 2.78, 5.50],
                   [-0.4341, 5.26, 28452.85, 1.18, -0.85, -303.61, 9.70, 0.38, -8.96],
                   [-0.0184, 2.54, 1419.53, 1.03, 3.35, 0.82, 0.66, 0.47, 0.47],
                   [-0.0424, 2.66, 31328.13, 0.70, 1.94, 9.74, 2.19, 3.14, 4.91],
                   [-0.4791, 8.55, 25731.26, 1.87, 2.18, 14.86, 0.94, 0.48, 9.08],
                   [-0.0144, 3.73, 6927.05, 1.47, 3.26, 5.76, 0.93, 3.11, 0.82]],
        "weights": [0.0370, 0.0185, 0.0533, 0.0259, 0.2180, 0.3070, 0.0764, 0.1543, 0.1089],
        "method": "TOPSIS",
        "page": 18,
    },
    "MCDM dynamique — Actions, prévision t+1 (p. 94)": {
        "alternatives": ["A1 Tech", "A2 Énergie", "A3 Bancaire"],
        "criteria": ["Rendement r (%)", "Volatilité σ", "Liquidité L"],
        "types": ["+", "-", "+"],
        "matrix": [[8, 0.25, 100],
                   [6, 0.15, 80],
                   [7, 0.20, 120]],
        "weights": None,
        "method": "TOPSIS",
        "page": 94,
    },
}

# ---------------------------------------------------------------------------
# AHP — Achat d'une voiture (p. 32–34)
# ---------------------------------------------------------------------------
AHP_CAR = {
    "criteria": ["Coût", "Confort", "Sécurité"],
    "alternatives": ["Voiture 1", "Voiture 2"],
    "criteria_matrix": [[1, 7, 3],
                        [1 / 7, 1, 1 / 3],
                        [1 / 3, 3, 1]],
    "alternative_matrices": [
        [[1, 7], [1 / 7, 1]],     # coût
        [[1, 0.2], [5, 1]],       # confort
        [[1, 1 / 9], [9, 1]],     # sécurité
    ],
    "page": 32,
}

# ---------------------------------------------------------------------------
# BWM — Choix d'un fournisseur (p. 37–43)
# ---------------------------------------------------------------------------
BWM_SUPPLIER = {
    "criteria": ["C1 Qualité", "C2 Prix", "C3 Délai de livraison", "C4 Engagement RSE"],
    "best": 0,
    "worst": 3,
    "a_BO": [1, 3, 4, 8],
    "a_OW": [8, 5, 3, 1],
    "expected_weights": [0.552, 0.207, 0.155, 0.086],
    "page": 37,
}

# ---------------------------------------------------------------------------
# Spearman — comparaison TOPSIS / VIKOR / PROMETHEE II (p. 78–79)
# ---------------------------------------------------------------------------
SPEARMAN_EXAMPLE = {
    "alternatives": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "rankings": {
        "TOPSIS": [2, 1, 3, 5, 4, 6],
        "VIKOR": [1, 2, 4, 3, 5, 6],
        "PROMETHEE II": [2, 1, 3, 6, 4, 5],
    },
    "expected": {("TOPSIS", "VIKOR"): 0.77, ("TOPSIS", "PROMETHEE II"): 0.94,
                 ("VIKOR", "PROMETHEE II"): 0.60},
    "page": 78,
}

# ---------------------------------------------------------------------------
# Seuil de décision — 6 décisions passées (p. 85–88)
# ---------------------------------------------------------------------------
ROC_EXAMPLE = {
    "alternatives": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "scores": [0.85, 0.78, 0.71, 0.69, 0.62, 0.54],
    "labels": [1, 1, 1, 0, 0, 0],   # 1 = Accepter, 0 = Rejeter
    "thresholds": [0.75, 0.65, 0.60],
    "page": 88,
}

# ---------------------------------------------------------------------------
# MAUT — alternative isolée (illustratif : le cours ne donne pas de chiffres)
# ---------------------------------------------------------------------------
MAUT_CREDIT = {
    "criteria": ["Revenu mensuel (DH)", "Taux d'endettement", "Ancienneté (ans)", "Incidents de paiement"],
    "types": ["+", "-", "+", "-"],
    "x": [12000, 0.35, 6, 1],
    "x_min": [3000, 0.0, 0, 0],
    "x_max": [30000, 0.6, 20, 5],
    "weights": [0.35, 0.30, 0.15, 0.20],
    "tau": 0.7,
    "illustrative": True,
}

# ---------------------------------------------------------------------------
# Évaluation floue d'une seule alternative (p. 134)
# ---------------------------------------------------------------------------
FUZZY_SINGLE = {
    "criteria": ["C1 : ROA", "C2 : Liquidité"],
    "weights": [0.6, 0.4],
    "tfns": [[1.0, 1.2, 1.5], [110, 120, 130]],
    "expected_S_fuzzy": [0.74, 0.85, 1.00],
    "expected_S": 0.86,
    "page": 134,
}

# ---------------------------------------------------------------------------
# TOPSIS flou (Chen, 2000) — Recrutement d'un ingénieur (p. 140–141)
# ---------------------------------------------------------------------------
CHEN_RECRUITMENT = {
    "alternatives": ["A1", "A2", "A3"],
    "criteria": ["C1 Stabilité émotionnelle", "C2 Communication orale", "C3 Personnalité",
                 "C4 Expérience passée", "C5 Confiance en soi"],
    "types": ["+", "+", "+", "+", "+"],
    "decision_makers": ["DM1", "DM2", "DM3"],
    # ratings[k][i][j] : décideur k, alternative i, critère j
    "ratings": [
        [["MG", "G", "F", "VG", "F"],
         ["G", "VG", "VG", "VG", "VG"],
         ["VG", "MG", "G", "G", "G"]],
        [["G", "MG", "G", "G", "F"],
         ["G", "VG", "VG", "VG", "MG"],
         ["G", "G", "MG", "VG", "G"]],
        [["MG", "F", "G", "VG", "F"],
         ["MG", "VG", "G", "VG", "G"],
         ["F", "VG", "VG", "MG", "MG"]],
    ],
    # weights[k][j] : importance du critère j selon le décideur k
    "weights": [
        ["H", "VH", "VH", "VH", "M"],
        ["VH", "VH", "H", "VH", "MH"],
        ["MH", "VH", "H", "VH", "MH"],
    ],
    "page": 140,
}

# ---------------------------------------------------------------------------
# TOPSIS flou intuitionniste (illustratif : sélection d'un fournisseur)
# ---------------------------------------------------------------------------
IF_SUPPLIER = {
    "alternatives": ["F1", "F2", "F3", "F4"],
    "criteria": ["Qualité", "Délai", "Service", "Prix"],
    "types": ["+", "+", "+", "-"],
    "decision_makers": ["DM1", "DM2", "DM3"],
    "dm_weights": [0.40, 0.35, 0.25],
    "ratings": [
        [["VG", "G", "MG", "F"], ["G", "VG", "G", "MG"], ["MG", "F", "VG", "G"], ["F", "MG", "G", "VG"]],
        [["G", "G", "G", "MG"], ["VG", "G", "MG", "G"], ["MG", "MG", "G", "F"], ["G", "F", "MG", "G"]],
        [["VG", "MG", "G", "F"], ["G", "G", "VG", "MG"], ["F", "MG", "G", "MG"], ["MG", "G", "F", "VG"]],
    ],
    "weights": [
        ["VI", "I", "M", "I"],
        ["VI", "M", "I", "VI"],
        ["I", "I", "M", "VI"],
    ],
    "illustrative": True,
}
