"""
rule_based.py
--------------
Règles métier explicites pour la détection d'anomalies.

Ces règles sont basées sur la connaissance du domaine :
- Montants extrêmes
- Transactions hors heures de bureau
- Sources non standard
- Montants ronds suspects
- Fréquence anormale

Chaque règle produit un flag binaire. Le score final est la somme pondérée.

Retourne :
    - rule_* : flags individuels par règle
    - score_rules : score composite [0,1]
    - anomaly_rules : flag si score_rules dépasse le seuil
"""

import numpy as np
import pandas as pd

from src.config import (
    RULE_AMOUNT_HIGH_Z,
    RULE_WEEKEND_WEIGHT,
    RULE_OUTSIDE_HOURS_WEIGHT,
    RULE_RARE_SUPPLIER_WEIGHT,
    RULE_ROUND_AMOUNT_THRESHOLD,
    RULE_HIGH_FREQ_THRESHOLD,
    RULES_ANOMALY_THRESHOLD,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DÉFINITION DES RÈGLES
# ─────────────────────────────────────────────────────────────────────────────

RULES = {
    # Règle 1 : montant globalement très élevé (z-score > seuil)
    "rule_high_amount_global": {
        "condition": lambda df: (df["amount_zscore_global"] > RULE_AMOUNT_HIGH_Z).astype(int),
        "weight": 0.25,
        "description": f"Montant avec z-score global > {RULE_AMOUNT_HIGH_Z}",
    },
    # Règle 2 : montant très élevé par rapport à l'historique utilisateur
    "rule_high_amount_vs_user": {
        "condition": lambda df: (df["amount_iqr_flag_per_user"] == 1).astype(int),
        "weight": 0.20,
        "description": "Montant en dehors du IQR par utilisateur",
    },
    # Règle 3 : transaction le week-end
    "rule_weekend": {
        "condition": lambda df: df["is_weekend"].astype(int),
        "weight": RULE_WEEKEND_WEIGHT,
        "description": "Transaction le week-end",
    },
    # Règle 4 : transaction hors heures de bureau
    "rule_outside_hours": {
        "condition": lambda df: df["is_outside_hours"].astype(int),
        "weight": RULE_OUTSIDE_HOURS_WEIGHT,
        "description": "Transaction hors heures de bureau",
    },
    # Règle 5 : fournisseur rare (peu utilisé)
    "rule_rare_supplier": {
        "condition": lambda df: df["is_rare_supplier"].astype(int),
        "weight": RULE_RARE_SUPPLIER_WEIGHT,
        "description": "Fournisseur utilisé moins de 3 fois",
    },
    # Règle 6 : spike brutal par rapport à la tendance récente
    "rule_rolling_spike": {
        "condition": lambda df: df["rolling_amount_spike_flag"].astype(int),
        "weight": 0.20,
        "description": "Montant x3+ vs moyenne glissante 7 jours",
    },
    # Règle 7 : fréquence journalière anormalement élevée
    "rule_high_daily_freq": {
        "condition": lambda df: (df["tx_daily_count"] > RULE_HIGH_FREQ_THRESHOLD).astype(int),
        "weight": 0.10,
        "description": f"Plus de {RULE_HIGH_FREQ_THRESHOLD} transactions le même jour",
    },
    # Règle 8 : montant rond suspect (multiple exact de 10000)
    "rule_round_amount": {
        "condition": lambda df: (
            (df["amount"] >= RULE_ROUND_AMOUNT_THRESHOLD) &
            (df["amount"] % 10000 == 0)
        ).astype(int),
        "weight": 0.05,
        "description": f"Montant round ≥ {RULE_ROUND_AMOUNT_THRESHOLD:,} XAF",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def run_rule_based(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les règles métier sur le DataFrame préprocessé.

    Args:
        df_processed : DataFrame complet après preprocessing (avec toutes les features).

    Returns:
        DataFrame avec les colonnes de flags et scores ajoutées.
    """
    logger.info(f"📏 Règles métier — {len(RULES)} règles à appliquer")

    result = df_processed.copy()
    total_weight = sum(r["weight"] for r in RULES.values())
    weighted_score = np.zeros(len(result))

    for rule_name, rule_config in RULES.items():
        flag = rule_config["condition"](result)
        result[rule_name] = flag
        weighted_score += flag.values * rule_config["weight"]
        n_flagged = flag.sum()
        logger.info(f"   → {rule_name}: {n_flagged} transactions flaggées ({n_flagged/len(result)*100:.1f}%)")

    # Score normalisé [0,1]
    result["score_rules"] = weighted_score / total_weight

    # Anomalie si score composite > seuil
    result["anomaly_rules"] = (result["score_rules"] >= RULES_ANOMALY_THRESHOLD).astype(int)
    n_anomalies = result["anomaly_rules"].sum()
    logger.info(f"   → {n_anomalies} anomalies détectées par règles ({n_anomalies/len(result)*100:.1f}%)")

    return result


def get_rule_explanations(row: pd.Series) -> list[str]:
    """
    Retourne la liste des règles déclenchées pour une ligne donnée.

    Args:
        row : Ligne du DataFrame avec les flags de règles.

    Returns:
        Liste de descriptions des règles actives.
    """
    explanations = []
    for rule_name, rule_config in RULES.items():
        if rule_name in row and row[rule_name] == 1:
            explanations.append(rule_config["description"])
    return explanations