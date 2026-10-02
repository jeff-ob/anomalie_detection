"""
scorer.py
----------
Orchestrateur du pipeline complet de détection d'anomalies.

Combine :
    - Isolation Forest  (anomalies multi-dimensionnelles)
    - LOF               (anomalies de densité locale)
    - DBSCAN            (clusters suspects, points isolés)
    - Règles métier     (règles explicites + comportement utilisateur)
    - Patterns de fraude (fractionnement, doublons, fournisseurs fantômes, inflation)

Score final = combinaison pondérée de tous les modèles.
"""

import pandas as pd
import numpy as np

from src.preprocessing import preprocess, get_ml_features
from src.models.isolation_forest import run_isolation_forest
from src.models.lof import run_lof
from src.models.dbscan import run_dbscan
from src.models.rule_based import run_rule_based, get_rule_explanations
from src.models.fraud_patterns import run_fraud_patterns
from src.config import (
    SCORER_WEIGHT_IF,
    SCORER_WEIGHT_LOF,
    SCORER_WEIGHT_RULES,
    FINAL_ANOMALY_THRESHOLD,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Poids des modèles (somme = 1.0)
WEIGHT_IF     = 0.25
WEIGHT_LOF    = 0.20
WEIGHT_DBSCAN = 0.15
WEIGHT_RULES  = 0.20
WEIGHT_FRAUD  = 0.20


def score_anomalies(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet : du DataFrame brut au score d anomalie final.

    Args:
        df_raw : DataFrame issu du data_loader.

    Returns:
        DataFrame avec scores, flags et explications.
    """
    logger.info("🚀 Démarrage du pipeline de scoring anomalies (v2)...")

    # 1. Preprocessing (inclut normalisation historique + profil user)
    df_processed = preprocess(df_raw)
    df_ml        = get_ml_features(df_processed)

    logger.info(f"   → {len(df_processed)} lignes à scorer, {df_ml.shape[1]} features ML")

    # 2. Modèles ML
    df_if    = run_isolation_forest(df_ml)
    df_lof   = run_lof(df_ml)
    df_dbscan = run_dbscan(df_ml)

    # 3. Règles métier (sur df_processed avec toutes les features)
    df_rules = run_rule_based(df_processed)

    # 4. Patterns de fraude
    df_fraud = run_fraud_patterns(df_processed)

    # 5. Score combiné pondéré
    score_final = (
        WEIGHT_IF     * df_if["score_if"].values          +
        WEIGHT_LOF    * df_lof["score_lof"].values         +
        WEIGHT_DBSCAN * df_dbscan["score_dbscan"].values   +
        WEIGHT_RULES  * df_rules["score_rules"].values     +
        WEIGHT_FRAUD  * df_fraud["fraud_score_combined"].values
    )

    is_fraud = df_fraud["is_fraud_pattern"].values
    is_anomaly = (
                    (score_final >= FINAL_ANOMALY_THRESHOLD) | (is_fraud == 1)
                ).astype(int)

    # 6. Construction du DataFrame résultat
    id_cols = [
        "id", "company_id", "user_id", "shop_id", "supplier_id",
        "reference", "amount", "currency", "source", "expense_type",
        "status", "created_at", "due_date", "paid_at",
    ]
    id_cols_present = [c for c in id_cols if c in df_processed.columns]
    result = df_processed[id_cols_present].copy().reset_index(drop=True)

    # Scores modèles
    result["score_if"]     = np.round(df_if["score_if"].values, 4)
    result["score_lof"]    = np.round(df_lof["score_lof"].values, 4)
    result["score_dbscan"] = np.round(df_dbscan["score_dbscan"].values, 4)
    result["score_rules"]  = np.round(df_rules["score_rules"].values, 4)
    result["score_fraud"]  = np.round(df_fraud["fraud_score_combined"].values, 4)
    result["score_final"]  = np.round(score_final, 4)

    # Flags individuels
    result["anomaly_if"]     = df_if["anomaly_if"].values
    result["anomaly_lof"]    = df_lof["anomaly_lof"].values
    result["anomaly_dbscan"] = df_dbscan["anomaly_dbscan"].values
    result["anomaly_rules"]  = df_rules["anomaly_rules"].values
    result["is_fraud_pattern"] = df_fraud["is_fraud_pattern"].values
    result["is_anomaly"]     = is_anomaly

    # Détail des patterns de fraude
    result["fraud_splitting"]      = np.round(df_fraud["fraud_splitting_score"].values, 4)
    result["fraud_duplicate"]      = np.round(df_fraud["fraud_duplicate_score"].values, 4)
    result["fraud_ghost_supplier"] = np.round(df_fraud["fraud_ghost_supplier_score"].values, 4)
    result["fraud_inflation"]      = np.round(df_fraud["fraud_inflation_score"].values, 4)

    # Explications textuelles des règles déclenchées
    rule_cols = [c for c in df_rules.columns if c.startswith("rule_")]
    df_rules_flags = df_rules[rule_cols].reset_index(drop=True)
    result["rules_triggered"] = df_rules_flags.apply(
        lambda row: get_rule_explanations(row), axis=1
    ).apply(lambda lst: " | ".join(lst) if lst else "")

    n_anomalies = is_anomaly.sum()
    logger.info(
        f"✅ Scoring terminé : {n_anomalies} anomalies ({n_anomalies/len(result)*100:.1f}%) "
        f"sur {len(result)} transactions"
    )
    logger.info(f"   → Score moyen : {score_final.mean():.3f} | max : {score_final.max():.3f}")

    return result


def get_top_anomalies(df_scored: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df_scored.nlargest(n, "score_final").reset_index(drop=True)


def get_anomaly_summary(df_scored: pd.DataFrame) -> dict:
    anomalies = df_scored[df_scored["is_anomaly"] == 1]

    summary = {
        "total_transactions":   len(df_scored),
        "total_anomalies":      len(anomalies),
        "anomaly_rate_pct":     round(len(anomalies) / len(df_scored) * 100, 2),
        "avg_score_final":      round(df_scored["score_final"].mean(), 4),
        "avg_score_anomalies":  round(anomalies["score_final"].mean(), 4) if len(anomalies) > 0 else 0,
        # Par modèle
        "flagged_if":      int(df_scored.get("anomaly_if", pd.Series(0)).sum()),
        "flagged_lof":     int(df_scored.get("anomaly_lof", pd.Series(0)).sum()),
        "flagged_dbscan":  int(df_scored.get("anomaly_dbscan", pd.Series(0)).sum()),
        "flagged_rules":   int(df_scored.get("anomaly_rules", pd.Series(0)).sum()),
        "flagged_fraud":   int(df_scored.get("is_fraud_pattern", pd.Series(0)).sum()),
        # Consensus
        "consensus_3_models": int((
            (df_scored.get("anomaly_if", 0) == 1) &
            (df_scored.get("anomaly_lof", 0) == 1) &
            (df_scored.get("anomaly_rules", 0) == 1)
        ).sum()),
        # Fraude détaillée
        "fraud_splitting":      int((df_scored.get("fraud_splitting", 0) > 0).sum()),
        "fraud_duplicates":     int((df_scored.get("fraud_duplicate", 0) > 0).sum()),
        "fraud_ghost_supplier": int((df_scored.get("fraud_ghost_supplier", 0) > 0).sum()),
        "fraud_inflation":      int((df_scored.get("fraud_inflation", 0) > 0).sum()),
    }

    if "amount" in df_scored.columns and len(anomalies) > 0:
        summary["anomaly_amount_total"] = int(anomalies["amount"].sum())
        summary["anomaly_amount_max"]   = int(anomalies["amount"].max())
        summary["anomaly_amount_mean"]  = int(anomalies["amount"].mean())

    return summary