"""
fraud_patterns.py
------------------
Détection de patterns de fraude spécifiques basés sur la logique métier.

Patterns détectés :
    1. Fractionnement   — plusieurs petites transactions vers le même fournisseur
                          dans une courte fenêtre de temps, dont la somme est élevée
    2. Doublons suspects — même montant + même fournisseur/user dans un délai court
    3. Fournisseur fantôme — fournisseur vu une seule fois, montant élevé vs médiane
    4. Inflation progressive — montants qui augmentent régulièrement pour un même user/supplier

La logique est basée sur des seuils statistiques dérivés des données elles-mêmes
(percentiles, médiane), pas des valeurs arbitraires.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FRACTIONNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def detect_splitting(df: pd.DataFrame, window_hours: int = 24) -> pd.Series:
    """
    Détecte le fractionnement : plusieurs transactions du même user vers le même
    fournisseur dans une fenêtre de temps, dont la somme dépasse le 75e percentile
    des montants globaux.

    Optimisé : Regroupement par (user_id, supplier_id) pour une complexité quasi-linéaire.
    """
    scores = pd.Series(0.0, index=df.index)
    amount_p75 = df["amount"].quantile(0.75)

    mask_supplier = df["supplier_id"].notna()
    df_sup = df[mask_supplier]

    if df_sup.empty:
        logger.info("  → Fractionnement : aucun fournisseur renseigné, skip")
        return scores

    window = pd.Timedelta(hours=window_hours)

    for (user_id, supp_id), group in df_sup.groupby(["user_id", "supplier_id"]):
        if len(group) < 3:
            continue
        group_sorted = group.sort_values("created_at")
        times = group_sorted["created_at"].values
        amounts = group_sorted["amount"].values
        indices = group_sorted.index.values

        for i in range(len(group_sorted)):
            t_curr = times[i]
            t_min = t_curr - window

            sub_amounts = []
            for j in range(i - 1, -1, -1):
                if times[j] < t_min:
                    break
                sub_amounts.append(amounts[j])

            if len(sub_amounts) >= 2:
                total = sum(sub_amounts) + amounts[i]
                if total > amount_p75:
                    count_score = min(len(sub_amounts) / 10.0, 1.0)
                    amount_score = min(total / (amount_p75 * 3), 1.0)
                    scores[indices[i]] = (count_score + amount_score) / 2

    n_flagged = (scores > 0).sum()
    logger.info(f"  → Fractionnement : {n_flagged} transactions suspectes")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOUBLONS SUSPECTS
# ─────────────────────────────────────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame, window_hours: int = 48) -> pd.Series:
    """
    Détecte les doublons suspects : même user + même montant (±1%) dans
    une fenêtre de temps courte.

    Optimisé : Regroupement par user_id avec balayage temporel local.
    """
    scores = pd.Series(0.0, index=df.index)
    window = pd.Timedelta(hours=window_hours)

    for user_id, group in df.groupby("user_id"):
        if len(group) < 2:
            continue
        group_sorted = group.sort_values("created_at")
        times = group_sorted["created_at"].values
        amounts = group_sorted["amount"].values
        indices = group_sorted.index.values

        for i in range(len(group_sorted)):
            t_curr = times[i]
            amt_curr = amounts[i]
            idx_curr = indices[i]
            t_min = t_curr - window

            for j in range(i - 1, -1, -1):
                if times[j] < t_min:
                    break
                amt_prev = amounts[j]
                if abs(amt_curr - amt_prev) / (amt_curr + 1) < 0.01:
                    time_diff_hours = (t_curr - times[j]) / np.timedelta64(1, "h")
                    recency_score = 1.0 - (time_diff_hours / window_hours)
                    scores[idx_curr] = max(scores[idx_curr], recency_score)
                    break

    n_flagged = (scores > 0).sum()
    logger.info(f"  → Doublons suspects : {n_flagged} transactions suspectes")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 3. FOURNISSEUR FANTÔME
# ─────────────────────────────────────────────────────────────────────────────

def detect_ghost_supplier(df: pd.DataFrame) -> pd.Series:
    """
    Détecte les fournisseurs fantômes : fournisseur vu exactement une fois
    avec un montant bien supérieur à la médiane des autres fournisseurs.

    Logique :
        - Fournisseur n'apparaît qu'une seule fois dans tout l'historique
        - Son montant dépasse le 90e percentile global
        - Score proportionnel au ratio montant / p90

    Args:
        df : DataFrame préprocessé.

    Returns:
        Série de scores [0,1] indexée comme df.
    """
    scores = pd.Series(0.0, index=df.index)

    if "supplier_id" not in df.columns:
        return scores

    # Comptage des apparitions par fournisseur
    supplier_counts = df["supplier_id"].value_counts()
    single_use_suppliers = set(supplier_counts[supplier_counts == 1].index)

    amount_p90 = df["amount"].quantile(0.90)

    mask = (
        df["supplier_id"].isin(single_use_suppliers) &
        df["supplier_id"].notna() &
        (df["amount"] > amount_p90)
    )

    if mask.any():
        ratio = df.loc[mask, "amount"] / amount_p90
        scores[mask] = ratio.clip(upper=1.0)

    n_flagged = (scores > 0).sum()
    logger.info(f"  → Fournisseurs fantômes : {n_flagged} transactions suspectes")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFLATION PROGRESSIVE
# ─────────────────────────────────────────────────────────────────────────────

def detect_progressive_inflation(df: pd.DataFrame, min_tx: int = 5) -> pd.Series:
    """
    Détecte une inflation progressive des montants pour un même user.
    Signe potentiel de test des limites de détection.

    Logique :
        - Pour chaque user avec au moins `min_tx` transactions
        - Calcule la corrélation de Spearman entre rang temporel et montant
        - Une corrélation > 0.7 sur les 10 dernières transactions = tendance haussière
        - Score = corrélation normalisée

    Args:
        df     : DataFrame préprocessé trié par created_at.
        min_tx : Nombre minimum de transactions pour calculer la tendance.

    Returns:
        Série de scores [0,1] indexée comme df.
    """
    from scipy.stats import spearmanr

    scores = pd.Series(0.0, index=df.index)
    df_work = df[["user_id", "amount", "created_at"]].copy().sort_values("created_at")

    for user_id, group in df_work.groupby("user_id"):
        if len(group) < min_tx:
            continue

        # Fenêtre glissante : 10 dernières transactions
        window_size = min(10, len(group))
        recent = group.tail(window_size)

        ranks   = np.arange(len(recent))
        amounts = recent["amount"].values

        if amounts.std() == 0:
            continue

        corr, pvalue = spearmanr(ranks, amounts)

        # Corrélation forte ET significative
        if corr > 0.70 and pvalue < 0.10:
            # Score proportionnel à la corrélation
            score = (corr - 0.70) / 0.30  # normalise [0.7, 1.0] → [0, 1]
            scores.loc[recent.index] = score

    n_flagged = (scores > 0).sum()
    logger.info(f"  → Inflation progressive : {n_flagged} transactions suspectes")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR
# ─────────────────────────────────────────────────────────────────────────────

def run_fraud_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique tous les détecteurs de fraude et agrège les scores.

    Args:
        df : DataFrame préprocessé complet.

    Returns:
        DataFrame avec les colonnes de scores fraude ajoutées :
            - fraud_splitting_score
            - fraud_duplicate_score
            - fraud_ghost_supplier_score
            - fraud_inflation_score
            - fraud_score_combined  : max des 4 scores (le pire pattern compte)
            - is_fraud_pattern      : 1 si fraud_score_combined > 0.40
    """
    logger.info("🕵️  Détection des patterns de fraude...")

    result = df.copy().reset_index(drop=True)

    result["fraud_splitting_score"]       = detect_splitting(result).values
    result["fraud_duplicate_score"]       = detect_duplicates(result).values
    result["fraud_ghost_supplier_score"]  = detect_ghost_supplier(result).values
    result["fraud_inflation_score"]       = detect_progressive_inflation(result).values

    # Score combiné : on prend le MAX (le pattern le plus grave suffit)
    result["fraud_score_combined"] = result[[
        "fraud_splitting_score",
        "fraud_duplicate_score",
        "fraud_ghost_supplier_score",
        "fraud_inflation_score",
    ]].max(axis=1)

    # Seuil : 40e percentile des scores > 0 (dynamique, pas arbitraire)
    nonzero_scores = result["fraud_score_combined"][result["fraud_score_combined"] > 0]
    if len(nonzero_scores) > 0:
        threshold = nonzero_scores.quantile(0.60)
    else:
        threshold = 0.40

    result["is_fraud_pattern"] = (result["fraud_score_combined"] >= threshold).astype(int)

    n_fraud = result["is_fraud_pattern"].sum()
    logger.info(
        f"✅ Patterns fraude : {n_fraud} transactions suspectes "
        f"(seuil dynamique = {threshold:.3f})"
    )
    return result