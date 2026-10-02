"""
model_registry.py
------------------
Registre central des modèles entraînés.

Responsabilités :
    - Charger les données historiques et entraîner tous les modèles (une seule fois)
    - Exposer score_transaction() et score_batch() pour scorer de nouvelles données
    - Garder en mémoire le scaler et les statistiques du preprocessing historique
      pour que les nouvelles transactions soient normalisées de façon cohérente

Cycle de vie :
    1. Au démarrage de l'API → ModelRegistry.initialize(sql_path)
    2. À chaque requête      → ModelRegistry.score_transaction(data) / .score_batch(data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.data_loader import load_expenses_from_sql
from src.preprocessing import preprocess, get_ml_features
from src.models.isolation_forest import IsolationForestDetector
from src.models.lof import LOFDetector
from src.models.dbscan import DBSCANDetector
from src.models.rule_based import run_rule_based, get_rule_explanations
from src.models.fraud_patterns import run_fraud_patterns
from src.config import FINAL_ANOMALY_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Poids des modèles dans le score final
WEIGHT_IF     = 0.25
WEIGHT_LOF    = 0.20
WEIGHT_DBSCAN = 0.15
WEIGHT_RULES  = 0.20
WEIGHT_FRAUD  = 0.20


class ModelRegistry:
    """
    Singleton qui charge les modèles une fois et les garde en mémoire.

    Usage :
        registry = ModelRegistry()
        registry.initialize(Path("data/expenses.sql"))
        result = registry.score_transaction(transaction_dict)
    """

    def __init__(self):
        self.if_detector    : Optional[IsolationForestDetector] = None
        self.lof_detector   : Optional[LOFDetector]             = None
        self.dbscan_detector: Optional[DBSCANDetector]          = None

        self._df_historical : Optional[pd.DataFrame] = None  # données preprocessées historiques
        self._ml_columns    : Optional[list[str]]    = None  # colonnes features ML
        self._is_ready      : bool                   = False
        self._n_train       : int                    = 0

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALISATION
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self, sql_path: Path) -> None:
        """
        Charge les données historiques et entraîne tous les modèles.
        À appeler une seule fois au démarrage de l'API.

        Args:
            sql_path : Chemin vers le fichier SQL historique.
        """
        logger.info("=" * 55)
        logger.info("🚀 Initialisation du ModelRegistry...")
        logger.info("=" * 55)

        # 1. Chargement et preprocessing des données historiques
        df_raw = load_expenses_from_sql(sql_path)
        self._df_historical_raw = df_raw
        df_processed = preprocess(df_raw)
        df_ml = get_ml_features(df_processed)

        self._df_historical = df_processed
        self._ml_columns    = list(df_ml.columns)
        self._n_train       = len(df_ml)

        logger.info(f"📊 Données historiques : {self._n_train:,} transactions, {len(self._ml_columns)} features")

        # 2. Entraînement des modèles ML
        logger.info("🔧 Entraînement des modèles...")

        self.if_detector = IsolationForestDetector()
        self.if_detector.fit(df_ml)

        self.lof_detector = LOFDetector()
        self.lof_detector.fit(df_ml)

        self.dbscan_detector = DBSCANDetector()
        self.dbscan_detector.fit(df_ml)

        self._is_ready = True
        logger.info("✅ ModelRegistry prêt.")

    # ─────────────────────────────────────────────────────────────────────────
    # SCORING — TRANSACTION UNIQUE
    # ─────────────────────────────────────────────────────────────────────────

    def score_transaction(self, transaction: dict) -> dict:
        """
        Score une transaction unique.

        Args:
            transaction : Dictionnaire avec les champs de la transaction
                          (mêmes colonnes que la table expenses).

        Returns:
            Dictionnaire avec score_final, is_anomaly, règles déclenchées, etc.
        """
        self._check_ready()
        df = pd.DataFrame([transaction])
        results = self._score_dataframe(df)
        return results[0]

    # ─────────────────────────────────────────────────────────────────────────
    # SCORING — BATCH
    # ─────────────────────────────────────────────────────────────────────────

    def score_batch(self, transactions: list[dict]) -> dict:
        """
        Score un batch de transactions.

        Args:
            transactions : Liste de dictionnaires (une transaction = un dict).

        Returns:
            Dictionnaire avec la liste des résultats + résumé global.
        """
        self._check_ready()
        df = pd.DataFrame(transactions)
        results = self._score_dataframe(df)

        n_anomalies = sum(1 for r in results if r["is_anomaly"])
        total_amount = sum(r.get("amount", 0) for r in results if r["is_anomaly"])

        return {
            "total":        len(results),
            "n_anomalies":  n_anomalies,
            "anomaly_rate": round(n_anomalies / len(results) * 100, 2),
            "anomaly_amount_total": total_amount,
            "results":      results,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SCORING INTERNE
    # ─────────────────────────────────────────────────────────────────────────

    def _score_dataframe(self, df_raw: pd.DataFrame) -> list[dict]:
        n_new = len(df_raw)

        # Marquer les nouvelles transactions avec un tag temporaire
        df_raw_tagged = df_raw.copy()
        df_raw_tagged["_is_new"] = True

        df_hist_tagged = self._df_historical_raw.copy()
        df_hist_tagged["_is_new"] = False

        # Concaténer historique + nouvelles
        df_combined = pd.concat(
            [df_hist_tagged, df_raw_tagged],
            ignore_index=True
        )

        # Preprocessing complet
        df_processed_full = preprocess(df_combined)
        df_ml_full        = get_ml_features(df_processed_full)

        # Extraire uniquement les nouvelles via le tag (résistant au réordonnancement)
        mask_new = df_processed_full.get("_is_new", pd.Series(False, index=df_processed_full.index)) == True
        df_processed  = df_processed_full[mask_new].reset_index(drop=True)
        df_ml         = df_ml_full[mask_new].reset_index(drop=True)

        # Aligner les colonnes
        df_ml = self._align_columns(df_ml)

        # Modèles ML
        df_if     = self.if_detector.predict(df_ml)
        df_lof    = self.lof_detector.predict(df_ml)
        df_dbscan = self.dbscan_detector.predict(df_ml)

        # Règles + fraude sur le combined, extraire les nouvelles
        df_rules_full = run_rule_based(df_processed_full)
        df_fraud_full = run_fraud_patterns(df_processed_full)

        df_rules = df_rules_full[mask_new.values].reset_index(drop=True)
        df_fraud = df_fraud_full[mask_new.values].reset_index(drop=True)

        # Score combiné
        score_final = (
            WEIGHT_IF     * df_if["score_if"].values          +
            WEIGHT_LOF    * df_lof["score_lof"].values         +
            WEIGHT_DBSCAN * df_dbscan["score_dbscan"].values   +
            WEIGHT_RULES  * df_rules["score_rules"].values     +
            WEIGHT_FRAUD  * df_fraud["fraud_score_combined"].values
        )

        # Anomalie si score >= seuil OU pattern fraude confirmé OU consensus ML OU règle métier critique
        ml_high_scores = (
            (df_if["score_if"].values > 0.70).astype(int) +
            (df_lof["score_lof"].values > 0.70).astype(int) +
            (df_dbscan["score_dbscan"].values > 0.70).astype(int)
        )
        consensus_ml  = ml_high_scores >= 2
        is_fraud      = df_fraud["is_fraud_pattern"].values
        rules_anomaly = df_rules["anomaly_rules"].values if "anomaly_rules" in df_rules.columns else np.zeros(len(df_rules))
        is_anomaly    = (
            (score_final >= FINAL_ANOMALY_THRESHOLD) |
            (is_fraud == 1) |
            (rules_anomaly == 1) |
            consensus_ml
        ).astype(int)

        # Construction des résultats
        rule_cols      = [c for c in df_rules.columns if c.startswith("rule_")]
        df_rules_flags = df_rules[rule_cols].reset_index(drop=True)

        results = []
        for i in range(len(df_processed)):
            row = df_processed.iloc[i]
            rules_triggered = get_rule_explanations(df_rules_flags.iloc[i])
            result = {
                "id":           str(row.get("id", "")),
                "amount":       float(row.get("amount", 0)),
                "currency":     str(row.get("currency", "")),
                "expense_type": str(row.get("expense_type", "")),
                "source":       str(row.get("source", "")),
                "created_at":   str(row.get("created_at", "")),
                "score_final":  round(float(score_final[i]), 4),
                "is_anomaly":   bool(is_anomaly[i]),
                "scores": {
                    "isolation_forest": round(float(df_if["score_if"].iloc[i]), 4),
                    "lof":              round(float(df_lof["score_lof"].iloc[i]), 4),
                    "dbscan":           round(float(df_dbscan["score_dbscan"].iloc[i]), 4),
                    "rules":            round(float(df_rules["score_rules"].iloc[i]), 4),
                    "fraud":            round(float(df_fraud["fraud_score_combined"].iloc[i]), 4),
                },
                "fraud_details": {
                    "splitting":      round(float(df_fraud["fraud_splitting_score"].iloc[i]), 4),
                    "duplicate":      round(float(df_fraud["fraud_duplicate_score"].iloc[i]), 4),
                    "ghost_supplier": round(float(df_fraud["fraud_ghost_supplier_score"].iloc[i]), 4),
                    "inflation":      round(float(df_fraud["fraud_inflation_score"].iloc[i]), 4),
                },
                "rules_triggered":  rules_triggered,
                "is_fraud_pattern": bool(df_fraud["is_fraud_pattern"].iloc[i]),
            }
            results.append(result)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _align_columns(self, df_ml: pd.DataFrame) -> pd.DataFrame:
        """
        Aligne les colonnes du DataFrame sur celles vues à l'entraînement.
        Ajoute les colonnes manquantes avec 0, ignore les colonnes inconnues.
        """
        for col in self._ml_columns:
            if col not in df_ml.columns:
                df_ml[col] = 0.0
        return df_ml[self._ml_columns]

    def _check_ready(self):
        if not self._is_ready:
            raise RuntimeError(
                "ModelRegistry non initialisé. "
                "Appelle initialize(sql_path) avant de scorer."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # INFOS
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def info(self) -> dict:
        return {
            "ready":          self._is_ready,
            "n_train":        self._n_train,
            "n_features":     len(self._ml_columns) if self._ml_columns else 0,
            "models":         ["IsolationForest", "LOF", "DBSCAN", "Rules", "FraudPatterns"],
            "threshold":      FINAL_ANOMALY_THRESHOLD,
        }


# ── Instance globale (partagée par toute l'API) ───────────────────────────────
registry = ModelRegistry()