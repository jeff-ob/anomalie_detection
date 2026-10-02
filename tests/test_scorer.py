"""
test_scorer.py
--------------
Tests d'intégration de bout en bout pour le moteur de scoring et d'arbitrage (src/scorer.py).
"""

import pytest
import pandas as pd
import numpy as np

from src.scorer import score_anomalies, get_top_anomalies, get_anomaly_summary


class TestScorerPipeline:
    """Tests d'intégration du pipeline complet de détection."""

    def test_score_anomalies_full_pipeline(self, sample_raw_df):
        scored_df = score_anomalies(sample_raw_df)

        assert isinstance(scored_df, pd.DataFrame)
        assert len(scored_df) > 0

        # Vérification des colonnes de score
        score_cols = ["score_if", "score_lof", "score_dbscan", "score_rules", "score_fraud", "score_final"]
        for col in score_cols:
            assert col in scored_df.columns
            assert scored_df[col].between(0.0, 1.0).all()

        # Vérification des flags
        flag_cols = ["anomaly_if", "anomaly_lof", "anomaly_dbscan", "anomaly_rules", "is_fraud_pattern", "is_anomaly"]
        for col in flag_cols:
            assert col in scored_df.columns
            assert set(scored_df[col].unique()).issubset({0, 1})

        # Explications textuelles
        assert "rules_triggered" in scored_df.columns

    def test_get_top_anomalies(self, sample_raw_df):
        scored_df = score_anomalies(sample_raw_df)
        top5 = get_top_anomalies(scored_df, n=5)

        assert len(top5) <= 5
        # Décroissance stricte ou égale du score final
        assert top5["score_final"].is_monotonic_decreasing

    def test_get_anomaly_summary(self, sample_raw_df):
        scored_df = score_anomalies(sample_raw_df)
        summary = get_anomaly_summary(scored_df)

        assert isinstance(summary, dict)
        required_keys = [
            "total_transactions",
            "total_anomalies",
            "anomaly_rate_pct",
            "avg_score_final",
            "flagged_if",
            "flagged_lof",
            "flagged_rules",
            "flagged_fraud",
        ]
        for k in required_keys:
            assert k in summary

        assert summary["total_transactions"] == len(scored_df)
        assert 0.0 <= summary["anomaly_rate_pct"] <= 100.0
