"""
test_fraud_patterns.py
----------------------
Tests unitaires pour les typologies avancées de fraude forensic (src/models/fraud_patterns.py).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.fraud_patterns import (
    detect_splitting,
    detect_duplicates,
    detect_ghost_supplier,
    detect_progressive_inflation,
    run_fraud_patterns,
)


class TestFraudTypologies:
    """Tests des 4 typologies de fraude forensic."""

    def test_detect_duplicates(self):
        t0 = datetime(2026, 6, 1, 10, 0, 0)
        df = pd.DataFrame({
            "user_id": ["USR_01", "USR_01", "USR_02"],
            "amount": [150_000.0, 150_000.0, 150_000.0],
            "created_at": [t0, t0 + timedelta(hours=2), t0 + timedelta(hours=3)],
        })
        scores = detect_duplicates(df, window_hours=48)
        # USR_01 a deux montants identiques en 2h -> scores[1] > 0
        assert scores.iloc[1] > 0
        # USR_02 est un utilisateur différent -> score = 0
        assert scores.iloc[2] == 0.0

    def test_detect_ghost_supplier(self):
        # 10 transactions avec SUPP_01 à montant normal, 1 transaction avec SUPP_GHOST à montant élevé
        suppliers = ["SUPP_01"] * 10 + ["SUPP_GHOST"]
        amounts = [10_000.0] * 10 + [5_000_000.0]
        df = pd.DataFrame({
            "supplier_id": suppliers,
            "amount": amounts,
        })
        scores = detect_ghost_supplier(df)
        assert scores.iloc[10] > 0.0
        assert (scores.iloc[:10] == 0.0).all()

    def test_detect_splitting(self):
        t0 = datetime(2026, 6, 1, 10, 0, 0)
        # 3 transactions pour USR_01 et SUPP_01 dans la même heure
        df = pd.DataFrame({
            "user_id": ["USR_01", "USR_01", "USR_01", "USR_OTHER"],
            "supplier_id": ["SUPP_01", "SUPP_01", "SUPP_01", "SUPP_OTHER"],
            "amount": [200_000.0, 200_000.0, 200_000.0, 10_000.0],
            "created_at": [t0, t0 + timedelta(minutes=10), t0 + timedelta(minutes=20), t0],
        })
        scores = detect_splitting(df, window_hours=24)
        assert (scores > 0).any()

    def test_detect_progressive_inflation(self):
        t0 = datetime(2026, 6, 1, 10, 0, 0)
        # 6 transactions à montants strictement croissants
        df = pd.DataFrame({
            "user_id": ["USR_INFLATION"] * 6,
            "amount": [10_000.0, 25_000.0, 50_000.0, 100_000.0, 200_000.0, 450_000.0],
            "created_at": [t0 + timedelta(days=i) for i in range(6)],
        })
        scores = detect_progressive_inflation(df, min_tx=5)
        # Corrélation Spearman parfaite = 1.0 -> doit flagger
        assert (scores > 0).any()


class TestFraudOrchestrator:
    """Tests d'intégration pour run_fraud_patterns."""

    def test_run_fraud_patterns_pipeline(self, sample_processed_df):
        res = run_fraud_patterns(sample_processed_df)

        expected_cols = [
            "fraud_splitting_score",
            "fraud_duplicate_score",
            "fraud_ghost_supplier_score",
            "fraud_inflation_score",
            "fraud_score_combined",
            "is_fraud_pattern",
        ]
        for col in expected_cols:
            assert col in res.columns

        assert res["fraud_score_combined"].between(0.0, 1.0).all()
        assert set(res["is_fraud_pattern"].unique()).issubset({0, 1})
