"""
test_models.py
--------------
Tests unitaires pour les modèles d'apprentissage non supervisé :
Isolation Forest, Local Outlier Factor (LOF) et DBSCAN.
"""

import pytest
import pandas as pd
import numpy as np

from src.models.isolation_forest import IsolationForestDetector, run_isolation_forest
from src.models.lof import LOFDetector, run_lof
from src.models.dbscan import DBSCANDetector, run_dbscan


class TestIsolationForest:
    """Tests pour le détecteur Isolation Forest."""

    def test_fit_and_predict_contract(self, sample_ml_features):
        detector = IsolationForestDetector(contamination=0.05, n_estimators=50, random_state=42)
        detector.fit(sample_ml_features)
        res = detector.predict(sample_ml_features)

        assert "score_if" in res.columns
        assert "anomaly_if" in res.columns
        assert res["score_if"].between(0.0, 1.0).all()
        assert set(res["anomaly_if"].unique()).issubset({0, 1})
        assert len(res) == len(sample_ml_features)

    def test_predict_before_fit_raises(self, sample_ml_features):
        detector = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="appelle fit.*avant predict"):
            detector.predict(sample_ml_features)

    def test_run_isolation_forest_convenience(self, sample_ml_features):
        res = run_isolation_forest(sample_ml_features)
        assert "score_if" in res.columns
        assert "anomaly_if" in res.columns


class TestLOF:
    """Tests pour le détecteur Local Outlier Factor."""

    def test_fit_and_predict_contract(self, sample_ml_features):
        detector = LOFDetector(n_neighbors=10, contamination=0.05)
        detector.fit(sample_ml_features)
        res = detector.predict(sample_ml_features)

        assert "score_lof" in res.columns
        assert "anomaly_lof" in res.columns
        assert res["score_lof"].between(0.0, 1.0).all()
        assert set(res["anomaly_lof"].unique()).issubset({0, 1})

    def test_predict_before_fit_raises(self, sample_ml_features):
        detector = LOFDetector()
        with pytest.raises(RuntimeError, match="appelle fit.*avant predict"):
            detector.predict(sample_ml_features)

    def test_run_lof_convenience(self, sample_ml_features):
        res = run_lof(sample_ml_features)
        assert "score_lof" in res.columns
        assert "anomaly_lof" in res.columns


class TestDBSCAN:
    """Tests pour le détecteur DBSCAN."""

    def test_fit_and_predict_contract(self, sample_ml_features):
        detector = DBSCANDetector(min_samples=3)
        detector.fit(sample_ml_features)
        res = detector.predict(sample_ml_features)

        assert "score_dbscan" in res.columns
        assert "anomaly_dbscan" in res.columns
        assert res["score_dbscan"].between(0.0, 1.0).all()
        assert set(res["anomaly_dbscan"].unique()).issubset({0, 1})

    def test_predict_before_fit_raises(self, sample_ml_features):
        detector = DBSCANDetector()
        with pytest.raises(RuntimeError, match="appelle fit.*avant predict"):
            detector.predict(sample_ml_features)

    def test_run_dbscan_convenience(self, sample_ml_features):
        res = run_dbscan(sample_ml_features)
        assert "score_dbscan" in res.columns
        assert "anomaly_dbscan" in res.columns
