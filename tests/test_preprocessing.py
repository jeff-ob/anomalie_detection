"""
test_preprocessing.py
---------------------
Tests unitaires et d'intégration pour le pipeline de nettoyage et de feature engineering (src/preprocessing.py).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.preprocessing import (
    clean_dataframe,
    add_temporal_features,
    add_amount_features,
    add_user_behavior_features,
    add_supplier_features,
    encode_categoricals,
    get_ml_features,
    preprocess,
    ML_FEATURE_COLUMNS,
)


class TestDataCleaning:
    """Tests du nettoyage initial des données."""

    def test_clean_dataframe_removes_invalid_rows(self, sample_raw_df):
        initial_count = len(sample_raw_df)
        cleaned = clean_dataframe(sample_raw_df)

        # Doit supprimer le soft deleted (i=95) et le montant négatif (i=96)
        assert len(cleaned) < initial_count
        assert (cleaned["amount"] <= 0).sum() == 0
        assert cleaned["amount"].isna().sum() == 0
        assert cleaned["created_at"].isna().sum() == 0
        if "deleted_at" in cleaned.columns:
            assert cleaned["deleted_at"].isna().all()

    def test_clean_dataframe_deduplicates_id(self):
        df_dupl = pd.DataFrame({
            "id": ["EXP_1", "EXP_1", "EXP_2"],
            "amount": [100.0, 100.0, 200.0],
            "created_at": [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-02")],
            "deleted_at": [None, None, None],
        })
        cleaned = clean_dataframe(df_dupl)
        assert len(cleaned) == 2
        assert cleaned["id"].tolist() == ["EXP_1", "EXP_2"]


class TestFeatureEngineering:
    """Tests de génération des features temporelles, montants, utilisateurs et fournisseurs."""

    def test_add_temporal_features(self, sample_raw_df):
        cleaned = clean_dataframe(sample_raw_df)
        df_temp = add_temporal_features(cleaned)

        expected_cols = ["hour_of_day", "day_of_week", "is_weekend", "is_outside_hours"]
        for col in expected_cols:
            assert col in df_temp.columns

        assert df_temp["hour_of_day"].between(0, 23).all()
        assert df_temp["day_of_week"].between(0, 6).all()
        assert set(df_temp["is_weekend"].unique()).issubset({0, 1})
        assert set(df_temp["is_outside_hours"].unique()).issubset({0, 1})

    def test_add_amount_features(self, sample_raw_df):
        cleaned = clean_dataframe(sample_raw_df)
        df_amt = add_amount_features(cleaned)

        expected_cols = [
            "amount_log",
            "amount_zscore_global",
            "amount_zscore_per_user",
            "amount_iqr_flag_global",
            "amount_iqr_flag_per_user",
        ]
        for col in expected_cols:
            assert col in df_amt.columns

        # Le montant extrême (15M) doit avoir un z-score élevé
        outlier_row = df_amt[df_amt["amount"] == 15_000_000.0]
        assert len(outlier_row) == 1
        assert outlier_row["amount_zscore_global"].iloc[0] > 2.0

    def test_add_supplier_features(self, sample_raw_df):
        cleaned = clean_dataframe(sample_raw_df)
        df_supp = add_supplier_features(cleaned)

        assert "supplier_usage_count" in df_supp.columns
        assert "is_rare_supplier" in df_supp.columns
        assert set(df_supp["is_rare_supplier"].unique()).issubset({0, 1})

    def test_encode_categoricals(self, sample_raw_df):
        cleaned = clean_dataframe(sample_raw_df)
        df_encoded = encode_categoricals(cleaned)

        for col in ["status_encoded", "expense_type_encoded", "source_encoded", "currency_encoded"]:
            assert col in df_encoded.columns
            assert pd.api.types.is_numeric_dtype(df_encoded[col])


class TestMLFeatureMatrixAndPipeline:
    """Tests sur l'extraction de la matrice ML et l'orchestration du preprocessing."""

    def test_get_ml_features_guarantees_zero_nans(self, sample_processed_df):
        X = get_ml_features(sample_processed_df)

        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(sample_processed_df)
        # Règle critique : 0 résidu NaN pour éviter tout plantage des modèles scikit-learn
        assert X.isna().sum().sum() == 0

    def test_get_ml_features_handles_all_nan_columns(self):
        # Créer un DataFrame contenant une colonne entièrement NaN
        df = pd.DataFrame({
            "amount": [100.0, 200.0],
            "payment_delay_days": [np.nan, np.nan],
            "hour_of_day": [10, 14],
        })
        X = get_ml_features(df)
        assert X.isna().sum().sum() == 0
        assert (X["payment_delay_days"] == 0.0).all()

    def test_preprocess_end_to_end(self, sample_raw_df):
        df_proc = preprocess(sample_raw_df)

        assert isinstance(df_proc, pd.DataFrame)
        assert len(df_proc) > 0
        assert "amount_log" in df_proc.columns
        assert "is_rare_supplier" in df_proc.columns
        assert "hour_deviation_from_user_norm" in df_proc.columns
