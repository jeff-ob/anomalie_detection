"""
test_rules.py
-------------
Tests unitaires pour le moteur de règles métier déterministes (src/models/rule_based.py).
"""

import pytest
import pandas as pd
import numpy as np

from src.models.rule_based import RULES, run_rule_based, get_rule_explanations
from src.config import RULE_ROUND_AMOUNT_THRESHOLD, RULE_HIGH_FREQ_THRESHOLD


class TestDeterministicRules:
    """Tests ciblés pour chaque règle métier individuelle."""

    def test_rule_high_amount_global(self):
        df = pd.DataFrame({"amount_zscore_global": [2.5, 3.1, 4.0, 1.0]})
        flags = RULES["rule_high_amount_global"]["condition"](df)
        assert flags.tolist() == [0, 1, 1, 0]

    def test_rule_weekend(self):
        df = pd.DataFrame({"is_weekend": [0, 1, 1, 0]})
        flags = RULES["rule_weekend"]["condition"](df)
        assert flags.tolist() == [0, 1, 1, 0]

    def test_rule_outside_hours(self):
        df = pd.DataFrame({"is_outside_hours": [1, 0, 1, 0]})
        flags = RULES["rule_outside_hours"]["condition"](df)
        assert flags.tolist() == [1, 0, 1, 0]

    def test_rule_rare_supplier(self):
        df = pd.DataFrame({"is_rare_supplier": [1, 0, 0, 1]})
        flags = RULES["rule_rare_supplier"]["condition"](df)
        assert flags.tolist() == [1, 0, 0, 1]

    def test_rule_high_daily_freq(self):
        limit = RULE_HIGH_FREQ_THRESHOLD
        df = pd.DataFrame({"tx_daily_count": [limit - 1, limit, limit + 1, limit + 5]})
        flags = RULES["rule_high_daily_freq"]["condition"](df)
        assert flags.tolist() == [0, 0, 1, 1]

    def test_rule_round_amount(self):
        limit = RULE_ROUND_AMOUNT_THRESHOLD
        df = pd.DataFrame({"amount": [50_000, limit, limit + 1234, limit + 20_000]})
        flags = RULES["rule_round_amount"]["condition"](df)
        # 50_000 < limit -> 0
        # limit (ex: 100_000) -> 1
        # limit + 1234 -> 0 (pas un multiple de 10_000)
        # limit + 20_000 -> 1 (multiple de 10_000)
        assert flags.tolist() == [0, 1, 0, 1]


class TestRuleEngineIntegration:
    """Tests d'intégration du moteur de règles sur DataFrame préprocessé."""

    def test_run_rule_based_outputs(self, sample_processed_df):
        res = run_rule_based(sample_processed_df)

        for rule_name in RULES.keys():
            assert rule_name in res.columns

        assert "score_rules" in res.columns
        assert "anomaly_rules" in res.columns
        assert res["score_rules"].between(0.0, 1.0).all()
        assert set(res["anomaly_rules"].unique()).issubset({0, 1})

    def test_get_rule_explanations(self):
        row = pd.Series({
            "rule_weekend": 1,
            "rule_outside_hours": 1,
            "rule_high_amount_global": 0,
        })
        explanations = get_rule_explanations(row)
        assert len(explanations) == 2
        assert "Transaction le week-end" in explanations
        assert "Transaction hors heures de bureau" in explanations
