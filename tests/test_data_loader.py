"""
test_data_loader.py
-------------------
Tests unitaires pour le chargeur et parser SQL de données (src/data_loader.py).
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import (
    _extract_column_names,
    _split_row_values,
    _parse_value,
    load_expenses_from_sql,
)


class TestDataLoaderInternalParsers:
    """Tests des fonctions internes de découpage et de typage SQL."""

    def test_extract_column_names_valid(self):
        sql = "INSERT INTO `expenses` (`id`, `amount`, `user_id`) VALUES ('1', 100, 'U1');"
        cols = _extract_column_names(sql)
        assert cols == ["id", "amount", "user_id"]

    def test_extract_column_names_without_backticks(self):
        sql = "INSERT INTO expenses (id, amount, user_id) VALUES ('1', 100, 'U1');"
        cols = _extract_column_names(sql)
        assert cols == ["id", "amount", "user_id"]

    def test_extract_column_names_invalid_raises_value_error(self):
        sql = "SELECT * FROM expenses WHERE id = 1;"
        with pytest.raises(ValueError, match="Impossible de trouver la liste de colonnes"):
            _extract_column_names(sql)

    def test_split_row_values_simple(self):
        row_str = "1, 'test', NULL, 42.5"
        vals = _split_row_values(row_str)
        assert vals == ["1", "'test'", "NULL", "42.5"]

    def test_split_row_values_with_json_and_commas(self):
        row_str = "'EXP_001', '{\"user\": \"john, doe\", \"qty\": 2}', NULL, 1500"
        vals = _split_row_values(row_str)
        assert len(vals) == 4
        assert vals[0] == "'EXP_001'"
        assert vals[1] == "'{\"user\": \"john, doe\", \"qty\": 2}'"
        assert vals[2] == "NULL"
        assert vals[3] == "1500"

    def test_split_row_values_escaped_quotes(self):
        row_str = r"'O\'Reilly', 'simple string'"
        vals = _split_row_values(row_str)
        assert len(vals) == 2
        assert r"O\'Reilly" in vals[0]

    def test_parse_value_types(self):
        assert _parse_value("NULL") is None
        assert _parse_value("null") is None
        assert _parse_value("'hello'") == "hello"
        assert _parse_value("123") == 123
        assert _parse_value("123.45") == 123.45
        assert _parse_value("'{\"status\": \"ok\"}'") == {"status": "ok"}
        assert _parse_value("'[\"a\", \"b\"]'") == ["a", "b"]


class TestDataLoaderLoadExpenses:
    """Tests d'intégration pour load_expenses_from_sql."""

    def test_load_expenses_from_temp_file(self, sample_sql_file: Path):
        df = load_expenses_from_sql(sample_sql_file, validate_columns=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "id" in df.columns
        assert "amount" in df.columns
        assert df["id"].tolist() == ["EXP_00001", "EXP_00002", "EXP_00003"]
        assert df["amount"].iloc[0] == 150000.0
        assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    def test_load_expenses_missing_file_raises(self, tmp_path):
        non_existent = tmp_path / "does_not_exist.sql"
        with pytest.raises(FileNotFoundError):
            load_expenses_from_sql(non_existent)

    def test_load_expenses_default_file_if_exists(self):
        from src.config import DEFAULT_SQL_FILE
        if DEFAULT_SQL_FILE.exists():
            df = load_expenses_from_sql(DEFAULT_SQL_FILE)
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= 4000
            assert "amount" in df.columns
            assert "user_id" in df.columns
