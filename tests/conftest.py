"""
conftest.py
-----------
Fixtures pytest partagées pour l'ensemble de la suite de tests d'AuditPulse.
Fournit des jeux de données synthétiques réalistes et des fichiers temporaires.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.preprocessing import preprocess, get_ml_features


@pytest.fixture
def sample_raw_data() -> list[dict]:
    """Génère un jeu de données synthétique représentatif du schéma SQL brut."""
    base_date = datetime(2026, 5, 1, 9, 0, 0)
    users = [f"USR_{i:04d}" for i in range(1, 6)]
    suppliers = [f"SUPP_{i:04d}" for i in range(1, 10)]
    sources = ["DASHBOARD", "FACEBOOK", "WHATSAPP", "POS", "API"]
    expense_types = ["merchandise-purchase", "raw_materials", "marketing", "logistics", "office"]
    
    rows = []
    for i in range(100):
        user = users[i % len(users)]
        supplier = suppliers[i % len(suppliers)] if i % 10 != 0 else None
        dt = base_date + timedelta(hours=i * 3, minutes=(i % 5) * 12)
        
        # Introduire quelques cas particuliers
        if i == 50:
            # Outlier de montant
            amount = 15_000_000.0
        elif i == 51:
            # Montant rond
            amount = 500_000.0
        elif i == 95:
            # Soft deleted
            deleted_at = dt + timedelta(days=1)
            amount = 25_000.0
        elif i == 96:
            # Montant invalide
            amount = -100.0
            deleted_at = None
        else:
            amount = float(10_000 + (i * 1234) % 150_000)
            deleted_at = None

        rows.append({
            "id": f"EXP_{i:05d}",
            "company_id": "COMP_0001",
            "user_id": user,
            "shop_id": "SHOP_0001",
            "supplier_id": supplier,
            "reference": f"REF-{i:05d}",
            "status": "APPROVED",
            "currency": "XAF",
            "amount": amount,
            "source": sources[i % len(sources)],
            "due_date": dt + timedelta(days=15),
            "paid_at": None,
            "items": '{"desc": "item sample", "qty": 1}',
            "shipping": None,
            "billing": '{"acc": "ACC_0001"}',
            "metadata": '{"device": "mobile"}',
            "created_at": dt,
            "updated_at": dt,
            "deleted_at": deleted_at,
            "expense_type": expense_types[i % len(expense_types)],
            "credit_code": "401000",
            "debit_code": "601000",
            "state": 1,
        })
    return rows


@pytest.fixture
def sample_raw_df(sample_raw_data) -> pd.DataFrame:
    """DataFrame brut directement instancié."""
    return pd.DataFrame(sample_raw_data)


@pytest.fixture
def sample_sql_file(tmp_path) -> Path:
    """Crée un fichier SQL temporaire valide avec la syntaxe MySQL exacte."""
    sql_path = tmp_path / "test_expenses.sql"
    sql_content = """-- Dump de test AuditPulse
INSERT INTO `expenses` (`id`, `company_id`, `user_id`, `shop_id`, `supplier_id`, `reference`, `status`, `currency`, `amount`, `source`, `due_date`, `paid_at`, `items`, `shipping`, `billing`, `metadata`, `created_at`, `updated_at`, `deleted_at`, `expense_type`, `credit_code`, `debit_code`, `state`) VALUES
('EXP_00001', 'COMP_0001', 'USR_0001', 'SHOP_0001', 'SUPP_0001', 'REF-001', 'APPROVED', 'XAF', 150000.00, 'DASHBOARD', '2026-05-15 00:00:00', NULL, '{"item": "Server", "qty": 1}', NULL, '{"acc": "ACC_001"}', NULL, '2026-05-01 09:30:00', '2026-05-01 09:30:00', NULL, 'office', '401', '601', 1),
('EXP_00002', 'COMP_0001', 'USR_0002', 'SHOP_0001', 'SUPP_0002', 'REF-002', 'APPROVED', 'XAF', 25000.50, 'FACEBOOK', '2026-05-20 00:00:00', NULL, '{"item": "Ad", "qty": 1}', NULL, NULL, NULL, '2026-05-02 14:15:00', '2026-05-02 14:15:00', NULL, 'marketing', '401', '623', 1),
('EXP_00003', 'COMP_0001', 'USR_0001', 'SHOP_0001', NULL, 'REF-003', 'PENDING', 'XAF', 500000.00, 'WHATSAPP', NULL, NULL, NULL, NULL, NULL, NULL, '2026-05-03 23:45:00', '2026-05-03 23:45:00', NULL, 'raw_materials', '401', '601', 1);
"""
    sql_path.write_text(sql_content, encoding="utf-8")
    return sql_path


@pytest.fixture
def sample_processed_df(sample_raw_df) -> pd.DataFrame:
    """DataFrame enrichi par la chaîne de preprocessing."""
    return preprocess(sample_raw_df)


@pytest.fixture
def sample_ml_features(sample_processed_df) -> pd.DataFrame:
    """Matrice numérique pour l'entraînement et l'inférence des modèles ML."""
    return get_ml_features(sample_processed_df)
