"""
test_api.py
-----------
Tests d'intégration pour l'API REST FastAPI (api/main.py).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.model_registry import registry
from src.config import DEFAULT_SQL_FILE


@pytest.fixture(scope="module")
def client():
    """Initialise le TestClient avec le lifespan de l'application."""
    # S'assurer que le registre est prêt
    if not registry.is_ready and DEFAULT_SQL_FILE.exists():
        registry.initialize(DEFAULT_SQL_FILE)

    with TestClient(app) as c:
        yield c


class TestAPIBasics:
    """Tests des endpoints de base et monitoring."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "AuditPulse" in data["message"]
        assert data["version"] == "1.0.0"
        assert "/docs" in data["docs"]

    def test_health_check_endpoint(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "initializing"]
        assert "models" in data
        assert isinstance(data["models"], list)


class TestAPIAnomalyDetection:
    """Tests des endpoints de détection unitaire et par batch."""

    def test_analyze_single_transaction_normal(self, client):
        payload = {
            "id": "EXP_TEST_001",
            "company_id": "COMP_0001",
            "user_id": "USR_0001",
            "amount": 25000.0,
            "currency": "XAF",
            "source": "DASHBOARD",
            "expense_type": "office",
            "created_at": "2026-05-02 10:30:00",
        }
        response = client.post("/api/v1/analyze/transaction", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "EXP_TEST_001"
        assert 0.0 <= data["score_final"] <= 1.0
        assert isinstance(data["is_anomaly"], bool)
        assert "scores" in data
        assert "fraud_details" in data
        assert "rules_triggered" in data

    def test_analyze_single_transaction_critical_anomaly(self, client):
        # Transaction aberrante : montant de 1 milliard à 2h du matin un dimanche
        payload = {
            "id": "EXP_TEST_ANOMALY",
            "company_id": "COMP_0001",
            "user_id": "USR_9999",
            "amount": 1_000_000_000.0,
            "currency": "XAF",
            "source": "FACEBOOK",
            "expense_type": "other",
            "created_at": "2026-05-03 02:15:00",  # Dimanche 2h du matin
        }
        response = client.post("/api/v1/analyze/transaction", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_anomaly"] is True
        assert data["score_final"] > 0.40
        assert len(data["rules_triggered"]) > 0

    def test_analyze_transaction_invalid_amount_raises_422(self, client):
        payload = {
            "id": "EXP_TEST_BAD",
            "company_id": "COMP_0001",
            "amount": -500.0,  # Doit être > 0
        }
        response = client.post("/api/v1/analyze/transaction", json=payload)
        assert response.status_code == 422

    def test_analyze_batch_and_top_anomalies(self, client):
        batch_payload = {
            "transactions": [
                {
                    "id": "EXP_B1",
                    "company_id": "COMP_0001",
                    "user_id": "USR_0001",
                    "amount": 15000.0,
                    "created_at": "2026-05-02 11:00:00",
                },
                {
                    "id": "EXP_B2",
                    "company_id": "COMP_0001",
                    "user_id": "USR_0002",
                    "amount": 800_000_000.0,  # Suspect
                    "created_at": "2026-05-03 03:00:00",
                },
            ]
        }
        batch_resp = client.post("/api/v1/analyze/batch", json=batch_payload)
        assert batch_resp.status_code == 200
        batch_data = batch_resp.json()
        assert batch_data["summary"]["total"] == 2
        assert len(batch_data["results"]) == 2

        # Test de GET /anomalies/top
        top_resp = client.get("/api/v1/anomalies/top?n=5")
        assert top_resp.status_code == 200
        top_data = top_resp.json()
        assert isinstance(top_data, list)
        if len(top_data) > 0:
            assert top_data[0]["score_final"] >= top_data[-1]["score_final"]
