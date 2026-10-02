"""
schemas.py
-----------
Modèles Pydantic pour la validation des entrées/sorties de l'API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    id:           str
    company_id:   str
    user_id:      Optional[str] = None
    shop_id:      Optional[str] = None
    supplier_id:  Optional[str] = None
    reference:    Optional[str] = None
    status:       Optional[str] = "pending"
    currency:     Optional[str] = "XAF"
    amount:       float         = Field(..., gt=0)
    source:       Optional[str] = "DASHBOARD"
    due_date:     Optional[str] = None
    paid_at:      Optional[str] = None
    created_at:   Optional[str] = None
    updated_at:   Optional[str] = None
    deleted_at:   Optional[str] = None
    expense_type: Optional[str] = None
    credit_code:  Optional[str] = None
    debit_code:   Optional[str] = None
    state:        Optional[int] = 1

    model_config = {"extra": "allow"}


class BatchInput(BaseModel):
    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=1000)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

class ScoresDetail(BaseModel):
    isolation_forest: float
    lof:              float
    dbscan:           float
    rules:            float
    fraud:            float


class FraudDetail(BaseModel):
    splitting:      float
    duplicate:      float
    ghost_supplier: float
    inflation:      float


class TransactionResult(BaseModel):
    id:               str
    amount:           float
    currency:         str
    expense_type:     str
    source:           str
    created_at:       str
    score_final:      float
    is_anomaly:       bool
    is_fraud_pattern: bool
    scores:           ScoresDetail
    fraud_details:    FraudDetail
    rules_triggered:  list[str]


class BatchSummary(BaseModel):
    total:                int
    n_anomalies:          int
    anomaly_rate:         float
    anomaly_amount_total: float


class BatchResult(BaseModel):
    summary: BatchSummary
    results: list[TransactionResult]


class HealthResponse(BaseModel):
    status:     str
    ready:      bool
    n_train:    int
    n_features: int
    models:     list[str]
    threshold:  float