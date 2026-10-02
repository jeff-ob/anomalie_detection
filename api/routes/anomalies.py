"""
anomalies.py
-------------
Endpoints de détection d'anomalies.

POST /analyze/transaction  → scorer une transaction unique
POST /analyze/batch        → scorer un batch de transactions
GET  /anomalies/top        → top N anomalies du dernier batch en mémoire
"""

from fastapi import APIRouter, HTTPException, Query
from api.schemas import (
    TransactionInput, BatchInput,
    TransactionResult, BatchResult, BatchSummary,
)
from src.model_registry import registry
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Stockage en mémoire du dernier batch scoré (pour GET /anomalies/top)
_last_batch_results: list[dict] = []


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze/transaction
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze/transaction",
    response_model=TransactionResult,
    tags=["Détection"],
    summary="Analyser une transaction unique",
)
def analyze_transaction(transaction: TransactionInput):
    """
    Analyse une transaction et retourne son score d'anomalie.

    **Champs obligatoires** : `id`, `company_id`, `amount`

    **Retourne** :
    - `score_final` : score combiné [0,1], 1 = très suspect
    - `is_anomaly`  : True si score >= 0.50
    - `rules_triggered` : liste des règles déclenchées (explications)
    - `scores` : détail par modèle (IF, LOF, DBSCAN, règles, fraude)
    - `fraud_details` : détail des patterns de fraude détectés
    """
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Modèles en cours de chargement, réessaie dans quelques secondes.")

    try:
        data   = transaction.model_dump()
        result = registry.score_transaction(data)
        logger.info(
            f"✅ Transaction {result['id'][:12]}... | "
            f"score={result['score_final']} | anomaly={result['is_anomaly']}"
        )
        return TransactionResult(**result)

    except Exception as e:
        logger.error(f"❌ Erreur scoring transaction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze/batch
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze/batch",
    response_model=BatchResult,
    tags=["Détection"],
    summary="Analyser un batch de transactions",
)
def analyze_batch(batch: BatchInput):
    """
    Analyse un batch de 1 à 1000 transactions.

    **Retourne** :
    - `summary` : résumé global (total, anomalies, taux, montant total suspect)
    - `results` : liste des résultats individuels (même structure que /analyze/transaction)
    """
    global _last_batch_results

    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Modèles en cours de chargement.")

    try:
        transactions = [t.model_dump() for t in batch.transactions]
        batch_result = registry.score_batch(transactions)

        _last_batch_results = batch_result["results"]

        logger.info(
            f"✅ Batch {batch_result['total']} transactions | "
            f"{batch_result['n_anomalies']} anomalies ({batch_result['anomaly_rate']}%)"
        )

        return BatchResult(
            summary=BatchSummary(
                total                = batch_result["total"],
                n_anomalies          = batch_result["n_anomalies"],
                anomaly_rate         = batch_result["anomaly_rate"],
                anomaly_amount_total = batch_result["anomaly_amount_total"],
            ),
            results=[TransactionResult(**r) for r in batch_result["results"]],
        )

    except Exception as e:
        logger.error(f"❌ Erreur scoring batch : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# GET /anomalies/top
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/anomalies/top",
    response_model=list[TransactionResult],
    tags=["Détection"],
    summary="Top N anomalies du dernier batch",
)
def get_top_anomalies(
    n: int = Query(default=20, ge=1, le=100, description="Nombre d'anomalies à retourner"),
):
    """
    Retourne les N transactions les plus suspectes du dernier batch analysé,
    triées par score décroissant.

    Nécessite d'avoir appelé `POST /analyze/batch` au préalable.
    """
    if not _last_batch_results:
        raise HTTPException(
            status_code=404,
            detail="Aucun batch analysé. Appelle d'abord POST /analyze/batch."
        )

    anomalies = [r for r in _last_batch_results if r["is_anomaly"]]
    top_n     = sorted(anomalies, key=lambda x: x["score_final"], reverse=True)[:n]

    return [TransactionResult(**r) for r in top_n]