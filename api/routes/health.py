"""
health.py
----------
Endpoint de statut de l'API et des modèles.
"""

from fastapi import APIRouter
from api.schemas import HealthResponse
from src.model_registry import registry

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Vérifie que l'API est opérationnelle et que les modèles sont chargés.

    Retourne :
        - status     : "ok" si tout est prêt, "initializing" sinon
        - ready      : True si les modèles sont entraînés
        - n_train    : Nombre de transactions historiques utilisées
        - n_features : Nombre de features ML
        - models     : Liste des modèles actifs
        - threshold  : Seuil de décision actuel
    """
    info = registry.info
    return HealthResponse(
        status     = "ok" if info["ready"] else "initializing",
        ready      = info["ready"],
        n_train    = info["n_train"],
        n_features = info["n_features"],
        models     = info["models"],
        threshold  = info["threshold"],
    )