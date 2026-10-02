"""
main.py
--------
Point d'entrée de l'API FastAPI.

Démarrage :
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

L'API charge et entraîne les modèles au démarrage via le lifespan,
puis les garde en mémoire pour toute la durée de vie du serveur.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, anomalies
from src.model_registry import registry
from src.config import DEFAULT_SQL_FILE
from src.utils.logger import get_logger

logger = get_logger("api")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — chargement des modèles au démarrage
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'API :
        - Au démarrage → entraîner les modèles sur les données historiques
        - À l'arrêt    → nettoyage (rien à faire ici)
    """
    logger.info("🔄 Démarrage de l'API — chargement des modèles...")
    registry.initialize(DEFAULT_SQL_FILE)
    logger.info("✅ API prête à recevoir des requêtes.")
    yield
    logger.info("🛑 Arrêt de l'API.")


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AuditPulse — Détection d'Anomalies Financières",
    description = (
        "API d'audit et de détection d'anomalies sur les dépenses d'entreprise.\n\n"
        "Combine 5 moteurs d'analyse : Isolation Forest, LOF, DBSCAN, "
        "Règles métier expertes et Détection de patterns de fraude."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — à restreindre en production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(health.router,    prefix="/api/v1")
app.include_router(anomalies.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "AuditPulse Financial Anomaly Detection API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }