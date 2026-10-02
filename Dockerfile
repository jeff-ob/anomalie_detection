# ==============================================================================
# AuditPulse — Production Dockerfile
# Multi-purpose container supporting CLI, FastAPI REST API, and Streamlit UI.
# ==============================================================================

FROM python:3.11-slim AS base

# Empêche la génération de fichiers .pyc et assure le flushing immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système légères requises (curl pour healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optimisation du cache Docker : copier et installer requirements en premier
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Création d'un utilisateur non-root pour la sécurité d'exécution
RUN useradd -u 1000 -m -s /bin/bash auditpulse && \
    mkdir -p /app/logs /app/data && \
    chown -R auditpulse:auditpulse /app

# Copie de l'ensemble du projet
COPY --chown=auditpulse:auditpulse src/ ./src/
COPY --chown=auditpulse:auditpulse api/ ./api/
COPY --chown=auditpulse:auditpulse dashboard/ ./dashboard/
COPY --chown=auditpulse:auditpulse data/ ./data/
COPY --chown=auditpulse:auditpulse tests/ ./tests/
COPY --chown=auditpulse:auditpulse main.py pytest.ini ./

# Bascule vers l'utilisateur non privilégié
USER auditpulse

# Exposition des ports de service :
# 8000 -> FastAPI REST API
# 8501 -> Streamlit Dashboard
EXPOSE 8000 8501

# Commande par défaut : Serveur API REST AuditPulse
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"]
