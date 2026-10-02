"""
config.py
---------
Paramètres globaux du projet : chemins, seuils métier, colonnes attendues.
Tout ce qui est "magic number" ou chemin doit vivre ici pour faciliter
la maintenance et les tests.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# CHEMINS
# ─────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR  = ROOT_DIR / "logs"

# Nom par défaut du fichier SQL source (anonymisé pour AuditPulse)
DEFAULT_SQL_FILE = DATA_DIR / "auditpulse_expenses.sql"

# ─────────────────────────────────────────────
# COLONNES ATTENDUES APRÈS PARSING
# ─────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "id", "company_id", "user_id", "shop_id", "supplier_id",
    "reference", "status", "currency", "amount", "source",
    "due_date", "paid_at", "items", "shipping", "billing",
    "metadata", "created_at", "updated_at", "deleted_at",
    "expense_type", "credit_code", "debit_code", "state",
]

# Colonnes de dates à parser automatiquement
DATE_COLUMNS = ["due_date", "paid_at", "created_at", "updated_at", "deleted_at"]

# ─────────────────────────────────────────────
# SEUILS DE DÉTECTION (règles métier)
# ─────────────────────────────────────────────

# Nombre d'écarts-types au-delà duquel un montant est "anormalement élevé"
ZSCORE_THRESHOLD = 3.0

# IQR : facteur multiplicateur pour définir les outliers (méthode de Tukey)
IQR_FACTOR = 1.5

# Heure de début / fin de la "plage normale" de transactions (heure locale)
BUSINESS_HOUR_START = 7   # 07h00
BUSINESS_HOUR_END   = 20  # 20h00

# Fréquence max de dépenses par utilisateur par jour avant alerte
MAX_TRANSACTIONS_PER_DAY = 10

# Nombre minimal d'utilisations d'un fournisseur pour qu'il soit considéré "connu"
RARE_SUPPLIER_THRESHOLD = 3

# Pourcentage de variation du montant moyen (rolling 7j) déclenchant une alerte
AMOUNT_VARIATION_THRESHOLD = 2.0  # x2 par rapport à la moyenne glissante

# ─────────────────────────────────────────────
# MODÈLES
# ─────────────────────────────────────────────

# Isolation Forest
ISOLATION_FOREST_CONTAMINATION = 0.05   # proportion estimée d'anomalies
ISOLATION_FOREST_N_ESTIMATORS  = 200
ISOLATION_FOREST_RANDOM_STATE  = 42

# Local Outlier Factor
LOF_N_NEIGHBORS    = 20
LOF_CONTAMINATION  = 0.05

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

API_HOST    = "0.0.0.0"
API_PORT    = 8000
API_VERSION = "v1"
API_PREFIX  = f"/api/{API_VERSION}"

# Aliases pour les modèles (compatibilité imports)
IF_CONTAMINATION  = ISOLATION_FOREST_CONTAMINATION
IF_N_ESTIMATORS   = ISOLATION_FOREST_N_ESTIMATORS
IF_MAX_SAMPLES    = "auto"
IF_RANDOM_STATE   = ISOLATION_FOREST_RANDOM_STATE

# ─────────────────────────────────────────────
# RÈGLES MÉTIER - PARAMÈTRES DÉTAILLÉS
# ─────────────────────────────────────────────

RULE_AMOUNT_HIGH_Z         = ZSCORE_THRESHOLD   # z-score seuil montant élevé
RULE_WEEKEND_WEIGHT        = 0.10               # poids règle week-end
RULE_OUTSIDE_HOURS_WEIGHT  = 0.10               # poids règle hors horaires
RULE_RARE_SUPPLIER_WEIGHT  = 0.10               # poids règle fournisseur rare
RULE_ROUND_AMOUNT_THRESHOLD = 100_000           # montant minimum pour "rond suspect"
RULE_HIGH_FREQ_THRESHOLD    = MAX_TRANSACTIONS_PER_DAY
RULES_ANOMALY_THRESHOLD     = 0.30              # score composite minimum pour être anomalie

# ─────────────────────────────────────────────
# SCORER COMBINÉ
# ─────────────────────────────────────────────

# Poids des modèles dans le score final combiné
SCORER_WEIGHT_IF    = 0.35
SCORER_WEIGHT_LOF   = 0.35
SCORER_WEIGHT_RULES = 0.30

# Seuil de score combiné pour déclarer une anomalie finale
FINAL_ANOMALY_THRESHOLD = 0.50