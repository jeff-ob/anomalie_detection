"""
preprocessing.py
----------------
Responsabilité : transformer le DataFrame brut en DataFrame enrichi,
prêt pour les modèles de détection d'anomalies.

Étapes :
    1. Nettoyage de base (doublons, lignes supprimées, valeurs aberrantes évidentes)
    2. Feature engineering temporel (heure, jour de semaine, délai paiement...)
    3. Features comportementales par utilisateur (rolling stats, fréquence...)
    4. Features fournisseur (rareté, concentration...)
    5. Features montant (z-score global, z-score par user, IQR flags)
    6. Encodage léger des catégorielles pour les modèles ML

Chaque groupe de features est dans une fonction dédiée pour la lisibilité
et pour permettre des tests unitaires indépendants.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.config import (
    BUSINESS_HOUR_START,
    BUSINESS_HOUR_END,
    MAX_TRANSACTIONS_PER_DAY,
    RARE_SUPPLIER_THRESHOLD,
    ZSCORE_THRESHOLD,
    IQR_FACTOR,
    AMOUNT_VARIATION_THRESHOLD,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _ensure_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force la conversion des colonnes datetime (utile quand les données viennent de l'API)."""
    datetime_cols = ["created_at", "updated_at", "deleted_at", "due_date", "paid_at"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 1. NETTOYAGE DE BASE
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage initial du DataFrame :
    - Supprime les lignes soft-deleted (deleted_at non null)
    - Supprime les doublons sur `id`
    - Retire les lignes sans montant (amount null ou <= 0)
    - Retire les lignes sans created_at (impossible de faire de l'analyse temporelle)

    Args:
        df : DataFrame brut issu de data_loader.

    Returns:
        DataFrame nettoyé.
    """
    initial_size = len(df)
    logger.info(f"🧹 Nettoyage démarré : {initial_size} lignes en entrée.")

    # Suppression des enregistrements logiquement supprimés
    if "deleted_at" in df.columns:
        deleted = df["deleted_at"].notna().sum()
        df = df[df["deleted_at"].isna()].copy()
        logger.info(f"  → {deleted} lignes soft-deleted supprimées.")

    # Déduplication sur l'identifiant métier
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["id"])
    dupl = before_dedup - len(df)
    if dupl:
        logger.warning(f"  → {dupl} doublons sur `id` supprimés.")

    # Lignes sans montant exploitable
    before = len(df)
    df = df[df["amount"].notna() & (df["amount"] > 0)]
    logger.info(f"  → {before - len(df)} lignes sans montant valide supprimées.")

    # Lignes sans date de création
    before = len(df)
    df = df[df["created_at"].notna()]
    logger.info(f"  → {before - len(df)} lignes sans `created_at` supprimées.")

    logger.info(
        f"✅ Nettoyage terminé : {len(df)} lignes conservées "
        f"({initial_size - len(df)} supprimées au total)."
    )
    return df.reset_index(drop=True)



# ─────────────────────────────────────────────────────────────────────────────
# 1b. NORMALISATION HISTORIQUE
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_NORMALIZATION = {
    "WATHSAPP": "WHATSAPP", "WHATSAP": "WHATSAPP", "WHATSAAP": "WHATSAPP",
    "WHASAPP": "WHATSAPP", "PAR WHATSAPP": "WHATSAPP",
    "TWITTER/X": "TWITTER",
    "APPEL": "PHONE",
    "VISITE EN MAGASIN": "BOUTIQUE", "PHYSIQUE": "BOUTIQUE", "SHOP": "BOUTIQUE",
    "VENTE DIRECT": "VENTE_DIRECTE", "COMMANDE DIRECT": "VENTE_DIRECTE",
    "COMMANDE DE MADAME": "VENTE_DIRECTE",
    "FACE": "FACEBOOK",
    "RÉSEAUX": "RESEAUX_SOCIAUX",
}


def normalize_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les incohérences historiques de saisie.

    - expense_type : uniformise la casse (tout en minuscule + strip)
    - source       : corrige les fautes de frappe via un mapping explicite
    - currency     : majuscule + strip
    """
    df = df.copy()

    if "expense_type" in df.columns:
        before = df["expense_type"].nunique()
        df["expense_type"] = df["expense_type"].fillna("unknown").str.strip().str.lower()
        after = df["expense_type"].nunique()
        logger.info(f"  → expense_type : {before} → {after} catégories après normalisation")

    if "source" in df.columns:
        before = df["source"].nunique()
        df["source"] = df["source"].fillna("UNKNOWN").str.strip().str.upper()
        df["source"] = df["source"].replace(SOURCE_NORMALIZATION)
        after = df["source"].nunique()
        logger.info(f"  → source : {before} → {after} catégories après normalisation")

    if "currency" in df.columns:
        df["currency"] = df["currency"].fillna("UNKNOWN").str.strip().str.upper()

    logger.info("✅ Normalisation historique terminée.")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES TEMPORELLES
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des features temporelles depuis `created_at` et `paid_at`.

    Features créées :
    - hour_of_day        : heure de création (0-23)
    - day_of_week        : jour de la semaine (0=lundi, 6=dimanche)
    - is_weekend         : bool, samedi ou dimanche
    - is_outside_hours   : bool, hors de la plage BUSINESS_HOUR_START–END
    - month              : mois (1-12)
    - payment_delay_days : délai entre created_at et paid_at (None si pas payé)
    - is_future_due      : bool, due_date dans le futur par rapport à created_at

    Args:
        df : DataFrame nettoyé.

    Returns:
        DataFrame enrichi de features temporelles.
    """
    logger.info("⏱️  Ajout des features temporelles...")

    df = df.copy()

    # Extraction depuis created_at
    df["hour_of_day"]      = df["created_at"].dt.hour
    df["day_of_week"]      = df["created_at"].dt.dayofweek  # 0=lundi
    df["is_weekend"]       = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"]            = df["created_at"].dt.month
    df["year"]             = df["created_at"].dt.year

    # Transaction hors plage horaire normale
    df["is_outside_hours"] = (
        (df["hour_of_day"] < BUSINESS_HOUR_START) |
        (df["hour_of_day"] >= BUSINESS_HOUR_END)
    ).astype(int)

    # Délai de paiement en jours
    if "paid_at" in df.columns:
        paid_dt = pd.to_datetime(df["paid_at"], errors="coerce")
        created_dt = pd.to_datetime(df["created_at"], errors="coerce")
        df["payment_delay_days"] = (
            (paid_dt - created_dt)
            .dt.total_seconds()
            .div(86400)        # conversion secondes → jours
            .round(2)
        )
    else:
        df["payment_delay_days"] = np.nan

    # Indicateur due_date dans le futur (possible dépense anticipée anormale)
    if "due_date" in df.columns:
        due_dt = pd.to_datetime(df["due_date"], errors="coerce")
        created_dt = pd.to_datetime(df["created_at"], errors="coerce")
        df["is_future_due"] = (due_dt > created_dt).fillna(False).astype(int)
    else:
        df["is_future_due"] = 0

    logger.info(
        f"  → Features créées : hour_of_day, day_of_week, is_weekend, "
        f"is_outside_hours, month, year, payment_delay_days, is_future_due"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURES MONTANT
# ─────────────────────────────────────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features basées sur le montant des dépenses.

    Features créées :
    - amount_log                : log(amount + 1) pour réduire l'asymétrie
    - amount_zscore_global      : z-score du montant sur l'ensemble des données
    - amount_zscore_per_user    : z-score du montant par rapport à l'utilisateur
    - amount_iqr_flag_global    : 1 si outlier IQR global
    - amount_iqr_flag_per_user  : 1 si outlier IQR par utilisateur
    - amount_vs_user_mean_ratio : ratio montant / moyenne de l'utilisateur

    Args:
        df : DataFrame avec feature temporelles.

    Returns:
        DataFrame enrichi de features montant.
    """
    logger.info("💰 Ajout des features montant...")

    df = df.copy()

    # Log-transform pour atténuer les valeurs extrêmes
    df["amount_log"] = np.log1p(df["amount"])

    # --- Z-score global ---
    global_mean = df["amount"].mean()
    global_std  = df["amount"].std()
    df["amount_zscore_global"] = (
        (df["amount"] - global_mean) / global_std
        if global_std > 0 else 0.0
    )
    df["amount_iqr_flag_global"] = _iqr_outlier_flag(df["amount"])

    logger.info(
        f"  → Global : mean={global_mean:.2f}, std={global_std:.2f}"
    )

    # --- Z-score et IQR par utilisateur ---
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "user_amount_mean", "std": "user_amount_std"}
    )
    df = df.join(user_stats, on="user_id")

    # Évite la division par zéro pour les utilisateurs avec 1 seule dépense
    df["amount_zscore_per_user"] = np.where(
        df["user_amount_std"] > 0,
        (df["amount"] - df["user_amount_mean"]) / df["user_amount_std"],
        0.0,
    )

    # Ratio : combien de fois le montant dépasse la moyenne de l'utilisateur
    df["amount_vs_user_mean_ratio"] = np.where(
        df["user_amount_mean"] > 0,
        df["amount"] / df["user_amount_mean"],
        1.0,
    )

    # IQR flag par utilisateur
    df["amount_iqr_flag_per_user"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: _iqr_outlier_flag(x))
    )

    logger.info(
        "  → Features créées : amount_log, amount_zscore_global, "
        "amount_zscore_per_user, amount_iqr_flag_global, "
        "amount_iqr_flag_per_user, amount_vs_user_mean_ratio"
    )
    return df


def _iqr_outlier_flag(series: pd.Series) -> pd.Series:
    """
    Retourne une Series binaire (0/1) indiquant les outliers selon la méthode IQR.

    Un outlier est défini comme :
        valeur < Q1 - IQR_FACTOR * IQR  OU  valeur > Q3 + IQR_FACTOR * IQR

    Args:
        series : Série numérique.

    Returns:
        pd.Series de 0/1.
    """
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_FACTOR * iqr
    upper = q3 + IQR_FACTOR * iqr
    return ((series < lower) | (series > upper)).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURES COMPORTEMENTALES PAR UTILISATEUR
# ─────────────────────────────────────────────────────────────────────────────

def add_user_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des features basées sur le comportement historique de chaque utilisateur.

    Features créées :
    - user_tx_count              : nombre total de transactions de l'utilisateur
    - user_tx_per_day            : moyenne de transactions par jour
    - tx_daily_count             : nb de transactions de cet user ce jour précis
    - tx_daily_count_flag        : 1 si tx_daily_count > MAX_TRANSACTIONS_PER_DAY
    - user_rolling_mean_7d       : moyenne glissante montant sur 7 jours par user
    - user_rolling_std_7d        : écart-type glissant montant sur 7 jours par user
    - amount_vs_rolling_mean_ratio: ratio montant / moyenne glissante 7j

    Note : Les rolling features nécessitent un tri chronologique.

    Args:
        df : DataFrame avec features temporelles et montant.

    Returns:
        DataFrame enrichi.
    """
    logger.info("👤 Ajout des features comportementales utilisateur...")

    df = df.copy().sort_values("created_at")

    # Nombre total de transactions par utilisateur
    user_tx_count = df.groupby("user_id")["id"].transform("count")
    df["user_tx_count"] = user_tx_count

    # Moyenne de transactions par jour (durée active = max_date - min_date)
    user_date_range = df.groupby("user_id")["created_at"].agg(
        lambda x: max((x.max() - x.min()).days, 1)  # au moins 1 jour
    ).rename("user_active_days")
    df = df.join(user_date_range, on="user_id")
    df["user_tx_per_day"] = df["user_tx_count"] / df["user_active_days"]

    # Nombre de transactions de l'utilisateur ce jour-là
    df["date_only"] = df["created_at"].dt.date
    df["tx_daily_count"] = (
        df.groupby(["user_id", "date_only"])["id"]
        .transform("count")
    )
    df["tx_daily_count_flag"] = (
        df["tx_daily_count"] > MAX_TRANSACTIONS_PER_DAY
    ).astype(int)

    # Statistiques glissantes sur 7 jours (nécessite un index temporel)
    df = _add_rolling_features(df)

    # Nettoyage de la colonne temporaire
    df.drop(columns=["date_only"], inplace=True, errors="ignore")

    logger.info(
        "  → Features créées : user_tx_count, user_tx_per_day, "
        "tx_daily_count, tx_daily_count_flag, "
        "user_rolling_mean_7d, user_rolling_std_7d, amount_vs_rolling_mean_ratio"
    )
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des statistiques glissantes sur 7 jours par utilisateur.

    Utilise un groupby + rolling sur l'index temporel pour chaque user.

    Args:
        df : DataFrame trié par created_at.

    Returns:
        DataFrame avec les colonnes rolling ajoutées.
    """
    df = df.set_index("created_at")

    rolling_mean = []
    rolling_std  = []

    for user_id, group in df.groupby("user_id"):
        # Fenêtre de 7 jours glissants, minimum 1 observation
        rm = group["amount"].rolling("7D", min_periods=1).mean()
        rs = group["amount"].rolling("7D", min_periods=1).std().fillna(0)
        rolling_mean.append(rm)
        rolling_std.append(rs)

    if rolling_mean:
        combined_mean = pd.concat(rolling_mean)
        combined_std  = pd.concat(rolling_std)
        # Gérer les index dupliqués
        df["user_rolling_mean_7d"] = combined_mean.groupby(level=0).first().reindex(df.index)
        df["user_rolling_std_7d"]  = combined_std.groupby(level=0).first().reindex(df.index)
    else:
        df["user_rolling_mean_7d"] = df["amount"]
        df["user_rolling_std_7d"]  = 0.0

    df = df.reset_index()

    # Ratio : détecte un changement brutal par rapport à la tendance récente
    df["amount_vs_rolling_mean_ratio"] = np.where(
        df["user_rolling_mean_7d"] > 0,
        df["amount"] / df["user_rolling_mean_7d"],
        1.0,
    )

    # Flag : changement brutal (ratio > seuil configuré)
    df["rolling_amount_spike_flag"] = (
        df["amount_vs_rolling_mean_ratio"] > AMOUNT_VARIATION_THRESHOLD
    ).astype(int)

    return df




def add_user_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de profil comportemental propres à chaque utilisateur.
    Détecte les écarts par rapport aux HABITUDES PERSONNELLES de chaque user,
    pas juste par rapport à la population globale.

    Features créées :
    - user_hour_mean / user_hour_std   : heure habituelle de transaction de cet user
    - hour_deviation_from_user_norm    : écart entre l heure actuelle et l habitude user
    - user_typical_amount_median       : montant médian habituel de cet user
    - amount_deviation_from_user_median: ratio montant / médiane perso
    - user_typical_source              : source la plus fréquente de cet user
    - is_unusual_source_for_user       : 1 si la source est inhabituelle pour cet user
    - user_typical_expense_type        : type de dépense le plus fréquent de cet user
    - is_unusual_type_for_user         : 1 si le type est inhabituel pour cet user
    - days_since_last_tx               : jours depuis la dernière transaction du user
    """
    logger.info("👤 Ajout des features de profil utilisateur...")
    df = df.copy().sort_values("created_at")

    # --- Heure habituelle par user ---
    user_hour_stats = df.groupby("user_id")["hour_of_day"].agg(["mean", "std"]).rename(
        columns={"mean": "user_hour_mean", "std": "user_hour_std"}
    )
    user_hour_stats["user_hour_std"] = user_hour_stats["user_hour_std"].fillna(0)
    df = df.join(user_hour_stats, on="user_id")

    # Écart en heures entre l heure actuelle et l habitude (normalisé par le std)
    df["hour_deviation_from_user_norm"] = np.where(
        df["user_hour_std"] > 0,
        np.abs(df["hour_of_day"] - df["user_hour_mean"]) / df["user_hour_std"],
        0.0,
    )

    # --- Montant médian par user ---
    user_median = df.groupby("user_id")["amount"].median().rename("user_typical_amount_median")
    df = df.join(user_median, on="user_id")
    df["amount_deviation_from_user_median"] = np.where(
        df["user_typical_amount_median"] > 0,
        df["amount"] / df["user_typical_amount_median"],
        1.0,
    )

    # --- Source habituelle par user ---
    user_typical_source = (
        df.groupby("user_id")["source"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN")
        .rename("user_typical_source")
    )
    df = df.join(user_typical_source, on="user_id")
    df["is_unusual_source_for_user"] = (
        df["source"] != df["user_typical_source"]
    ).astype(int)

    # --- Type de dépense habituel par user ---
    user_typical_type = (
        df.groupby("user_id")["expense_type"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown")
        .rename("user_typical_expense_type")
    )
    df = df.join(user_typical_type, on="user_id")
    df["is_unusual_type_for_user"] = (
        df["expense_type"] != df["user_typical_expense_type"]
    ).astype(int)

    # --- Jours depuis la dernière transaction ---
    df = df.sort_values("created_at")
    df["days_since_last_tx"] = (
        df.groupby("user_id")["created_at"]
        .diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
    )

    logger.info(
        "  → Features créées : hour_deviation_from_user_norm, "
        "amount_deviation_from_user_median, is_unusual_source_for_user, "
        "is_unusual_type_for_user, days_since_last_tx"
    )
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURES FOURNISSEUR
# ─────────────────────────────────────────────────────────────────────────────

def add_supplier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features liées aux fournisseurs.

    Features créées :
    - supplier_usage_count  : nombre de fois que ce fournisseur a été utilisé
    - is_rare_supplier      : 1 si supplier_usage_count < RARE_SUPPLIER_THRESHOLD
    - supplier_avg_amount   : montant moyen par fournisseur
    - amount_vs_supplier_avg: ratio montant / moyenne fournisseur

    Note : les transactions sans fournisseur (supplier_id = None) reçoivent
    is_rare_supplier = 0 car l'absence de fournisseur est normale dans ce contexte.

    Args:
        df : DataFrame.

    Returns:
        DataFrame enrichi.
    """
    logger.info("🏪 Ajout des features fournisseur...")

    df = df.copy()

    # Compte les utilisations par fournisseur (ignore les null)
    supplier_counts = (
        df[df["supplier_id"].notna()]
        .groupby("supplier_id")["id"]
        .count()
        .rename("supplier_usage_count")
    )
    df = df.join(supplier_counts, on="supplier_id")

    # Pour les transactions sans fournisseur → count = 0 (neutre)
    df["supplier_usage_count"] = df["supplier_usage_count"].fillna(0).astype(int)

    # Fournisseur rarement utilisé
    df["is_rare_supplier"] = (
        (df["supplier_id"].notna()) &
        (df["supplier_usage_count"] < RARE_SUPPLIER_THRESHOLD)
    ).astype(int)

    # Montant moyen par fournisseur
    supplier_avg = (
        df[df["supplier_id"].notna()]
        .groupby("supplier_id")["amount"]
        .mean()
        .rename("supplier_avg_amount")
    )
    df = df.join(supplier_avg, on="supplier_id")
    df["supplier_avg_amount"] = df["supplier_avg_amount"].fillna(df["amount"])

    df["amount_vs_supplier_avg"] = np.where(
        df["supplier_avg_amount"] > 0,
        df["amount"] / df["supplier_avg_amount"],
        1.0,
    )

    logger.info(
        "  → Features créées : supplier_usage_count, is_rare_supplier, "
        "supplier_avg_amount, amount_vs_supplier_avg"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENCODAGE DES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles nécessaires aux modèles ML.

    On utilise un simple label-encoding (pas de one-hot pour garder
    la matrice de features compacte) car les modèles non supervisés
    (IF, LOF) n'en ont pas besoin différemment.

    Colonnes encodées :
    - status       → status_encoded
    - expense_type → expense_type_encoded
    - source       → source_encoded
    - currency     → currency_encoded

    Args:
        df : DataFrame.

    Returns:
        DataFrame avec colonnes encodées ajoutées.
    """
    logger.info("🔠 Encodage des colonnes catégorielles...")

    df = df.copy()
    cat_cols = ["status", "expense_type", "source", "currency"]

    for col in cat_cols:
        if col not in df.columns:
            logger.warning(f"  → Colonne '{col}' absente, encodage ignoré.")
            continue
        encoded_col = f"{col}_encoded"
        df[encoded_col] = df[col].astype("category").cat.codes
        # -1 = valeur nulle / inconnue après cat.codes
        logger.info(
            f"  → '{col}' → '{encoded_col}' "
            f"({df[col].nunique()} catégories)"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de preprocessing complet.
    Enchaîne toutes les étapes dans l'ordre logique.

    Args:
        df : DataFrame brut issu de data_loader.load_expenses_from_sql().

    Returns:
        DataFrame prêt pour la modélisation.
    """
    logger.info("🚀 Démarrage du pipeline de preprocessing...")

    df = _ensure_datetime_columns(df)
    df = clean_dataframe(df)
    df = normalize_historical_data(df)
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_user_behavior_features(df)
    df = add_user_profile_features(df)
    df = add_supplier_features(df)
    df = encode_categoricals(df)

    # Résumé final
    n_rows, n_cols = df.shape
    feature_cols = [c for c in df.columns if c not in [
        "id", "company_id", "user_id", "shop_id", "supplier_id",
        "reference", "items", "shipping", "billing", "metadata",
        "created_at", "updated_at", "deleted_at", "paid_at", "due_date",
    ]]
    logger.info(
        f"✅ Preprocessing terminé : {n_rows} lignes, {n_cols} colonnes totales, "
        f"{len(feature_cols)} features exploitables."
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SÉLECTION DES FEATURES POUR LES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

# Liste des features numériques à passer aux modèles ML
ML_FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "amount_zscore_global",
    "amount_zscore_per_user",
    "amount_vs_user_mean_ratio",
    "amount_vs_rolling_mean_ratio",
    "amount_vs_supplier_avg",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_outside_hours",
    "payment_delay_days",
    "user_tx_count",
    "user_tx_per_day",
    "tx_daily_count",
    "supplier_usage_count",
    "is_rare_supplier",
    "rolling_amount_spike_flag",
    "hour_deviation_from_user_norm",
    "amount_deviation_from_user_median",
    "is_unusual_source_for_user",
    "is_unusual_type_for_user",
    "days_since_last_tx",
    "status_encoded",
    "expense_type_encoded",
    "source_encoded",
    "currency_encoded",
]


def get_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait uniquement les features numériques pour les modèles ML,
    en gérant les colonnes manquantes et les NaN résiduels.

    Args:
        df : DataFrame preprocessé.

    Returns:
        DataFrame ne contenant que les features ML, sans NaN.
    """
    available = [c for c in ML_FEATURE_COLUMNS if c in df.columns]
    missing   = [c for c in ML_FEATURE_COLUMNS if c not in df.columns]

    if missing:
        logger.warning(f"⚠️  Features ML manquantes (ignorées) : {missing}")

    X = df[available].copy()

    # Imputation simple des NaN résiduels par la médiane de chaque colonne
    nan_counts = X.isna().sum()
    if nan_counts.any():
        logger.info(f"  → Imputation médiane sur : {nan_counts[nan_counts > 0].to_dict()}")
        X = X.fillna(X.median(numeric_only=True))

    # Sécurité contre les colonnes 100% NaN (dont la médiane est NaN, ex: payment_delay_days)
    if X.isna().any().any():
        empty_cols = X.columns[X.isna().all()].tolist()
        if empty_cols:
            logger.warning(f"⚠️ Colonnes intégralement NaN imputées à 0.0 : {empty_cols}")
        X = X.fillna(0.0)

    logger.info(f"✅ Matrice features ML : {X.shape[0]} lignes × {X.shape[1]} colonnes (0 NaN résiduel)")
    return X