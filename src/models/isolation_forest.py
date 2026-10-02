"""
isolation_forest.py
--------------------
Isolation Forest avec séparation fit() / predict().

- fit()     : entraîne le modèle sur les données historiques
- predict() : applique le modèle entraîné sur de nouvelles données
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import IF_CONTAMINATION, IF_N_ESTIMATORS, IF_MAX_SAMPLES, IF_RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IsolationForestDetector:
    """Wrapper Isolation Forest avec état fit/predict séparé."""

    def __init__(
        self,
        contamination: float = IF_CONTAMINATION,
        n_estimators:  int   = IF_N_ESTIMATORS,
        max_samples        = IF_MAX_SAMPLES,
        random_state:  int   = IF_RANDOM_STATE,
    ):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.random_state  = random_state

        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(self, df_features: pd.DataFrame) -> "IsolationForestDetector":
        """Entraîne le modèle sur les données historiques."""
        X = self._prepare(df_features)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        raw = self.model.score_samples(X_scaled)
        self._train_min = (-raw).min()
        self._train_max = (-raw).max()

        n = len(df_features)
        logger.info(f"🌲 IF entraîné sur {n:,} échantillons (contamination={self.contamination})")
        return self

    # ── Prédiction ───────────────────────────────────────────────────────────

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le modèle entraîné sur de nouvelles données.

        Returns:
            DataFrame avec score_if [0,1] et anomaly_if (0/1).
        """
        self._check_fitted()
        X        = self._prepare(df_features)
        X_scaled = self.scaler.transform(X)

        raw_scores  = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)

        scores = self._normalize(raw_scores)

        result = df_features.copy()
        result["score_if"]   = np.round(scores, 4)
        result["anomaly_if"] = (predictions == -1).astype(int)
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df.values.astype(float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        inv = -raw
        mn = self._train_min if hasattr(self, "_train_min") and self._train_min is not None else inv.min()
        mx = self._train_max if hasattr(self, "_train_max") and self._train_max is not None else inv.max()
        if mx > mn:
            return np.clip((inv - mn) / (mx - mn), 0.0, 1.0)
        return np.zeros_like(inv)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("IsolationForestDetector : appelle fit() avant predict().")


# ── Fonction utilitaire (rétrocompatibilité scorer legacy) ────────────────────

def run_isolation_forest(df_features: pd.DataFrame) -> pd.DataFrame:
    """Fit + predict en une passe (usage hors API)."""
    detector = IsolationForestDetector()
    detector.fit(df_features)
    return detector.predict(df_features)