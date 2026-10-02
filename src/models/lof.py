"""
lof.py
-------
Local Outlier Factor avec séparation fit() / predict().

Note : LOF de sklearn ne supporte pas nativement le predict() sur de nouvelles
données sans novelty=True. On utilise novelty=True pour pouvoir scorer
des transactions individuelles après entraînement.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.config import LOF_N_NEIGHBORS, LOF_CONTAMINATION
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LOFDetector:
    """Wrapper LOF avec état fit/predict séparé."""

    def __init__(
        self,
        n_neighbors:   int   = LOF_N_NEIGHBORS,
        contamination: float = LOF_CONTAMINATION,
    ):
        self.n_neighbors   = n_neighbors
        self.contamination = contamination

        self.scaler = StandardScaler()
        # novelty=True : permet predict() sur nouvelles données
        self.model  = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )
        self._fitted      = False
        self._train_scores = None  # scores sur données d'entraînement

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(self, df_features: pd.DataFrame) -> "LOFDetector":
        """Entraîne le modèle sur les données historiques."""
        X = self._prepare(df_features)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)
        self._fitted = True

        # Scores sur les données d'entraînement pour la normalisation future
        raw = self.model.score_samples(X_scaled)
        self._train_min = (-raw).min()
        self._train_max = (-raw).max()

        n = len(df_features)
        logger.info(f"🔭 LOF entraîné sur {n:,} échantillons (n_neighbors={self.n_neighbors})")
        return self

    # ── Prédiction ───────────────────────────────────────────────────────────

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Applique le modèle entraîné sur de nouvelles données."""
        self._check_fitted()
        X        = self._prepare(df_features)
        X_scaled = self.scaler.transform(X)

        raw_scores  = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)
        scores      = self._normalize(raw_scores)

        result = df_features.copy()
        result["score_lof"]   = np.round(scores, 4)
        result["anomaly_lof"] = (predictions == -1).astype(int)
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df.values.astype(float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        inv = -raw
        mn  = self._train_min if self._train_min is not None else inv.min()
        mx  = self._train_max if self._train_max is not None else inv.max()
        if mx > mn:
            return np.clip((inv - mn) / (mx - mn), 0.0, 1.0)
        return np.zeros_like(inv)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("LOFDetector : appelle fit() avant predict().")


# ── Fonction utilitaire (rétrocompatibilité) ──────────────────────────────────

def run_lof(df_features: pd.DataFrame) -> pd.DataFrame:
    """Fit + predict en une passe (usage hors API)."""
    detector = LOFDetector()
    detector.fit(df_features)
    return detector.predict(df_features)