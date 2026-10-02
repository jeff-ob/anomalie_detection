"""
dbscan.py
----------
DBSCAN avec séparation fit() / predict().

Pour les nouvelles transactions, on utilise le modèle entraîné pour trouver
le cluster le plus proche (via NearestNeighbors) et décider si la transaction
est un point bruit ou appartient à un cluster suspect.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DBSCANDetector:
    """Wrapper DBSCAN avec état fit/predict séparé."""

    def __init__(self, min_samples: int = 5, min_cluster_size: int = 3):
        self.min_samples      = min_samples
        self.min_cluster_size = min_cluster_size

        self.scaler  = StandardScaler()
        self._fitted = False
        self._labels          = None
        self._X_train_scaled  = None
        self._eps             = None
        self._cluster_sizes   = {}

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(self, df_features: pd.DataFrame) -> "DBSCANDetector":
        """Entraîne DBSCAN sur les données historiques."""
        X = self._prepare(df_features)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self._eps = self._auto_eps(X_scaled, self.min_samples)
        logger.info(f"🔵 DBSCAN — eps auto={self._eps:.4f}, min_samples={self.min_samples}")

        model  = DBSCAN(eps=self._eps, min_samples=self.min_samples, n_jobs=-1)
        labels = model.fit_predict(X_scaled)

        self._labels         = labels
        self._X_train_scaled = X_scaled

        # Taille de chaque cluster
        for lbl in set(labels):
            if lbl != -1:
                self._cluster_sizes[lbl] = (labels == lbl).sum()

        n_clusters = len(self._cluster_sizes)
        n_noise    = (labels == -1).sum()
        logger.info(f"   → {n_clusters} clusters, {n_noise} points bruit ({n_noise/len(labels)*100:.1f}%)")

        # NearestNeighbors pour predict() sur nouvelles données
        self._nbrs = NearestNeighbors(n_neighbors=self.min_samples, n_jobs=-1)
        self._nbrs.fit(X_scaled)

        self._fitted = True
        return self

    # ── Prédiction ───────────────────────────────────────────────────────────

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit le score DBSCAN pour de nouvelles transactions.

        Pour chaque point :
        - Cherche les min_samples voisins les plus proches dans les données d'entraînement
        - Si la distance moyenne > eps → point bruit (score élevé)
        - Si les voisins appartiennent à un petit cluster → score modéré
        - Sinon → score faible
        """
        self._check_fitted()
        X        = self._prepare(df_features)
        X_scaled = self.scaler.transform(X)

        distances, indices = self._nbrs.kneighbors(X_scaled)
        avg_dist = distances.mean(axis=1)

        scores = np.zeros(len(df_features))
        for i, (dist, idx) in enumerate(zip(avg_dist, indices)):
            neighbor_labels = self._labels[idx]
            unique, counts  = np.unique(neighbor_labels[neighbor_labels != -1], return_counts=True)

            if dist > self._eps:
                # Point isolé des clusters connus → bruit
                scores[i] = min(dist / (self._eps * 2), 1.0)
            elif len(unique) == 0:
                # Tous les voisins sont du bruit
                scores[i] = 0.8
            else:
                dominant_label = unique[counts.argmax()]
                cluster_size   = self._cluster_sizes.get(dominant_label, 0)
                if cluster_size < self._min_cluster_size:
                    scores[i] = 0.6
                else:
                    scores[i] = min(dist / self._eps * 0.4, 0.4)

        threshold = np.percentile(scores, 95)
        anomalies = (scores >= threshold).astype(int)

        result = df_features.copy()
        result["score_dbscan"]   = np.round(scores, 4)
        result["anomaly_dbscan"] = anomalies
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df.values.astype(float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _auto_eps(self, X_scaled: np.ndarray, k: int) -> float:
        k    = min(k, X_scaled.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_scaled)
        dists, _ = nbrs.kneighbors(X_scaled)
        return float(max(np.percentile(dists[:, -1], 50), 0.5))

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("DBSCANDetector : appelle fit() avant predict().")

    @property
    def _min_cluster_size(self):
        return self.min_cluster_size


# ── Fonction utilitaire (rétrocompatibilité) ──────────────────────────────────

def run_dbscan(df_features: pd.DataFrame) -> pd.DataFrame:
    detector = DBSCANDetector()
    detector.fit(df_features)
    return detector.predict(df_features)