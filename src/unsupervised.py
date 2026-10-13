import logging
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)


def _extract_numeric(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Helper to ensure only numeric data reaches StandardScaler, PCA, and Clustering."""
    if isinstance(X, pd.DataFrame):
        numeric_df = X.select_dtypes(include=[np.number])
        return numeric_df.to_numpy(dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 30, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "PCATransformer":
        X_clean = _extract_numeric(X)
        X_scaled = self.scaler_.fit_transform(X_clean)
        
        # Adjust n_components if features are fewer than target n_components
        actual_components = min(self.n_components, X_scaled.shape[1])
        self.pca_ = PCA(n_components=actual_components, random_state=self.random_state)
        self.pca_.fit(X_scaled)
        
        logger.info(
            f"PCA fitted with {actual_components} components, "
            f"explained variance: {self.pca_.explained_variance_ratio_.sum():.4f}"
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.pca_ is None:
            raise ValueError("PCATransformer not fitted yet")
        X_clean = _extract_numeric(X)
        X_scaled = self.scaler_.transform(X_clean)
        return self.pca_.transform(X_scaled)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_explained_variance_ratio(self) -> np.ndarray:
        if self.pca_ is None:
            raise ValueError("PCA not fitted yet")
        return self.pca_.explained_variance_ratio_

    def get_components(self) -> np.ndarray:
        if self.pca_ is None:
            raise ValueError("PCA not fitted yet")
        return self.pca_.components_


class KMeansClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 10, random_state: int = 42, n_init: int = 10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.kmeans_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "KMeansClusterer":
        X_clean = _extract_numeric(X)
        X_scaled = self.scaler_.fit_transform(X_clean)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init
        )
        self.kmeans_.fit(X_scaled)
        logger.info(
            f"KMeans fitted with {self.n_clusters} clusters, "
            f"inertia: {self.kmeans_.inertia_:.4f}"
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.kmeans_ is None:
            raise ValueError("KMeansClusterer not fitted yet")
        X_clean = _extract_numeric(X)
        X_scaled = self.scaler_.transform(X_clean)
        return self.kmeans_.predict(X_scaled).reshape(-1, 1)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.transform(X).flatten()

    def get_cluster_centers(self) -> np.ndarray:
        if self.kmeans_ is None:
            raise ValueError("KMeans not fitted yet")
        return self.kmeans_.cluster_centers_

    def get_inertia(self) -> float:
        if self.kmeans_ is None:
            raise ValueError("KMeans not fitted yet")
        return self.kmeans_.inertia_


class DBSCANAnomalyDetector(BaseEstimator, TransformerMixin):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "DBSCANAnomalyDetector":
        X_clean = _extract_numeric(X)
        # Sample to prevent out-of-memory or freezing on huge datasets (e.g. 100k+ rows)
        if len(X_clean) > 10000:
            np.random.seed(42)
            idx = np.random.choice(len(X_clean), size=10000, replace=False)
            X_fit = X_clean[idx]
        else:
            X_fit = X_clean

        X_scaled = self.scaler_.fit_transform(X_fit)
        self.dbscan_ = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        self.dbscan_.fit(X_scaled)
        
        n_clusters = len(set(self.dbscan_.labels_)) - (1 if -1 in self.dbscan_.labels_ else 0)
        n_noise = list(self.dbscan_.labels_).count(-1)
        logger.info(f"DBSCAN fitted: {n_clusters} clusters, {n_noise} noise points")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.dbscan_ is None:
            raise ValueError("DBSCANAnomalyDetector not fitted yet")
        X_clean = _extract_numeric(X)
        X_scaled = self.scaler_.transform(X_clean)
        # Approximate outlier distance from core samples
        if hasattr(self.dbscan_, "components_") and len(self.dbscan_.components_) > 0:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1).fit(self.dbscan_.components_)
            distances, _ = nn.kneighbors(X_scaled)
            anomaly_scores = (distances > self.eps).astype(int)
        else:
            anomaly_scores = np.zeros((len(X_clean), 1), dtype=int)
        return anomaly_scores.reshape(-1, 1)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.transform(X).flatten()


class UnsupervisedPipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_pca_components: int = 30,
        n_clusters: int = 10,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        random_state: int = 42
    ):
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state

        self.pca_ = PCATransformer(n_components=n_pca_components, random_state=random_state)
        self.kmeans_ = KMeansClusterer(n_clusters=n_clusters, random_state=random_state)
        self.dbscan_ = DBSCANAnomalyDetector(eps=dbscan_eps, min_samples=dbscan_min_samples)

        self.is_fitted_ = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "UnsupervisedPipeline":
        logger.info("Fitting unsupervised pipeline...")
        self.pca_.fit(X)
        X_pca = self.pca_.transform(X)

        self.kmeans_.fit(X_pca)
        self.dbscan_.fit(X_pca)

        self.is_fitted_ = True
        logger.info("Unsupervised pipeline fitted successfully")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("UnsupervisedPipeline not fitted yet")

        X_pca = self.pca_.transform(X)
        kmeans_labels = self.kmeans_.predict(X_pca)
        dbscan_scores = self.dbscan_.predict(X_pca)

        index = X.index if isinstance(X, pd.DataFrame) else range(len(X_pca))
        result = pd.DataFrame(index=index)
        
        num_components = X_pca.shape[1]
        for i in range(num_components):
            result[f"pca_{i}"] = X_pca[:, i]
            
        result["kmeans_cluster"] = kmeans_labels
        result["dbscan_anomaly"] = dbscan_scores

        return result

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_pca_explained_variance(self) -> np.ndarray:
        return self.pca_.get_explained_variance_ratio()

    def get_kmeans_inertia(self) -> float:
        return self.kmeans_.get_inertia()


def train_unsupervised_models(
    X_train: pd.DataFrame,
    n_pca_components: int = 30,
    n_clusters: int = 10,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    pipeline = UnsupervisedPipeline(
        n_pca_components=n_pca_components,
        n_clusters=n_clusters,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        random_state=random_state
    )

    pipeline.fit(X_train)

    return {
        "pipeline": pipeline,
        "pca": pipeline.pca_,
        "kmeans": pipeline.kmeans_,
        "dbscan": pipeline.dbscan_,
        "pca_explained_variance": pipeline.get_pca_explained_variance(),
        "kmeans_inertia": pipeline.get_kmeans_inertia()
    }


def save_unsupervised_models(models: Dict[str, Any], path: str) -> None:
    joblib.dump(models, path)


def load_unsupervised_models(path: str) -> Dict[str, Any]:
    return joblib.load(path)