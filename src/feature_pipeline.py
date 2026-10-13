import logging
from typing import Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "f_classif", k: int = 30):
        self.method = method
        self.k = k
        self.selector_ = None
        self.selected_indices_ = None
        self.feature_names_in_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any) -> "FeatureSelector":
        n_features = X.shape[1]
        k_val = min(self.k, n_features)

        if self.method == "f_classif":
            self.selector_ = SelectKBest(score_func=f_classif, k=k_val)
        elif self.method == "mutual_info":
            self.selector_ = SelectKBest(score_func=mutual_info_classif, k=k_val)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.selector_.fit(X, y)
        self.selected_indices_ = np.where(self.selector_.get_support())[0]

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
        elif hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"feature_{i}" for i in range(n_features)])

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.selector_ is None:
            raise ValueError("FeatureSelector not fitted yet")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_]
        return X[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        if self.selected_indices_ is None:
            raise ValueError("FeatureSelector not fitted yet")
        if input_features is not None:
            return [input_features[i] for i in self.selected_indices_]
        if self.feature_names_in_ is not None:
            return [self.feature_names_in_[i] for i in self.selected_indices_]
        return [f"feature_{i}" for i in self.selected_indices_]


class ClusterFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_model=None):
        self.cluster_model = cluster_model

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> "ClusterFeatureAdder":
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.cluster_model is None or not hasattr(self.cluster_model, "predict"):
            return X

        # Adapt to whether cluster_model expects raw numeric columns or full encoded matrix
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            # If the cluster model was fitted on fewer features (e.g., raw numeric), slice them
            expected_features = getattr(self.cluster_model, "n_features_in_", None)
            if expected_features is not None and expected_features != X.shape[1]:
                numeric_df = X.select_dtypes(include=[np.number])
                if numeric_df.shape[1] >= expected_features:
                    X_input = numeric_df.iloc[:, :expected_features]
                else:
                    return X
            else:
                X_input = X

            try:
                clusters = self.cluster_model.predict(X_input)
                X_out["kmeans_cluster"] = clusters
                return X_out
            except Exception as e:
                logger.warning(f"ClusterFeatureAdder skipped due to dimension mismatch: {e}")
                return X

        # Numpy array input
        expected_features = getattr(self.cluster_model, "n_features_in_", None)
        if expected_features is not None and expected_features != X.shape[1]:
            if X.shape[1] >= expected_features:
                X_input = X[:, :expected_features]
            else:
                return X
        else:
            X_input = X

        try:
            clusters = self.cluster_model.predict(X_input).reshape(-1, 1)
            return np.hstack([X, clusters])
        except Exception as e:
            logger.warning(f"ClusterFeatureAdder skipped due to dimension mismatch: {e}")
            return X


def build_preprocessing_pipeline(
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    add_cluster_feature: bool = False,
    cluster_model=None,
    feature_selection_k: int = 30
) -> Pipeline:
    cat_features = categorical_features or settings.categorical_features
    num_features = numerical_features or settings.numerical_features

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numerical_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_features),
            ("num", numerical_transformer, num_features)
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False
    )

    steps = [("preprocessor", preprocessor)]

    if add_cluster_feature and cluster_model is not None:
        steps.append(("cluster_adder", ClusterFeatureAdder(cluster_model)))

    steps.append(("feature_selector", FeatureSelector(method="f_classif", k=feature_selection_k)))

    return Pipeline(steps)


def build_full_pipeline(
    estimator,
    categorical_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    add_cluster_feature: bool = False,
    cluster_model=None,
    feature_selection_k: int = 30
) -> Pipeline:
    preprocessing = build_preprocessing_pipeline(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        add_cluster_feature=add_cluster_feature,
        cluster_model=cluster_model,
        feature_selection_k=feature_selection_k
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", estimator)
    ])


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)


def get_feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps.get("preprocessing", pipeline)
    if hasattr(preprocessor, "named_steps") and "feature_selector" in preprocessor.named_steps:
        feature_selector = preprocessor.named_steps["feature_selector"]
        return feature_selector.get_feature_names_out()
    return []


def get_preprocessor_feature_names(pipeline: Pipeline) -> List[str]:
    return get_feature_names(pipeline)


class DriftDetector(BaseEstimator, TransformerMixin):
    def __init__(self, reference_data: pd.DataFrame, features: List[str], alpha: float = 0.05):
        valid_features = [f for f in features if f in reference_data.columns]
        self.reference_data = reference_data[valid_features].copy()
        self.features = valid_features
        self.alpha = alpha
        self.drift_detected_ = False
        self.p_values_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DriftDetector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.drift_detected_ = False
        self.p_values_ = {}

        for feature in self.features:
            if feature in X.columns and feature in self.reference_data.columns:
                ref_series = self.reference_data[feature].dropna()
                cur_series = X[feature].dropna()
                if len(ref_series) > 0 and len(cur_series) > 0:
                    _, p_val = stats.ks_2samp(ref_series, cur_series)
                    self.p_values_[feature] = float(p_val)
                    if p_val < self.alpha:
                        self.drift_detected_ = True

        return X

    def check_drift(self, X: pd.DataFrame) -> Tuple[bool, dict]:
        self.transform(X)
        return self.drift_detected_, self.p_values_