import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.config import settings
from src.feature_pipeline import build_full_pipeline, get_feature_names

logger = logging.getLogger(__name__)


class TreeModelTrainer:
    def __init__(
        self,
        model_type: str = "rf",
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.params = params or {}
        self.random_state = random_state
        self.pipeline_ = None
        self.best_params_ = None
        self.cv_results_ = None

    def _create_estimator(self):
        if self.model_type == "rf":
            default_params = settings.rf_params.copy()
            default_params.update(self.params)
            default_params["random_state"] = self.random_state
            return RandomForestClassifier(**default_params)
        elif self.model_type == "xgb":
            default_params = settings.xgb_params_binary.copy()
            default_params.update(self.params)
            default_params["random_state"] = self.random_state
            return xgb.XGBClassifier(**default_params)
        elif self.model_type == "xgb_multiclass":
            default_params = settings.xgb_params_multiclass.copy()
            default_params.update(self.params)
            default_params["random_state"] = self.random_state
            return xgb.XGBClassifier(**default_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def build_pipeline(
        self,
        categorical_features=None,
        numerical_features=None,
        add_cluster_feature=False,
        cluster_model=None,
        feature_selection_k=30,
    ) -> Pipeline:
        estimator = self._create_estimator()
        pipeline = build_full_pipeline(
            estimator=estimator,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            add_cluster_feature=add_cluster_feature,
            cluster_model=cluster_model,
            feature_selection_k=feature_selection_k,
        )
        return pipeline

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_features=None,
        numerical_features=None,
        add_cluster_feature=False,
        cluster_model=None,
        feature_selection_k=30,
    ) -> "TreeModelTrainer":
        self.pipeline_ = self.build_pipeline(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            add_cluster_feature=add_cluster_feature,
            cluster_model=cluster_model,
            feature_selection_k=feature_selection_k,
        )
        self.pipeline_.fit(X_train, y_train)
        logger.info(f"{self.model_type.upper()} model fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise ValueError("Model not fitted yet")
        return self.pipeline_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise ValueError("Model not fitted yet")
        return self.pipeline_.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self.pipeline_ is None:
            raise ValueError("Model not fitted yet")
        return self.pipeline_.score(X, y)

    def get_feature_importance_gini(self) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise ValueError("Model not fitted yet")

        feature_names = get_feature_names(self.pipeline_)
        classifier = self.pipeline_.named_steps["classifier"]

        if hasattr(classifier, "feature_importances_"):
            importance = classifier.feature_importances_
        else:
            importance = np.ones(len(feature_names))

        return pd.DataFrame(
            {
                "feature": feature_names,
                "gini_importance": importance,
            }
        ).sort_values("gini_importance", ascending=False)

    def get_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise ValueError("Model not fitted yet")

        feature_names = get_feature_names(self.pipeline_)
        X_transformed = self.pipeline_.named_steps["preprocessing"].transform(X)

        result = permutation_importance(
            self.pipeline_.named_steps["classifier"],
            X_transformed,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        return pd.DataFrame(
            {
                "feature": feature_names,
                "perm_importance_mean": result.importances_mean,
                "perm_importance_std": result.importances_std,
            }
        ).sort_values("perm_importance_mean", ascending=False)

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        cv: int = 5,
        scoring: str = "f1_weighted",
        categorical_features=None,
        numerical_features=None,
        add_cluster_feature=False,
        cluster_model=None,
        feature_selection_k=30,
    ) -> Dict[str, Any]:
        if param_grid is None:
            if self.model_type == "rf":
                param_grid = {
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__max_depth": [10, 20, None],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                }
            elif self.model_type in ("xgb", "xgb_multiclass"):
                param_grid = {
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__max_depth": [6, 8, 10],
                    "classifier__learning_rate": [0.01, 0.1, 0.2],
                    "classifier__subsample": [0.8, 1.0],
                    "classifier__colsample_bytree": [0.8, 1.0],
                }

        base_pipeline = self.build_pipeline(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            add_cluster_feature=add_cluster_feature,
            cluster_model=cluster_model,
            feature_selection_k=feature_selection_k,
        )

        cv_strategy = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=settings.random_state
        )

        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)

        self.pipeline_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_

        logger.info(f"Best params: {self.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return {
            "best_params": self.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": self.cv_results_,
        }

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise ValueError("No pipeline to save")
        joblib.dump(self.pipeline_, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TreeModelTrainer":
        pipeline = joblib.load(path)
        instance = cls()
        instance.pipeline_ = pipeline
        return instance


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    tune: bool = False,
    **kwargs,
) -> TreeModelTrainer:
    trainer = TreeModelTrainer(model_type="rf", params=settings.rf_params)
    if tune:
        trainer.tune_hyperparameters(X_train, y_train, **kwargs)
    else:
        trainer.fit(X_train, y_train, **kwargs)
    return trainer


def train_xgboost_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    tune: bool = False,
    **kwargs,
) -> TreeModelTrainer:
    trainer = TreeModelTrainer(model_type="xgb", params=settings.xgb_params_binary)
    if tune:
        trainer.tune_hyperparameters(X_train, y_train, **kwargs)
    else:
        trainer.fit(X_train, y_train, **kwargs)
    return trainer


def train_xgboost_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    tune: bool = False,
    **kwargs,
) -> TreeModelTrainer:
    trainer = TreeModelTrainer(
        model_type="xgb_multiclass", params=settings.xgb_params_multiclass
    )
    if tune:
        trainer.tune_hyperparameters(X_train, y_train, **kwargs)
    else:
        trainer.fit(X_train, y_train, **kwargs)
    return trainer


def get_tree_models() -> Dict[str, TreeModelTrainer]:
    return {
        "random_forest": TreeModelTrainer(
            model_type="rf", params=settings.rf_params
        ),
        "xgboost_binary": TreeModelTrainer(
            model_type="xgb", params=settings.xgb_params_binary
        ),
        "xgboost_multiclass": TreeModelTrainer(
            model_type="xgb_multiclass", params=settings.xgb_params_multiclass
        ),
    }