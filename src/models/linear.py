import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib
import logging

from src.config import settings
from src.feature_pipeline import build_full_pipeline, get_feature_names

logger = logging.getLogger(__name__)


class LinearModelTrainer:
    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        multi_class: str = "auto"
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.multi_class = multi_class
        self.pipeline_ = None
        self.best_params_ = None
        self.cv_results_ = None

    def _create_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            multi_class=self.multi_class,
            n_jobs=-1
        )

    def build_pipeline(
        self,
        categorical_features=None,
        numerical_features=None,
        add_cluster_feature=False,
        cluster_model=None,
        feature_selection_k=30
    ) -> Pipeline:
        estimator = self._create_estimator()
        pipeline = build_full_pipeline(
            estimator=estimator,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            add_cluster_feature=add_cluster_feature,
            cluster_model=cluster_model,
            feature_selection_k=feature_selection_k
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
        feature_selection_k=30
    ) -> "LinearModelTrainer":
        self.pipeline_ = self.build_pipeline(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            add_cluster_feature=add_cluster_feature,
            cluster_model=cluster_model,
            feature_selection_k=feature_selection_k
        )
        self.pipeline_.fit(X_train, y_train)
        logger.info(f"Linear model ({self.penalty}) fitted successfully")
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

    def get_feature_importance(self) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise ValueError("Model not fitted yet")

        feature_names = get_feature_names(self.pipeline_)
        classifier = self.pipeline_.named_steps["classifier"]

        if hasattr(classifier, "coef_"):
            coef = classifier.coef_
            if coef.ndim > 1:
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef)
        else:
            importance = np.ones(len(feature_names))

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

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
        feature_selection_k=30
    ) -> Dict[str, Any]:
        if param_grid is None:
            if self.penalty == "l1":
                param_grid = {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "classifier__solver": ["saga"],
                    "classifier__penalty": ["l1"]
                }
            else:
                param_grid = {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "classifier__solver": ["lbfgs", "newton-cg"],
                    "classifier__penalty": ["l2"]
                }

        base_pipeline = self.build_pipeline(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            add_cluster_feature=add_cluster_feature,
            cluster_model=cluster_model,
            feature_selection_k=feature_selection_k
        )

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=settings.random_state)

        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
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
            "cv_results": self.cv_results_
        }

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise ValueError("No pipeline to save")
        joblib.dump(self.pipeline_, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LinearModelTrainer":
        pipeline = joblib.load(path)
        instance = cls()
        instance.pipeline_ = pipeline
        return instance


def train_logistic_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    tune: bool = False,
    **kwargs
) -> LinearModelTrainer:
    trainer = LinearModelTrainer(
        penalty="l1",
        C=settings.lr_l1_params.get("C", 1.0),
        solver=settings.lr_l1_params.get("solver", "saga"),
        max_iter=settings.lr_l1_params.get("max_iter", 1000),
        random_state=settings.lr_l1_params.get("random_state", 42),
        multi_class=settings.lr_l1_params.get("multi_class", "auto")
    )

    if tune:
        trainer.tune_hyperparameters(X_train, y_train, **kwargs)
    else:
        trainer.fit(X_train, y_train, **kwargs)

    return trainer


def train_logistic_l2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    tune: bool = False,
    **kwargs
) -> LinearModelTrainer:
    trainer = LinearModelTrainer(
        penalty="l2",
        C=settings.lr_l2_params.get("C", 1.0),
        solver=settings.lr_l2_params.get("solver", "lbfgs"),
        max_iter=settings.lr_l2_params.get("max_iter", 1000),
        random_state=settings.lr_l2_params.get("random_state", 42),
        multi_class=settings.lr_l2_params.get("multi_class", "auto")
    )

    if tune:
        trainer.tune_hyperparameters(X_train, y_train, **kwargs)
    else:
        trainer.fit(X_train, y_train, **kwargs)

    return trainer


def get_linear_models() -> Dict[str, LinearModelTrainer]:
    return {
        "logistic_l1": LinearModelTrainer(**settings.lr_l1_params),
        "logistic_l2": LinearModelTrainer(**settings.lr_l2_params)
    }