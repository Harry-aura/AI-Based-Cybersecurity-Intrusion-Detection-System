import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.data_ingestion import (
    ingest_pipeline,
    load_processed_data,
    prepare_features_targets,
)
from src.evaluation import (
    compare_models_mcnemar,
    evaluate_model_on_splits,
    summarize_evaluation_results,
)
from src.feature_pipeline import build_preprocessing_pipeline, save_pipeline
from src.models.linear import get_linear_models, train_logistic_l1, train_logistic_l2
from src.models.tree_based import (
    get_tree_models,
    train_random_forest,
    train_xgboost_binary,
    train_xgboost_multiclass,
)
from src.unsupervised import save_unsupervised_models, train_unsupervised_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _to_json_serializable(obj: Any) -> Any:
    """Helper to convert NumPy/Pandas scalar types into standard Python types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


class ModelTrainer:
    def __init__(
        self,
        artifacts_dir: Optional[Path] = None,
        random_state: int = 42,
    ):
        self.artifacts_dir = artifacts_dir or settings.model_artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.X_train = None
        self.y_train_binary = None
        self.y_train_multiclass = None
        self.X_val = None
        self.y_val_binary = None
        self.y_val_multiclass = None
        self.X_test = None
        self.y_test_binary = None
        self.y_test_multiclass = None

        self.unsupervised_models = None
        self.preprocessing_pipeline = None

        self.trained_models_binary = {}
        self.trained_models_multiclass = {}

        self.evaluation_results_binary = {}
        self.evaluation_results_multiclass = {}

    def load_data(self, force_download: bool = False) -> "ModelTrainer":
        logger.info("Loading data...")
        self.train_df, self.val_df, self.test_df = ingest_pipeline(
            force_download=force_download
        )

        self.X_train, self.y_train_binary = prepare_features_targets(
            self.train_df, "binary"
        )
        _, self.y_train_multiclass = prepare_features_targets(
            self.train_df, "multiclass"
        )

        self.X_val, self.y_val_binary = prepare_features_targets(
            self.val_df, "binary"
        )
        _, self.y_val_multiclass = prepare_features_targets(
            self.val_df, "multiclass"
        )

        self.X_test, self.y_test_binary = prepare_features_targets(
            self.test_df, "binary"
        )
        _, self.y_test_multiclass = prepare_features_targets(
            self.test_df, "multiclass"
        )

        logger.info(
            f"Data loaded - Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}"
        )
        return self

    def train_unsupervised(self) -> "ModelTrainer":
        logger.info("Training unsupervised models...")
        self.unsupervised_models = train_unsupervised_models(
            self.X_train,
            n_pca_components=settings.pca_components,
            n_clusters=settings.kmeans_clusters,
            dbscan_eps=settings.dbscan_eps,
            dbscan_min_samples=settings.dbscan_min_samples,
            random_state=self.random_state,
        )

        save_unsupervised_models(
            self.unsupervised_models,
            str(self.artifacts_dir / "unsupervised_models.joblib"),
        )

        self.preprocessing_pipeline = build_preprocessing_pipeline(
            add_cluster_feature=True,
            cluster_model=self.unsupervised_models["kmeans"],
            feature_selection_k=settings.pca_components,
        )
        self.preprocessing_pipeline.fit(self.X_train, self.y_train_binary)

        save_pipeline(
            self.preprocessing_pipeline,
            str(self.artifacts_dir / "preprocessing_pipeline.joblib"),
        )

        logger.info("Unsupervised models trained and saved")
        return self

    def train_binary_models(self, tune: bool = False) -> "ModelTrainer":
        logger.info("Training binary classification models...")
        kmeans_model = (
            self.unsupervised_models["kmeans"]
            if self.unsupervised_models
            else None
        )

        lr_l1 = train_logistic_l1(
            self.X_train,
            self.y_train_binary,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_binary["logistic_l1"] = lr_l1

        lr_l2 = train_logistic_l2(
            self.X_train,
            self.y_train_binary,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_binary["logistic_l2"] = lr_l2

        rf = train_random_forest(
            self.X_train,
            self.y_train_binary,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_binary["random_forest"] = rf

        xgb_bin = train_xgboost_binary(
            self.X_train,
            self.y_train_binary,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_binary["xgboost"] = xgb_bin

        self._save_binary_models()
        logger.info("Binary models trained and saved")
        return self

    def train_multiclass_models(self, tune: bool = False) -> "ModelTrainer":
        logger.info("Training multi-class classification models...")
        kmeans_model = (
            self.unsupervised_models["kmeans"]
            if self.unsupervised_models
            else None
        )

        lr_l1 = train_logistic_l1(
            self.X_train,
            self.y_train_multiclass,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_multiclass["logistic_l1"] = lr_l1

        lr_l2 = train_logistic_l2(
            self.X_train,
            self.y_train_multiclass,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_multiclass["logistic_l2"] = lr_l2

        rf = train_random_forest(
            self.X_train,
            self.y_train_multiclass,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_multiclass["random_forest"] = rf

        xgb_multi = train_xgboost_multiclass(
            self.X_train,
            self.y_train_multiclass,
            tune=tune,
            add_cluster_feature=True,
            cluster_model=kmeans_model,
            feature_selection_k=settings.pca_components,
        )
        self.trained_models_multiclass["xgboost"] = xgb_multi

        self._save_multiclass_models()
        logger.info("Multi-class models trained and saved")
        return self

    def _save_binary_models(self) -> None:
        for name, model in self.trained_models_binary.items():
            model.save(str(self.artifacts_dir / f"binary_{name}.joblib"))

    def _save_multiclass_models(self) -> None:
        for name, model in self.trained_models_multiclass.items():
            model.save(str(self.artifacts_dir / f"multiclass_{name}.joblib"))

    def evaluate_all(self) -> "ModelTrainer":
        logger.info("Evaluating all models...")

        multiclass_labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]

        for name, model in self.trained_models_binary.items():
            logger.info(f"Evaluating binary model: {name}")
            self.evaluation_results_binary[name] = evaluate_model_on_splits(
                model.pipeline_,
                self.X_train,
                self.y_train_binary,
                self.X_val,
                self.y_val_binary,
                self.X_test,
                self.y_test_binary,
                task_type="binary",
            )

        for name, model in self.trained_models_multiclass.items():
            logger.info(f"Evaluating multiclass model: {name}")
            self.evaluation_results_multiclass[name] = evaluate_model_on_splits(
                model.pipeline_,
                self.X_train,
                self.y_train_multiclass,
                self.X_val,
                self.y_val_multiclass,
                self.X_test,
                self.y_test_multiclass,
                task_type="multiclass",
                labels=multiclass_labels,
            )

        self._run_mcnemar_tests()
        self._save_evaluation_results()
        logger.info("Evaluation completed")
        return self

    def _run_mcnemar_tests(self) -> None:
        logger.info("Running McNemar's tests...")

        best_binary_model = self._get_best_binary_model()
        linear_model = self.trained_models_binary.get("logistic_l2")

        if best_binary_model and linear_model:
            y_pred_best = best_binary_model.predict(self.X_test)
            y_pred_linear = linear_model.predict(self.X_test)

            result = compare_models_mcnemar(
                "Best_Ensemble",
                "Logistic_L2",
                self.y_test_binary.values,
                y_pred_best,
                y_pred_linear,
            )

            with open(self.artifacts_dir / "mcnemar_test.json", "w") as f:
                json.dump(result, f, indent=2, default=_to_json_serializable)

            logger.info(
                f"McNemar's test: {result['conclusion']} (p={result['p_value']:.6f})"
            )

    def _get_best_binary_model(self):
        best_score = -1
        best_model = None
        for name, model in self.trained_models_binary.items():
            score = model.score(self.X_val, self.y_val_binary)
            if score > best_score:
                best_score = score
                best_model = model
        return best_model

    def _save_evaluation_results(self) -> None:
        summary_binary = summarize_evaluation_results(
            self.evaluation_results_binary, "binary"
        )
        summary_multiclass = summarize_evaluation_results(
            self.evaluation_results_multiclass, "multiclass"
        )

        summary_binary.to_csv(
            self.artifacts_dir / "evaluation_summary_binary.csv", index=False
        )
        summary_multiclass.to_csv(
            self.artifacts_dir / "evaluation_summary_multiclass.csv", index=False
        )

        with open(
            self.artifacts_dir / "evaluation_results_binary.json", "w"
        ) as f:
            json.dump(
                self.evaluation_results_binary,
                f,
                indent=2,
                default=_to_json_serializable,
            )

        with open(
            self.artifacts_dir / "evaluation_results_multiclass.json", "w"
        ) as f:
            json.dump(
                self.evaluation_results_multiclass,
                f,
                indent=2,
                default=_to_json_serializable,
            )

    def select_best_models(self) -> Tuple[Any, Any]:
        best_binary = self._get_best_binary_model()

        best_multi = None
        best_multi_score = -1
        for name, model in self.trained_models_multiclass.items():
            score = model.score(self.X_val, self.y_val_multiclass)
            if score > best_multi_score:
                best_multi_score = score
                best_multi = model

        if best_binary:
            joblib.dump(
                best_binary.pipeline_,
                self.artifacts_dir / "best_binary_model.joblib",
            )

        if best_multi:
            joblib.dump(
                best_multi.pipeline_,
                self.artifacts_dir / "best_multiclass_model.joblib",
            )

        metadata = {
            "best_binary_model": (
                type(best_binary).__name__ if best_binary else None
            ),
            "best_multiclass_model": (
                type(best_multi).__name__ if best_multi else None
            ),
            "training_date": datetime.now().isoformat(),
            "random_state": self.random_state,
        }

        with open(self.artifacts_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=_to_json_serializable)

        return best_binary, best_multi


def run_full_training(
    force_download: bool = False,
    tune_hyperparameters: bool = False,
    artifacts_dir: Optional[Path] = None,
) -> ModelTrainer:
    trainer = ModelTrainer(artifacts_dir=artifacts_dir)
    trainer.load_data(force_download=force_download)
    trainer.train_unsupervised()
    trainer.train_binary_models(tune=tune_hyperparameters)
    trainer.train_multiclass_models(tune=tune_hyperparameters)
    trainer.evaluate_all()
    trainer.select_best_models()
    return trainer


if __name__ == "__main__":
    trainer = run_full_training(force_download=False, tune_hyperparameters=False)
    print("Training completed successfully!")