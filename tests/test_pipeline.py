import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

from src.config import settings
from src.feature_pipeline import (
    build_preprocessing_pipeline, build_full_pipeline,
    FeatureSelector, ClusterFeatureAdder, DriftDetector
)
from src.unsupervised import (
    PCATransformer, KMeansClusterer, DBSCANAnomalyDetector, UnsupervisedPipeline
)
from src.models.linear import LinearModelTrainer
from src.models.tree_based import TreeModelTrainer
from src.evaluation import (
    compute_binary_metrics, compute_multiclass_metrics,
    mcnemar_test, stratified_cv_evaluation
)


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def sample_data_multiclass():
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=5, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def nsl_kdd_like_data():
    np.random.seed(42)
    n_samples = 500
    data = {
        "duration": np.random.randint(0, 1000, n_samples),
        "protocol_type": np.random.choice(["tcp", "udp", "icmp"], n_samples),
        "service": np.random.choice(["http", "smtp", "ftp", "dns"], n_samples),
        "flag": np.random.choice(["SF", "S0", "REJ", "RSTR"], n_samples),
        "src_bytes": np.random.randint(0, 10000, n_samples),
        "dst_bytes": np.random.randint(0, 10000, n_samples),
        "label": np.random.choice(["normal", "dos", "probe", "r2l", "u2r"], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05])
    }
    df = pd.DataFrame(data)
    return df


class TestFeaturePipeline:
    def test_build_preprocessing_pipeline(self, sample_data):
        X, y = sample_data
        X_cat = X.copy()
        X_cat["protocol_type"] = np.random.choice(["tcp", "udp"], len(X))
        X_cat["service"] = np.random.choice(["http", "ftp"], len(X))
        X_cat["flag"] = np.random.choice(["SF", "S0"], len(X))

        cat_features = ["protocol_type", "service", "flag"]
        num_features = [c for c in X_cat.columns if c not in cat_features]

        pipeline = build_preprocessing_pipeline(
            categorical_features=cat_features,
            numerical_features=num_features,
            feature_selection_k=10
        )

        pipeline.fit(X_cat, y)
        X_transformed = pipeline.transform(X_cat)

        assert X_transformed.shape[1] <= 10
        assert isinstance(pipeline, Pipeline)

    def test_build_full_pipeline(self, sample_data):
        X, y = sample_data
        from sklearn.linear_model import LogisticRegression

        pipeline = build_full_pipeline(
            estimator=LogisticRegression(random_state=42),
            feature_selection_k=10
        )

        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        probas = pipeline.predict_proba(X)

        assert len(preds) == len(X)
        assert probas.shape == (len(X), 2)

    def test_feature_selector(self, sample_data):
        X, y = sample_data
        selector = FeatureSelector(method="f_classif", k=5)
        selector.fit(X, y)
        X_selected = selector.transform(X)

        assert X_selected.shape[1] == 5
        assert len(selector.selected_features_) == 5

    def test_cluster_feature_adder(self, sample_data):
        X, y = sample_data
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)

        adder = ClusterFeatureAdder(cluster_model=kmeans)
        X_with_cluster = adder.transform(X)

        assert "kmeans_cluster" in X_with_cluster.columns
        assert X_with_cluster["kmeans_cluster"].nunique() <= 3

    def test_drift_detector(self, sample_data):
        X, y = sample_data
        reference = X.iloc[:100]
        current = X.iloc[100:200]

        detector = DriftDetector(reference, features=X.columns.tolist()[:5], alpha=0.05)
        drift_detected, p_values = detector.check_drift(current)

        assert isinstance(drift_detected, bool)
        assert isinstance(p_values, dict)
        assert len(p_values) <= 5


class TestUnsupervisedPipeline:
    def test_pca_transformer(self, sample_data):
        X, y = sample_data
        pca = PCATransformer(n_components=5, random_state=42)
        pca.fit(X)
        X_pca = pca.transform(X)

        assert X_pca.shape == (len(X), 5)
        assert len(pca.get_explained_variance_ratio()) == 5

    def test_kmeans_clusterer(self, sample_data):
        X, y = sample_data
        kmeans = KMeansClusterer(n_clusters=5, random_state=42)
        kmeans.fit(X)
        clusters = kmeans.predict(X)

        assert len(clusters) == len(X)
        assert clusters.min() >= 0
        assert clusters.max() < 5

    def test_dbscan_anomaly_detector(self, sample_data):
        X, y = sample_data
        dbscan = DBSCANAnomalyDetector(eps=1.0, min_samples=5)
        dbscan.fit(X)
        anomalies = dbscan.predict(X)

        assert len(anomalies) == len(X)
        assert set(anomalies).issubset({0, 1})

    def test_unsupervised_pipeline(self, sample_data):
        X, y = sample_data
        pipeline = UnsupervisedPipeline(
            n_pca_components=5,
            n_clusters=3,
            dbscan_eps=1.0,
            dbscan_min_samples=5,
            random_state=42
        )
        pipeline.fit(X)
        result = pipeline.transform(X)

        assert "kmeans_cluster" in result.columns
        assert "dbscan_anomaly" in result.columns
        assert result.shape[0] == len(X)


class TestLinearModels:
    def test_logistic_l1(self, sample_data):
        X, y = sample_data
        trainer = LinearModelTrainer(penalty="l1", C=1.0, solver="saga", max_iter=500)
        trainer.fit(X, y)

        preds = trainer.predict(X)
        probas = trainer.predict_proba(X)

        assert len(preds) == len(X)
        assert probas.shape == (len(X), 2)

    def test_logistic_l2(self, sample_data):
        X, y = sample_data
        trainer = LinearModelTrainer(penalty="l2", C=1.0, solver="lbfgs", max_iter=500)
        trainer.fit(X, y)

        preds = trainer.predict(X)
        assert len(preds) == len(X)

    def test_feature_importance(self, sample_data):
        X, y = sample_data
        trainer = LinearModelTrainer(penalty="l2", max_iter=500)
        trainer.fit(X, y)

        importance = trainer.get_feature_importance()
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) <= X.shape[1]


class TestTreeModels:
    def test_random_forest(self, sample_data):
        X, y = sample_data
        trainer = TreeModelTrainer(model_type="rf", params={"n_estimators": 10, "max_depth": 5})
        trainer.fit(X, y)

        preds = trainer.predict(X)
        probas = trainer.predict_proba(X)

        assert len(preds) == len(X)
        assert probas.shape == (len(X), 2)

    def test_xgboost_binary(self, sample_data):
        X, y = sample_data
        trainer = TreeModelTrainer(model_type="xgb", params={"n_estimators": 10, "max_depth": 3})
        trainer.fit(X, y)

        preds = trainer.predict(X)
        assert len(preds) == len(X)

    def test_xgboost_multiclass(self, sample_data_multiclass):
        X, y = sample_data_multiclass
        trainer = TreeModelTrainer(model_type="xgb_multiclass", params={"n_estimators": 10, "max_depth": 3})
        trainer.fit(X, y)

        preds = trainer.predict(X)
        probas = trainer.predict_proba(X)

        assert len(preds) == len(X)
        assert probas.shape == (len(X), 5)

    def test_gini_importance(self, sample_data):
        X, y = sample_data
        trainer = TreeModelTrainer(model_type="rf", params={"n_estimators": 10})
        trainer.fit(X, y)

        importance = trainer.get_feature_importance_gini()
        assert "feature" in importance.columns
        assert "gini_importance" in importance.columns

    def test_permutation_importance(self, sample_data):
        X, y = sample_data
        trainer = TreeModelTrainer(model_type="rf", params={"n_estimators": 10})
        trainer.fit(X, y)

        importance = trainer.get_permutation_importance(X, y, n_repeats=3)
        assert "feature" in importance.columns
        assert "perm_importance_mean" in importance.columns


class TestEvaluation:
    def test_compute_binary_metrics(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7])

        metrics = compute_binary_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "false_alarm_rate" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_compute_multiclass_metrics(self):
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 1])
        y_proba = np.random.rand(8, 5)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]

        metrics = compute_multiclass_metrics(y_true, y_pred, y_proba, labels)

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "per_class" in metrics
        assert len(metrics["per_class"]) == 5

    def test_mcnemar_test(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred2 = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])

        result = mcnemar_test(y_true, y_pred1, y_pred2)

        assert "chi2" in result
        assert "p_value" in result
        assert "significant" in result
        assert 0 <= result["p_value"] <= 1

    def test_stratified_cv_evaluation(self, sample_data):
        X, y = sample_data
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(random_state=42, max_iter=500)
        results = stratified_cv_evaluation(estimator, X, y, cv=3, task_type="binary")

        assert "accuracy" in results
        assert "cv_folds" in results
        assert results["cv_folds"] == 3


class TestConfig:
    def test_settings_loaded(self):
        assert settings.random_state == 42
        assert settings.train_split == 0.7
        assert len(settings.categorical_features) == 3
        assert len(settings.numerical_features) > 30
        assert "Normal" in settings.attack_categories
        assert "DoS" in settings.attack_categories

    def test_label_mappings(self):
        from src.config import map_label_to_binary, map_label_to_multiclass, get_category_from_label

        assert map_label_to_binary("normal") == 0
        assert map_label_to_binary("back") == 1
        assert map_label_to_multiclass("normal") == 0
        assert map_label_to_multiclass("back") == 1
        assert get_category_from_label("back") == "DoS"
        assert get_category_from_label("satan") == "Probe"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])