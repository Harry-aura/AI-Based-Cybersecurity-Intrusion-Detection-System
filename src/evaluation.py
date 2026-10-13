import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "false_alarm_rate": 0.0,
        "specificity": 0.0
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["false_alarm_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="weighted")
        except ValueError:
            metrics["roc_auc_ovr"] = 0.0
            metrics["roc_auc_ovo"] = 0.0

    per_class = {}
    if labels:
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
        for label in labels:
            if label in report:
                per_class[label] = {
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"]
                }

    metrics["per_class"] = per_class
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def stratified_cv_evaluation(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
    task_type: str = "binary"
) -> Dict[str, Any]:
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    y_pred_cv = cross_val_predict(estimator, X, y, cv=cv_strategy, method="predict")
    y_proba_cv = cross_val_predict(estimator, X, y, cv=cv_strategy, method="predict_proba")

    if task_type == "binary":
        if y_proba_cv.ndim > 1:
            y_proba_cv = y_proba_cv[:, 1]
        metrics = compute_binary_metrics(y.values, y_pred_cv, y_proba_cv)
    else:
        metrics = compute_multiclass_metrics(y.values, y_pred_cv, y_proba_cv)

    metrics["cv_folds"] = cv
    return metrics


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray
) -> Dict[str, float]:
    correct1 = (y_pred_model1 == y_true)
    correct2 = (y_pred_model2 == y_true)

    n00 = np.sum((~correct1) & (~correct2))
    n01 = np.sum((~correct1) & (correct2))
    n10 = np.sum((correct1) & (~correct2))
    n11 = np.sum((correct1) & (correct2))

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    p_value = 1 - stats.chi2.cdf(chi2, df=1) if chi2 > 0 else 1.0

    return {
        "chi2": chi2,
        "p_value": p_value,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "significant": p_value < 0.05
    }


def compare_models_mcnemar(
    model1_name: str,
    model2_name: str,
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Dict[str, Any]:
    result = mcnemar_test(y_true, y_pred1, y_pred2)
    result["model1"] = model1_name
    result["model2"] = model2_name
    result["conclusion"] = (
        f"{model1_name} significantly different from {model2_name}"
        if result["significant"]
        else f"No significant difference between {model1_name} and {model2_name}"
    )
    return result


def evaluate_model_on_splits(
    model,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    task_type: str = "binary",
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    results = {}

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        if task_type == "binary":
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
            results[split_name] = compute_binary_metrics(y.values, y_pred, y_proba)
        else:
            results[split_name] = compute_multiclass_metrics(y.values, y_pred, y_proba, labels)

        results[split_name]["confusion_matrix"] = compute_confusion_matrix(y.values, y_pred).tolist()

    return results


def get_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    indices = np.linspace(0, len(fpr) - 1, min(n_points, len(fpr)), dtype=int)

    return {
        "fpr": fpr[indices],
        "tpr": tpr[indices],
        "thresholds": thresholds[indices]
    }


def get_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    indices = np.linspace(0, len(precision) - 1, min(n_points, len(precision)), dtype=int)

    return {
        "precision": precision[indices],
        "recall": recall[indices],
        "thresholds": thresholds[indices] if len(thresholds) > 0 else np.array([])
    }


def compute_feature_importance_ranking(
    feature_names: List[str],
    importances: np.ndarray,
    top_k: int = 20
) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_k)
    return df


def summarize_evaluation_results(
    results: Dict[str, Dict[str, Any]],
    task_type: str = "binary"
) -> pd.DataFrame:
    summary_rows = []

    for model_name, model_results in results.items():
        row = {"model": model_name}
        for split_name, split_metrics in model_results.items():
            if isinstance(split_metrics, dict):
                for metric_name, value in split_metrics.items():
                    if metric_name not in ("confusion_matrix", "per_class", "cv_folds"):
                        row[f"{split_name}_{metric_name}"] = value
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)