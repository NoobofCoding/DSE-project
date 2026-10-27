from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS_DIR, MODELS_DIR
from src.evaluation import compute_classification_metrics, threshold_metrics


def build_model_dict() -> dict[str, Pipeline]:
    return {
        "dummy_classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=42)),
        ]),
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=5,
                max_depth=6,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]),
    }


def choose_best_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, objective: str = "f1", min_precision: float = 0.0, min_recall: float = 0.0) -> tuple[float, dict]:
    thresholds = np.linspace(0.1, 0.9, 81)
    selected = None
    for threshold in thresholds:
        metrics = threshold_metrics(y_true, probabilities, float(threshold))
        if metrics["precision"] < min_precision or metrics["recall"] < min_recall:
            continue
        candidate = {
            "threshold": float(threshold),
            **metrics,
            "objective_value": metrics.get(objective, metrics["balanced_accuracy"]),
        }
        if selected is None or candidate["objective_value"] > selected["objective_value"]:
            selected = candidate
        elif (
            np.isclose(candidate["objective_value"], selected["objective_value"])
            and candidate["confusion_matrix"]["fn"] < selected["confusion_matrix"]["fn"]
        ):
            selected = candidate
    if selected is None:
        fallback_metrics = threshold_metrics(y_true, probabilities, 0.5)
        selected = {
            "threshold": 0.5,
            **fallback_metrics,
            "objective_value": fallback_metrics.get(objective, fallback_metrics["balanced_accuracy"]),
        }
    return float(selected["threshold"]), selected


def create_train_validation_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def evaluate_threshold_grid(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray | None = None, objective: str = "f1") -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    records = []
    for threshold in thresholds:
        metrics = threshold_metrics(y_true, probabilities, float(threshold))
        records.append({
            "threshold": float(threshold),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "false_positive_count": int(metrics["confusion_matrix"]["fp"]),
            "false_negative_count": int(metrics["confusion_matrix"]["fn"]),
            "flagged_fraction": float(metrics["flagged_fraction"]),
            "objective_value": float(metrics.get(objective, metrics["balanced_accuracy"])),
        })
    return pd.DataFrame(records)


def cross_validate_models(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5, random_state: int = 42, objective: str = "f1") -> dict:
    cv = StratifiedKFold(n_splits=min(cv_folds, int((y == 1).sum())), shuffle=True, random_state=random_state)
    if cv.n_splits < 2:
        raise ValueError("Cross-validation requires at least two positive examples in the minority class for a valid fold split.")

    model_results = []
    for model_name, estimator in build_model_dict().items():
        fold_metrics = []
        oof_probabilities = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in cv.split(X, y):
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            val_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
            oof_probabilities[val_idx] = val_prob
            metrics = threshold_metrics(y.iloc[val_idx], val_prob, 0.5)
            metrics["fold"] = int(len(fold_metrics) + 1)
            fold_metrics.append(metrics)

        threshold_df = evaluate_threshold_grid(y, oof_probabilities, objective=objective)
        selected_threshold, selected_summary = choose_best_threshold(y, oof_probabilities, objective=objective)
        selected_metrics = threshold_metrics(y, oof_probabilities, selected_threshold)

        metric_names = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
        summary_by_metric = {}
        for metric_name in metric_names:
            values = [float(entry[metric_name]) for entry in fold_metrics]
            summary_by_metric[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }

        model_results.append({
            "model": model_name,
            "cv_folds": int(len(fold_metrics)),
            "selected_threshold": float(selected_threshold),
            "selected_threshold_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in selected_metrics.items() if k not in {"confusion_matrix"}},
            "default_threshold_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in threshold_metrics(y, oof_probabilities, 0.5).items() if k not in {"confusion_matrix"}},
            "mean_std_metrics": summary_by_metric,
            "threshold_analysis": threshold_df.to_dict(orient="records"),
            "minority_class_support": int(np.sum(y == 1)),
        })

    best_model = sorted(model_results, key=lambda row: (row["mean_std_metrics"]["balanced_accuracy"]["mean"], row["mean_std_metrics"]["f1"]["mean"]), reverse=True)[0]
    return {"models": model_results, "best_model": best_model["model"], "selection_objective": objective, "cv_splits": int(cv.n_splits)}


def compute_feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    scores = importance.importances_mean
    feature_names = list(X.columns)
    result = pd.DataFrame({"feature": feature_names, "importance_mean": scores, "importance_std": importance.importances_std})
    return result.sort_values(["importance_mean", "feature"], ascending=[False, True]).reset_index(drop=True)


def save_threshold_analysis(results: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path


def train_and_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path = MODELS_DIR,
    predefined_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if predefined_split is None:
        X_train, X_test, y_train, y_test = create_train_validation_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = predefined_split

    cv_summary = cross_validate_models(X_train, y_train, cv_folds=5, random_state=42, objective="f1")
    best_model_name = cv_summary["best_model"]
    best_cv_model = next(item for item in cv_summary["models"] if item["model"] == best_model_name)
    selected_threshold = float(best_cv_model["selected_threshold"])

    selected_pipeline = clone(build_model_dict()[best_model_name])
    selected_pipeline.fit(X_train, y_train)
    test_prob = selected_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = threshold_metrics(y_test, test_prob, selected_threshold)
    final_model_path = output_dir / f"{best_model_name}.joblib"
    joblib.dump(selected_pipeline, final_model_path)

    feature_importance = compute_feature_importance(selected_pipeline, X_test, y_test)
    feature_importance_path = output_dir / f"{best_model_name}_feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    feature_importance_metrics_path = METRICS_DIR / f"{best_model_name}_feature_importance.csv"
    feature_importance.to_csv(feature_importance_metrics_path, index=False)

    threshold_analysis = evaluate_threshold_grid(y_train, selected_pipeline.predict_proba(X_train)[:, 1], objective="f1")
    threshold_path = output_dir / f"{best_model_name}_threshold_analysis.csv"
    threshold_analysis.to_csv(threshold_path, index=False)
    threshold_metrics_path = METRICS_DIR / f"{best_model_name}_threshold_analysis.csv"
    threshold_analysis.to_csv(threshold_metrics_path, index=False)

    summary = {
        "methodology": {
            "evaluation_strategy": "stratified train/test split followed by stratified 5-fold cv on the training data",
            "random_state": 42,
            "test_size": 0.2,
            "cv_folds": 5,
            "threshold_selection_objective": "f1",
            "leakage_note": "Threshold tuning used training-only out-of-fold predictions; the held-out test set was not used for model or threshold selection.",
        },
        "cv_summary": cv_summary,
        "selected_model": best_model_name,
        "selected_threshold": selected_threshold,
        "final_test_metrics": final_metrics,
        "feature_importance": feature_importance.head(20).to_dict(orient="records"),
        "paths": {
            "model_path": str(final_model_path),
            "feature_importance_path": str(feature_importance_path),
            "threshold_analysis_path": str(threshold_path),
        },
    }

    summary_path = output_dir / "model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
