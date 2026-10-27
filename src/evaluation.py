from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


def threshold_metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, predictions, probabilities)
    metrics["threshold"] = float(threshold)
    metrics["flagged_fraction"] = float(np.mean(predictions))
    return metrics


def save_metric_table(metrics: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(path, index=False)
    return path


def load_metric_summary(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data
