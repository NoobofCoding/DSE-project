from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis import create_eda_outputs
from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PRIMARY_DATASET_CSV, PROJECT_ROOT
from src.data_loader import audit_dataset, extract_features_targets, load_dataset, save_processed_data, validate_dataset
from src.evaluation import compute_classification_metrics, save_metric_table, threshold_metrics
from src.explainability import run_feature_analysis
from src.feature_selection import remove_constant_features, select_top_features, summarize_feature_quality
from src.modeling import create_train_validation_split, train_and_compare_models
from src.preprocessing import prepare_feature_frame
from src.reporting import save_project_summary


def run_audit() -> dict:
    df = load_dataset(PRIMARY_DATASET_CSV)
    validate_dataset(df)
    audit = audit_dataset(df)
    quality = {
        "dataset": {
            "rows": audit["shape"]["rows"],
            "columns": audit["shape"]["columns"],
            "target_counts": audit["target_counts"],
            "class_imbalance_ratio": audit["class_imbalance_ratio"],
            "duplicate_rows": audit["duplicate_rows"],
            "feature_count": audit["feature_count"],
            "sensor_numeric_dtypes": audit["sensor_numeric_dtypes"],
        },
        "missingness": audit["missingness"],
    }
    save_project_summary(quality, PROJECT_ROOT / "reports" / "data_quality_report.json")
    create_eda_outputs(df, FIGURES_DIR)
    print("Audit complete.")
    return quality


def run_analysis() -> dict:
    df = load_dataset(PRIMARY_DATASET_CSV)
    X, y = extract_features_targets(df)
    X_clean = prepare_feature_frame(df)
    X_nonconstant = remove_constant_features(X_clean)
    selected_X, selected_features = select_top_features(X_nonconstant, y, k=min(40, X_nonconstant.shape[1]))
    quality = summarize_feature_quality(X_nonconstant)
    report = {
        "feature_quality": quality,
        "selected_features": selected_features,
        "shape_after_cleaning": {"rows": int(X_nonconstant.shape[0]), "columns": int(X_nonconstant.shape[1])},
    }
    save_project_summary(report, PROJECT_ROOT / "reports" / "feature_analysis.json")
    print("Analysis complete.")
    return report


def run_explain() -> dict:
    df = load_dataset(PRIMARY_DATASET_CSV)
    summary = run_feature_analysis(df, output_dir=METRICS_DIR, figure_dir=FIGURES_DIR)
    save_project_summary({"sensor_feature_analysis": summary}, PROJECT_ROOT / "reports" / "sensor_feature_analysis_summary.json")
    print("Feature explanation complete.")
    return summary


def run_train() -> dict:
    df = load_dataset(PRIMARY_DATASET_CSV)
    X, y = extract_features_targets(df)
    X_clean = prepare_feature_frame(df)
    X_nonconstant = remove_constant_features(X_clean)
    X_train, X_test, y_train, y_test = create_train_validation_split(X_nonconstant, y, test_size=0.2, random_state=42)
    selected_X_train, selected_features = select_top_features(X_train, y_train, k=min(40, X_train.shape[1]))
    selected_X_test = X_test[selected_features]
    training_summary = train_and_compare_models(
        selected_X_train,
        y_train,
        predefined_split=(selected_X_train, selected_X_test, y_train, y_test),
    )
    cv_rows = []
    for row in training_summary["cv_summary"]["models"]:
        metrics = row["mean_std_metrics"]
        cv_rows.append({
            "model": row["model"],
            "selected_threshold": row["selected_threshold"],
            "balanced_accuracy_mean": metrics["balanced_accuracy"]["mean"],
            "balanced_accuracy_std": metrics["balanced_accuracy"]["std"],
            "precision_mean": metrics["precision"]["mean"],
            "precision_std": metrics["precision"]["std"],
            "recall_mean": metrics["recall"]["mean"],
            "recall_std": metrics["recall"]["std"],
            "f1_mean": metrics["f1"]["mean"],
            "f1_std": metrics["f1"]["std"],
            "roc_auc_mean": metrics["roc_auc"]["mean"],
            "roc_auc_std": metrics["roc_auc"]["std"],
            "average_precision_mean": metrics["average_precision"]["mean"],
            "average_precision_std": metrics["average_precision"]["std"],
        })
    pd.DataFrame(cv_rows).to_csv(METRICS_DIR / "cv_summary.csv", index=False)
    print("Training complete.")
    return training_summary


def run_evaluate() -> dict:
    df = load_dataset(PRIMARY_DATASET_CSV)
    X, y = extract_features_targets(df)
    X_clean = prepare_feature_frame(df)
    X_nonconstant = remove_constant_features(X_clean)
    X_train, X_test, y_train, y_test = create_train_validation_split(X_nonconstant, y, test_size=0.2, random_state=42)
    selected_X_train, selected_features = select_top_features(X_train, y_train, k=min(40, X_train.shape[1]))
    selected_X_test = X_test[selected_features]
    summary = train_and_compare_models(
        selected_X_train,
        y_train,
        predefined_split=(selected_X_train, selected_X_test, y_train, y_test),
    )
    final_metrics = summary["final_test_metrics"]
    flattened = {
        "model": summary["selected_model"],
        "selected_threshold": summary["selected_threshold"],
        "accuracy": final_metrics["accuracy"],
        "balanced_accuracy": final_metrics["balanced_accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1": final_metrics["f1"],
        "roc_auc": final_metrics["roc_auc"],
        "average_precision": final_metrics["average_precision"],
        "false_positive_count": final_metrics["confusion_matrix"]["fp"],
        "false_negative_count": final_metrics["confusion_matrix"]["fn"],
        "true_negative_count": final_metrics["confusion_matrix"]["tn"],
        "true_positive_count": final_metrics["confusion_matrix"]["tp"],
        "flagged_fraction": final_metrics["flagged_fraction"],
    }
    pd.DataFrame([flattened]).to_csv(METRICS_DIR / "model_metrics.csv", index=False)
    save_project_summary(summary, PROJECT_ROOT / "reports" / "final_model_summary.json")
    print("Evaluation complete.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the semiconductor yield analysis project.")
    parser.add_argument("--task", choices=["audit", "analyze", "explain", "train", "evaluate"], required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task == "audit":
        run_audit()
    elif args.task == "analyze":
        run_analysis()
    elif args.task == "explain":
        run_explain()
    elif args.task == "train":
        run_train()
    elif args.task == "evaluate":
        run_evaluate()


if __name__ == "__main__":
    main()
