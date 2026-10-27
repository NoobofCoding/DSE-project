from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from statsmodels.stats.multitest import multipletests

from src.config import FIGURES_DIR, METRICS_DIR
from src.data_loader import extract_features_targets
from src.feature_selection import remove_constant_features, select_top_features
from src.modeling import build_model_dict
from src.preprocessing import prepare_feature_frame


def compute_statistical_feature_ranking(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Rank training-set features with a robust nonparametric comparison and BH adjustment."""
    X = remove_constant_features(X.copy())
    records: list[dict] = []
    valid_mask = y.notna()
    X_valid = X.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].astype(int)
    for feature in X_valid.columns:
        pass_values = X_valid.loc[y_valid == 0, feature].dropna()
        fail_values = X_valid.loc[y_valid == 1, feature].dropna()
        missing_count = int(X_valid[feature].isna().sum())
        missing_fraction = float(missing_count / len(X_valid))
        pass_count = int(pass_values.shape[0])
        fail_count = int(fail_values.shape[0])
        if pass_count < 2 or fail_count < 2:
            records.append({
                "feature": feature,
                "missing_count": missing_count,
                "missing_fraction": missing_fraction,
                "pass_sample_count": pass_count,
                "fail_sample_count": fail_count,
                "effect_size": np.nan,
                "p_value": np.nan,
                "adjusted_p_value": 1.0,
                "significant_after_fdr": False,
            })
            continue
        try:
            statistic, p_value = mannwhitneyu(pass_values, fail_values, alternative="two-sided", method="auto")
        except Exception:
            statistic = np.nan
            p_value = np.nan
        if np.isnan(p_value):
            effect_size = np.nan
        else:
            n1 = pass_count
            n2 = fail_count
            effect_size = float((2.0 * statistic / (n1 * n2)) - 1.0)
        records.append({
            "feature": feature,
            "missing_count": missing_count,
            "missing_fraction": missing_fraction,
            "pass_sample_count": pass_count,
            "fail_sample_count": fail_count,
            "effect_size": float(effect_size) if np.isfinite(effect_size) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            "adjusted_p_value": 1.0,
            "significant_after_fdr": False,
        })

    ranking = pd.DataFrame(records)
    valid_p = ranking[ranking["p_value"].notna()].copy()
    if not valid_p.empty:
        adjusted = np.clip(multipletests(valid_p["p_value"].to_numpy(), method="fdr_bh")[1], 0.0, 1.0)
        ranking.loc[ranking["p_value"].notna(), "adjusted_p_value"] = adjusted
        ranking.loc[ranking["p_value"].notna(), "significant_after_fdr"] = adjusted < alpha
    return ranking.sort_values(["adjusted_p_value", "p_value"], ascending=[True, True], na_position="last").reset_index(drop=True)


def compute_feature_selection_stability(X: pd.DataFrame, y: pd.Series, top_k: int = 40, cv_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Estimate how often each feature is selected across stratified CV folds."""
    y = pd.Series(y).reset_index(drop=True)
    X = remove_constant_features(X.copy()).reset_index(drop=True)
    class_min = int(y.value_counts().min())
    n_splits = min(cv_folds, class_min)
    if n_splits < 2:
        raise ValueError("Feature selection stability requires at least two samples in each class for cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    counts = {feature: 0 for feature in X.columns}
    for train_idx, _ in cv.split(X, y):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_train_imputed = remove_constant_features(X_train_imputed)
        if X_train_imputed.empty:
            continue
        k = min(top_k, X_train_imputed.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train_imputed, y_train)
        selected = X_train_imputed.columns[selector.get_support()].tolist()
        for col in selected:
            counts[col] += 1
    stability = pd.DataFrame({
        "feature": list(counts.keys()),
        "selection_frequency": [counts[col] / n_splits for col in counts],
        "selection_count": [counts[col] for col in counts],
    })
    return stability.sort_values(["selection_frequency", "feature"], ascending=[False, True]).reset_index(drop=True)


def compute_model_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "balanced_accuracy",
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on a held-out validation set, not on the training data."""
    importance = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    result = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    })
    return result.sort_values(["importance_mean", "feature"], ascending=[False, True]).reset_index(drop=True)


def generate_sensor_feature_analysis(
    df: pd.DataFrame,
    model_name: str = "logistic_regression",
    top_k: int = 40,
    output_dir: str | Path = METRICS_DIR,
    figure_dir: str | Path = FIGURES_DIR,
    random_state: int = 42,
) -> dict:
    """Generate a leakage-safe, audit-friendly sensor-feature analysis."""
    X, y = extract_features_targets(df)
    X_clean = remove_constant_features(prepare_feature_frame(df))
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_clean,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    if X_train.shape[1] == 0:
        raise ValueError("Feature analysis requires at least one valid sensor feature after preprocessing.")

    train_selected_X, selected_features = select_top_features(X_train, y_train, k=min(top_k, X_train.shape[1]))
    valid_selected_X = X_valid[selected_features].copy()

    model = build_model_dict()[model_name]
    model.fit(train_selected_X, y_train)
    importance_df = compute_model_feature_importance(model, valid_selected_X, y_valid)
    stability_df = compute_feature_selection_stability(X_train, y_train, top_k=min(top_k, X_train.shape[1]), cv_folds=5, random_state=random_state)
    statistical_df = compute_statistical_feature_ranking(X_train, y_train, alpha=0.05)

    combined = statistical_df.merge(importance_df, on="feature", how="outer").merge(stability_df, on="feature", how="outer")
    combined = combined.sort_values(["selection_frequency", "importance_mean", "adjusted_p_value"], ascending=[False, False, True], na_position="last").reset_index(drop=True)

    metrics_dir = Path(output_dir)
    figures_dir = Path(figure_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    statistical_path = metrics_dir / "statistical_feature_analysis.csv"
    importance_path = metrics_dir / "permutation_importance.csv"
    stability_path = metrics_dir / "feature_selection_stability.csv"
    sensor_path = metrics_dir / "sensor_feature_analysis.csv"

    statistical_df.to_csv(statistical_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    combined.to_csv(sensor_path, index=False)

    save_top_feature_importance(importance_df, figures_dir / "top_sensor_importance.png")
    save_feature_selection_stability(stability_df, figures_dir / "feature_selection_stability.png")
    save_effect_size_plot(statistical_df, figures_dir / "effect_size_vs_adjusted_pvalue.png")
    save_missingness_plot(combined, figures_dir / "important_sensor_missingness.png")
    save_distribution_plot(X_train, y_train, selected_features[:4], figures_dir / "leading_sensor_distributions.png")

    summary = {
        "analysis_method": "Leakage-safe training-data statistical ranking plus validation-set permutation importance and training-fold feature-selection stability.",
        "model_name": model_name,
        "selected_features": selected_features,
        "leading_features": combined.head(10).to_dict(orient="records"),
        "paths": {
            "sensor_feature_analysis_csv": str(sensor_path),
            "statistical_feature_analysis_csv": str(statistical_path),
            "permutation_importance_csv": str(importance_path),
            "feature_selection_stability_csv": str(stability_path),
            "top_sensor_importance_png": str(figures_dir / "top_sensor_importance.png"),
            "feature_selection_stability_png": str(figures_dir / "feature_selection_stability.png"),
            "effect_size_vs_adjusted_pvalue_png": str(figures_dir / "effect_size_vs_adjusted_pvalue.png"),
            "important_sensor_missingness_png": str(figures_dir / "important_sensor_missingness.png"),
            "leading_sensor_distributions_png": str(figures_dir / "leading_sensor_distributions.png"),
        },
    }
    return summary


def save_top_feature_importance(importance_df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = importance_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1], xerr=top["importance_std"][::-1], color="steelblue", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Sensor Features by Permutation Importance")
    ax.set_xlabel("Mean balanced-accuracy decrease")
    ax.set_ylabel("Sensor")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_feature_selection_stability(stability_df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = stability_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["selection_frequency"][::-1], color="darkorange")
    ax.set_title("Feature Selection Stability Across Cross-Validation Folds")
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Sensor")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_effect_size_plot(statistical_df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = statistical_df.dropna(subset=["effect_size", "adjusted_p_value"]).copy()
    if valid.empty:
        return path
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["effect_size"], -np.log10(valid["adjusted_p_value"]), s=25, edgecolor="black", linewidth=0.4)
    for _, row in valid.head(10).iterrows():
        ax.annotate(row["feature"], (row["effect_size"], -np.log10(row["adjusted_p_value"])), fontsize=8)
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", label="FDR = 0.05")
    ax.set_title("Effect Size vs Adjusted Significance")
    ax.set_xlabel("Effect size (rank-biserial)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_missingness_plot(combined_df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = combined_df.dropna(subset=["missing_fraction"]).head(15).copy()
    if top.empty:
        return path
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["missing_fraction"][::-1], color="mediumseagreen")
    ax.set_title("Missingness of Leading Sensor Features")
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("Sensor")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_distribution_plot(X: pd.DataFrame, y: pd.Series, features: list[str], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not features or X.empty:
        return path
    relevant = features[:4]
    fig, axes = plt.subplots(len(relevant), 1, figsize=(8, 3 * len(relevant)))
    if len(relevant) == 1:
        axes = [axes]
    for ax, feature in zip(axes, relevant):
        pass_values = X.loc[y == 0, feature].dropna()
        fail_values = X.loc[y == 1, feature].dropna()
        if not pass_values.empty and not fail_values.empty:
            data = pd.DataFrame({"Pass": pass_values, "Fail": fail_values})
            data.boxplot(ax=ax)
        ax.set_title(f"Distribution of {feature}")
        ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_feature_analysis(df: pd.DataFrame, output_dir: str | Path = METRICS_DIR, figure_dir: str | Path = FIGURES_DIR) -> dict:
    summary = generate_sensor_feature_analysis(df, model_name="logistic_regression", output_dir=output_dir, figure_dir=figure_dir)
    return summary
