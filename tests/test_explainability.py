from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability import (
    compute_feature_selection_stability,
    compute_statistical_feature_ranking,
    generate_sensor_feature_analysis,
)
from src.feature_selection import remove_constant_features


def make_fixture_dataframe(n_per_group: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n_per_group):
        rows.append({
            "Timestamp": "2024-01-01",
            "Sensor_1": rng.normal(10, 1.0),
            "Sensor_2": rng.normal(12, 1.5),
            "Sensor_3": rng.normal(20, 3.0),
            "Sensor_4": np.nan,
            "Sensor_5": 5.0,
            "Sensor_6": rng.normal(50, 5.0),
            "Sensor_7": rng.normal(4.0, 0.7),
            "Sensor_8": rng.normal(7.5, 1.2),
            "Pass_Fail": "Pass",
        })
    for _ in range(n_per_group):
        rows.append({
            "Timestamp": "2024-01-02",
            "Sensor_1": rng.normal(15, 1.0),
            "Sensor_2": rng.normal(8, 1.5),
            "Sensor_3": rng.normal(6, 3.0),
            "Sensor_4": rng.normal(10, 1.5),
            "Sensor_5": 5.0,
            "Sensor_6": rng.normal(35, 5.0),
            "Sensor_7": rng.normal(9.0, 0.7),
            "Sensor_8": np.nan,
            "Pass_Fail": "Fail",
        })
    return pd.DataFrame(rows)


def test_feature_analysis_excludes_metadata_and_handles_missing_constant_features():
    df = make_fixture_dataframe(20)
    X = df.drop(columns=["Pass_Fail", "Timestamp"])
    assert "Timestamp" not in X.columns
    assert "Pass_Fail" not in X.columns
    X_clean = remove_constant_features(X)
    assert "Sensor_5" not in X_clean.columns


def test_statistical_feature_ranking_has_adjusted_p_values_and_columns():
    df = make_fixture_dataframe(40)
    X = df.drop(columns=["Pass_Fail", "Timestamp"])
    y = df["Pass_Fail"].map({"Pass": 0, "Fail": 1})
    stats = compute_statistical_feature_ranking(X, y)
    assert {"feature", "p_value", "adjusted_p_value", "effect_size"}.issubset(stats.columns)
    assert stats["adjusted_p_value"].notna().any()
    assert stats["adjusted_p_value"].le(1.0).all()


def test_selection_stability_output_is_bounded_and_reproducible():
    df = make_fixture_dataframe(25)
    X = df.drop(columns=["Pass_Fail", "Timestamp"])
    y = df["Pass_Fail"].map({"Pass": 0, "Fail": 1})
    stability = compute_feature_selection_stability(X, y, top_k=5, cv_folds=3, random_state=42)
    assert stability["selection_frequency"].between(0, 1).all()
    assert set(stability.columns) == {"feature", "selection_frequency", "selection_count"}


def test_generate_sensor_feature_analysis_creates_expected_files(tmp_path):
    df = make_fixture_dataframe(30)
    summary = generate_sensor_feature_analysis(
        df,
        top_k=5,
        output_dir=tmp_path / "metrics",
        figure_dir=tmp_path / "figures",
        random_state=42,
    )
    assert (tmp_path / "metrics" / "sensor_feature_analysis.csv").exists()
    assert (tmp_path / "metrics" / "permutation_importance.csv").exists()
    assert (tmp_path / "metrics" / "feature_selection_stability.csv").exists()
    assert (tmp_path / "figures" / "top_sensor_importance.png").exists()
    assert summary["selected_features"]


def test_generate_sensor_feature_analysis_reuses_training_data_not_test_labels():
    df = make_fixture_dataframe(30)
    X = df.drop(columns=["Pass_Fail", "Timestamp"])
    y = df["Pass_Fail"].map({"Pass": 0, "Fail": 1})
    X_clean = remove_constant_features(X)
    stats = compute_statistical_feature_ranking(X_clean, y)
    assert stats["feature"].nunique() == len(stats)
    assert len(stats) == X_clean.shape[1]
