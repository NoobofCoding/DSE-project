from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from src.config import (
    LEGACY_DATASET_CSV,
    LEGACY_DATASET_XLSX,
    PRIMARY_DATASET_CSV,
    PRIMARY_DATASET_XLSX,
    TARGET_COL,
    TIMESTAMP_COL,
)


def locate_dataset(path: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([
        PRIMARY_DATASET_CSV,
        PRIMARY_DATASET_XLSX,
        LEGACY_DATASET_CSV,
        LEGACY_DATASET_XLSX,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_dataset(path: str | Path | None = None, file_type: str | None = None) -> pd.DataFrame:
    resolved = locate_dataset(path)
    if resolved is None:
        raise FileNotFoundError("No dataset file was found in the project or workspace root.")
    if file_type is None:
        file_type = resolved.suffix.lower().lstrip('.')
    if file_type.lower() == 'csv':
        return pd.read_csv(resolved)
    if file_type.lower() in {'xlsx', 'xls'}:
        return pd.read_excel(resolved)
    raise ValueError(f"Unsupported dataset type: {file_type}")


def get_feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL, timestamp_col: str = TIMESTAMP_COL) -> list[str]:
    excluded = {target_col, timestamp_col}
    return [col for col in df.columns if col not in excluded]


def validate_dataset(df: pd.DataFrame, target_col: str = TARGET_COL, timestamp_col: str = TIMESTAMP_COL) -> list[str]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the dataset.")
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' is missing from the dataset.")
    feature_cols = get_feature_columns(df, target_col=target_col, timestamp_col=timestamp_col)
    if not feature_cols:
        raise ValueError("No sensor columns were detected after excluding target and timestamp.")
    for col in feature_cols:
        values = df[col]
        converted = pd.to_numeric(values, errors='coerce')
        invalid = values.notna() & converted.isna()
        if invalid.any():
            invalid_examples = int(invalid.sum())
            raise ValueError(f"Column '{col}' contains non-numeric values ({invalid_examples} invalid entries).")
    return feature_cols


def extract_features_targets(df: pd.DataFrame, target_col: str = TARGET_COL, timestamp_col: str = TIMESTAMP_COL) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = validate_dataset(df, target_col=target_col, timestamp_col=timestamp_col)
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    if y.isna().any():
        raise ValueError("Target column contains missing values; labels must be complete.")
    y = y.astype(str).str.strip()
    label_map = {'pass': 0, 'fail': 1, 'Pass': 0, 'Fail': 1, '0': 0, '1': 1, '-1': 0, '1.0': 1, '-1.0': 0}
    invalid = ~y.map(lambda label: label in label_map)
    if invalid.any():
        unexpected = sorted(y[invalid].unique().tolist())[:10]
        raise ValueError(f"Unexpected target labels: {unexpected}")
    y = y.map(label_map)
    return X, y.astype(int)


def audit_dataset(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    feature_cols = get_feature_columns(df, target_col=target_col)
    target_counts = df[target_col].value_counts(dropna=False).to_dict()
    missingness = df.isna().sum().sort_values(ascending=False)
    missing_summary = {
        "count": int(missingness.sum()),
        "by_column": {str(col): int(val) for col, val in missingness.items()},
        "top_missing_columns": {str(col): int(val) for col, val in missingness.head(10).items()},
    }
    duplicate_rows = int(df.duplicated().sum())
    total_rows, total_cols = df.shape
    imbalance_ratio = float((df[target_col].value_counts(normalize=True).max()) / (df[target_col].value_counts(normalize=True).min() if df[target_col].nunique() > 1 else 1.0))
    return {
        "shape": {"rows": total_rows, "columns": total_cols},
        "target_counts": target_counts,
        "class_imbalance_ratio": imbalance_ratio,
        "duplicate_rows": duplicate_rows,
        "feature_count": len(feature_cols),
        "missingness": missing_summary,
        "timestamp_dtype": str(df['Timestamp'].dtype) if 'Timestamp' in df.columns else None,
        "sensor_numeric_dtypes": df[feature_cols].dtypes.value_counts().to_dict(),
    }


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
