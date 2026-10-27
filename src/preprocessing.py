from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COL, TIMESTAMP_COL


def build_preprocessing_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline


def prepare_feature_frame(df: pd.DataFrame, target_col: str = TARGET_COL, timestamp_col: str = TIMESTAMP_COL) -> pd.DataFrame:
    exclude = {target_col, timestamp_col}
    feature_columns = [col for col in df.columns if col not in exclude]
    if not feature_columns:
        raise ValueError("No feature columns available after excluding the target and timestamp columns.")
    X = df[feature_columns].copy()
    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors='coerce')
    return X


def impute_features(X: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(X)
    return pd.DataFrame(values, columns=X.columns, index=X.index)
