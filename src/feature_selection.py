from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from src.preprocessing import impute_features


def remove_constant_features(X: pd.DataFrame, tolerance: float = 1e-8) -> pd.DataFrame:
    constant_mask = X.nunique(dropna=True) > 1
    selected = X.loc[:, constant_mask].copy()
    if tolerance > 0:
        std_mask = selected.std(axis=0, ddof=0) > tolerance
        selected = selected.loc[:, std_mask].copy()
    return selected


def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 40) -> tuple[pd.DataFrame, list[str]]:
    X_imputed = impute_features(X)
    X_imputed = remove_constant_features(X_imputed)
    if X_imputed.shape[1] < k:
        k = X_imputed.shape[1]
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_imputed, y)
    feature_names = X_imputed.columns[selector.get_support()].tolist()
    return pd.DataFrame(X_selected, columns=feature_names, index=X_imputed.index), feature_names


def summarize_feature_quality(X: pd.DataFrame) -> dict:
    stds = X.std(ddof=0, skipna=True)
    var_threshold = stds > 0
    return {
        "total_features": int(X.shape[1]),
        "constant_features": int((~var_threshold).sum()),
        "near_constant_features": int((stds <= 1e-3).sum()),
        "remaining_features": int(var_threshold.sum()),
    }
