from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, bytes)):
        return value
    return str(value)


def save_summary_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_payload = _to_jsonable(payload)
    path.write_text(json.dumps(clean_payload, indent=2), encoding='utf-8')
    return path


def save_metrics_table(results: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    return path


def save_project_summary(summary: dict, output_path: str | Path) -> Path:
    return save_summary_json(summary, output_path)
