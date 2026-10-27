import pandas as pd
import pytest

from src.data_loader import extract_features_targets, load_dataset, validate_dataset


def test_load_valid_csv():
    df = load_dataset('data/raw/secom_combined.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Pass_Fail' in df.columns


def test_missing_required_target_column():
    df = pd.DataFrame({'Timestamp': ['2020-01-01'], 'Sensor_1': [1.2]})
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_non_numeric_sensor_values():
    df = pd.DataFrame({'Timestamp': ['2020-01-01'], 'Sensor_1': ['bad'], 'Pass_Fail': ['Pass']})
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_missing_value_handling():
    df = pd.DataFrame({'Timestamp': ['2020-01-01'], 'Sensor_1': [float('nan')], 'Pass_Fail': ['Pass']})
    X, y = extract_features_targets(df)
    assert X.shape[1] == 1
    assert y.iloc[0] == 0
