import pandas as pd

from src.data_loader import extract_features_targets
from src.preprocessing import build_preprocessing_pipeline, prepare_feature_frame


def test_preprocessing_pipeline_fits_and_transforms():
    df = pd.DataFrame({
        'Timestamp': ['2024-01-01', '2024-01-02'],
        'Sensor_1': [1.0, 2.0],
        'Sensor_2': [10.0, 20.0],
        'Pass_Fail': ['Pass', 'Fail'],
    })
    X, y = extract_features_targets(df)
    X_df = prepare_feature_frame(df)
    pipeline = build_preprocessing_pipeline(list(X_df.columns))
    transformed = pipeline.fit_transform(X_df)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] == X_df.shape[1]
    assert y.tolist() == [0, 1]
