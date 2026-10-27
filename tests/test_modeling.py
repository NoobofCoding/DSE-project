import numpy as np
import pandas as pd

from src.data_loader import extract_features_targets
from src.modeling import build_model_dict, choose_best_threshold, create_train_validation_split, evaluate_threshold_grid


def test_model_training_and_prediction_shape():
    df = pd.DataFrame({
        'Timestamp': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Sensor_1': [1.0, 2.0, 3.0, 4.0],
        'Sensor_2': [10.0, 20.0, 30.0, 40.0],
        'Sensor_3': [5.0, 6.0, 7.0, 8.0],
        'Pass_Fail': ['Pass', 'Pass', 'Fail', 'Fail'],
    })
    X, y = extract_features_targets(df)
    X_train, X_test, y_train, y_test = create_train_validation_split(X, y, test_size=0.5, random_state=42)
    model = build_model_dict()['logistic_regression']
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert set(y_test.unique().tolist()).issubset({0, 1})
    assert hasattr(model.named_steps['classifier'], 'coef_')
    assert len(model.named_steps['classifier'].coef_) > 0


def test_threshold_grid_and_selection():
    y = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.25, 0.65], dtype=float)
    threshold_df = evaluate_threshold_grid(y, probabilities)
    assert 'threshold' in threshold_df.columns
    assert 'f1' in threshold_df.columns
    threshold, details = choose_best_threshold(y, probabilities, objective='f1')
    assert 0.1 <= threshold <= 0.9
    assert 'objective_value' in details
