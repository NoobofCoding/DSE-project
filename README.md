# Semiconductor Manufacturing Yield Intelligence and Process Fault Analysis

This project investigates whether anonymous SECOM sensor measurements can support decisions about manufacturing yield and defect risk. The workflow is designed as a transparent, CPU-friendly data-science study: dataset auditing, leakage-safe model validation, threshold analysis, explainable sensor ranking, and dashboard reporting.

## Single entry point

Use the launcher from the project root:

```powershell
.\.venv\Scripts\python.exe start.py
```

If the environment is activated, the same entry point also works:

```powershell
python start.py
```

The menu includes:

1. Run full pipeline
2. Audit dataset
3. Run exploratory and statistical analysis
4. Explain sensor features
5. Train models
6. Evaluate models
7. Launch dashboard
8. Run tests
9. Exit

## Dataset source and validation

The authoritative raw data are preserved under `data/raw/` and the original source material remains in `secom/`. The combined dataset was verified to contain:

- 1567 rows
- 592 total columns
- 590 sensor features + `Timestamp` + `Pass_Fail`
- 1463 Pass observations and 104 Fail observations
- 41,951 missing sensor values
- zero duplicate rows

The `Timestamp` column is excluded from model features, and the target label `Pass_Fail` is never used as a predictor.

## Installation

```powershell
cd "C:\Users\bg057\OneDrive\Documents\DSE-project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dependencies

The project uses pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, statsmodels, plotly, joblib, streamlit, and pytest.

## Outputs

Generated project artifacts are saved under:

- `reports/data_quality_report.json`
- `reports/feature_analysis.json`
- `reports/sensor_feature_analysis_summary.json`
- `reports/metrics/model_metrics.csv`
- `reports/metrics/cv_summary.csv`
- `reports/metrics/sensor_feature_analysis.csv`
- `reports/metrics/statistical_feature_analysis.csv`
- `reports/metrics/permutation_importance.csv`
- `reports/metrics/feature_selection_stability.csv`
- `reports/figures/`
- `models/*.joblib`

## Common workflows

```powershell
python run_project.py --task audit
python run_project.py --task explain
python run_project.py --task train
python run_project.py --task evaluate
```

## Explainable sensor analysis workflow

The feature-analysis module applies a leakage-safe protocol:

- split the data into training and test subsets before fitting feature-ranking and preprocessing steps
- compare Pass and Fail distributions using Mann-Whitney U tests
- adjust p-values with the Benjamini-Hochberg FDR procedure
- compute permutation importance on a validation subset or held-out data
- assess feature-selection stability across stratified training folds
- keep the final test set untouched for the final evaluation

All results are reported as sensor-level evidence, not as causal claims. Anonymous sensor IDs remain labels only (`Sensor_60`, etc.), with no physical meaning assigned.

## Dashboard

The visualization-first dashboard reads saved reports and model artifacts rather than retraining models when controls change. It includes:

- dataset composition, class proportions, and actual Pass/Fail counts
- missingness distribution, defined availability bands, and integrity checks
- an interactive sensor distribution and Pass-versus-Fail explorer
- permutation importance, selection stability, adjusted statistical evidence, and sensor missingness
- cross-validation model comparisons with fold variability
- held-out test confusion matrix and ROC/precision-recall curves when saved-model inference reproduces the saved test counts
- validation-only threshold exploration based on out-of-fold results

The test-set curves are shown only when predictions from the saved model on the fixed split reproduce the recorded confusion matrix. The dashboard does not use test labels to tune thresholds. Use the launcher for normal operation:

```powershell
.\.venv\Scripts\python.exe start.py
```

For direct local development, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.headless true --server.address 127.0.0.1
```

## Known limitations

This is an observational manufacturing dataset with strong class imbalance, many missing measurements, and no live production validation. High performance in a single report is not proof of causal effect or operational reliability. The workflow is intended for decision support and research, not automated factory control.

## Academic integrity

The project explicitly distinguishes between association, predictive utility, and causation. No fabricated metrics or unsupported claims are reported in the generated outputs.
