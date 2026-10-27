# Semiconductor Manufacturing Yield Intelligence and Process Fault Analysis

## Abstract

This project evaluates whether anonymous sensor measurements from the SECOM manufacturing dataset contain useful signals for distinguishing Pass and Fail outcomes. The workflow combines dataset auditing, leakage-safe preprocessing, class-imbalance-aware evaluation, feature analysis, threshold trade-off analysis, and dashboard reporting. It is intentionally framed as a decision-support study rather than a validated factory-control system.

## Problem statement and objectives

The central question is whether sensor measurements contain enough discriminative signal to identify anomalous production outcomes while remaining honest about uncertainty and class imbalance. The project addresses this by validating the raw dataset, preserving the original data, checking leakage risks, comparing suitable classical models, and ranking sensors with explainability methods that do not assume causal effects.

## Dataset and validation protocol

The dataset is the local SECOM combined export preserved in `data/raw/`. It contains `1567` observations and `592` columns: `590` sensor features, one `Timestamp` column, and one binary `Pass_Fail` label. The audited target distribution is `1463` Pass and `104` Fail, with `41951` missing sensor values and zero duplicate rows. All validation steps were performed on the actual files before any model fitting or report generation.

The evaluation protocol is leakage-safe by design: the final held-out test set is kept untouched for final assessment, while cross-validation and threshold tuning are performed on the training data only. Excluded metadata and targets are never used as predictors, and all preprocessing and feature-selection steps are fit within the training-data workflow.

## Data preprocessing and missingness

The project removes `Timestamp` and `Pass_Fail` from the feature matrix and converts each sensor to numeric values. Missing values are handled with median imputation inside the pipeline. Constant or near-constant features are excluded after checking the observed variance, and rows are not silently dropped because the raw sample is the available observational dataset. This preserves the original manufacturing record count and keeps the workflow faithful to the source data.

## Exploratory and statistical analysis

The EDA includes class-distribution, missingness, and correlation plots. The class balance is strongly skewed: the minority class is only about 6.6% of observations, which naturally limits confidence in any threshold or feature ranking that aims to recover Fail cases. The statistical analysis uses the Mann-Whitney U test for sensor-by-sensor group comparisons, followed by Benjamini-Hochberg FDR adjustment across the tested features. This allows the project to separate nominal significance from predictive usefulness without presenting every p-value as a discovery.

## Explainable sensor analysis

The explainability extension uses three complementary views:

1. Statistical ranking on training data: each sensor is compared between Pass and Fail groups using a robust rank-based test; effect sizes and adjusted p-values are recorded.
2. Model-based importance: permutation importance is computed on a held-out validation subset, using balanced accuracy as the optimization target. This estimates how much model performance changes when each sensor is shuffled while keeping the rest of the data fixed.
3. Feature-selection stability: stratified training folds are used to measure how often a sensor is selected across repeated feature-ranking procedures.

Results from the current generated analysis show the leading validation permutation-importance features as `Sensor_60`, `Sensor_122`, `Sensor_349`, `Sensor_65`, and `Sensor_206`, according to `reports/metrics/sensor_feature_analysis.csv`. Statistical ranking and fold stability provide complementary evidence; not all sensors with statistical differences have the largest out-of-sample importance. This disagreement is not a contradiction: statistical association and predictive utility are different questions, and correlated sensors can distribute or mask each other’s apparent importance.

The project therefore reports the results separately rather than collapsing them into a single arbitrary score. Anonymous sensor IDs are never interpreted as causal or physical variables without the original domain documentation.

## Model training and validation

The project compares CPU-friendly classical models: dummy baseline, logistic regression, decision tree, random forest, and gradient boosting. Cross-validation is performed on the training portion only, with a fixed random seed, stratified folds, and metrics reported as mean and standard deviation across folds. Threshold tuning is also performed on validation-level probabilities before the final test-set check, so the held-out test labels are not used to select the operating threshold.

## Evaluation metrics and threshold choice

The final metric table reports accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, and average precision. Since the Fail class is rare, the project emphasizes balanced accuracy and recall alongside confusion-matrix counts, because a very high accuracy score can still be dominated by the majority class. The selected threshold is reported separately from the default threshold and the final held-out test performance.

The current executed evaluation saved in `reports/metrics/model_metrics.csv` reports the logistic-regression model as the selected model under the cross-validation comparison, with a validation-selected threshold of `0.78` and the following final held-out test metrics:

- Accuracy: 0.8854
- Balanced accuracy: 0.5849
- Precision: 0.2000
- Recall: 0.2381
- F1: 0.2174
- ROC-AUC: 0.7265
- Average precision: 0.1760
- True negatives: 273
- False positives: 20
- False negatives: 16
- True positives: 5

These figures should be interpreted as decision-support estimates, not as production-ready defect detection guarantees.

## Dashboard and reproducibility

The dashboard loads actual saved outputs from the reports directory and displays the dataset overview, cross-validation summary, final metrics, explainable sensor table, and threshold trade-off panel. The launcher remains the primary user entry point via `start.py`, and the full pipeline is reproducible from the project root within the existing `.venv`.

## Limitations

- observational dataset without causal interpretation
- severe class imbalance and limited minority-class count in the test set
- many missing sensor values and sparse coverage across features
- anonymous sensors without verified physical meanings
- no live factory validation or prospective deployment evidence
- sensitivity to threshold choice and operational cost of false alarms

## Conclusion

The project demonstrates a transparent, leakage-aware machine-learning workflow for semiconductor-yield analysis. It preserves the original data, reports actual results rather than fabricated estimates, and explains where sensor associations are suggestive but not causal. The final evaluation supports the use of the implemented workflow for exploratory decision support while remaining appropriately conservative about the limits of the evidence.

## References

1. SECOM dataset documentation in `secom/secom.names`.
2. UCI Machine Learning Repository, semiconductor manufacturing dataset provenance.
3. Scikit-learn documentation for pipelines, cross-validation, permutation importance, and evaluation metrics.
4. Benjamini, Y., and Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.
