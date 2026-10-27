from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_COL = "Pass_Fail"
TIMESTAMP_COL = "Timestamp"
PRIMARY_DATASET_CSV = RAW_DATA_DIR / "secom_combined.csv"
PRIMARY_DATASET_XLSX = RAW_DATA_DIR / "secom_combined.xlsx"
LEGACY_DATASET_CSV = PROJECT_ROOT / "secom_combined.csv"
LEGACY_DATASET_XLSX = PROJECT_ROOT / "secom_combined.xlsx"
