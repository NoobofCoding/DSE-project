from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_project import run_analysis, run_audit, run_evaluate, run_explain, run_train


def print_banner() -> None:
    print("\nSEMICONDUCTOR YIELD INTELLIGENCE")
    print("=" * 35)


def ensure_dependencies() -> None:
    missing = []
    for module in ["pandas", "numpy", "sklearn", "streamlit", "matplotlib", "seaborn", "pytest"]:
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing required dependencies: " + ", ".join(missing) + ". "
            "Install them with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )


def run_full_pipeline() -> None:
    print("[1/5] Running dataset audit...")
    run_audit()
    print("[2/5] Running exploratory and statistical analysis...")
    run_analysis()
    print("[3/5] Generating explainable sensor analysis...")
    run_explain()
    print("[4/5] Training models...")
    run_train()
    print("[5/5] Evaluating models...")
    run_evaluate()

    required_paths = [
        PROJECT_ROOT / "reports" / "data_quality_report.json",
        PROJECT_ROOT / "reports" / "feature_analysis.json",
        PROJECT_ROOT / "reports" / "metrics" / "sensor_feature_analysis.csv",
        PROJECT_ROOT / "reports" / "metrics" / "model_metrics.csv",
        PROJECT_ROOT / "reports" / "figures" / "pass_fail_distribution.png",
        PROJECT_ROOT / "reports" / "figures" / "missingness.png",
        PROJECT_ROOT / "reports" / "figures" / "correlation_heatmap.png",
        PROJECT_ROOT / "reports" / "figures" / "top_sensor_importance.png",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Full pipeline did not generate all expected outputs: " + "; ".join(missing))

    print("\nPipeline completed successfully.")
    print("Main outputs:")
    for path in required_paths:
        print(f"- {path}")


def launch_dashboard() -> None:
    dashboard_script = PROJECT_ROOT / "app" / "dashboard.py"
    if not dashboard_script.exists():
        raise FileNotFoundError(f"Dashboard script not found: {dashboard_script}")
    python_executable = Path(sys.executable)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
    print(f"Launching dashboard with: {python_executable} -m streamlit run {dashboard_script}")
    try:
        subprocess.run(
            [
                str(python_executable),
                "-m",
                "streamlit",
                "run",
                str(dashboard_script),
                "--server.headless",
                "true",
                "--server.address",
                "127.0.0.1",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")


def run_tests() -> None:
    print("Running project tests...")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(PROJECT_ROOT), check=False)


def main() -> None:
    print_banner()
    try:
        ensure_dependencies()
    except RuntimeError as exc:
        print(f"Dependency check failed: {exc}")
        return

    while True:
        print("1. Run full pipeline")
        print("2. Audit dataset")
        print("3. Run exploratory and statistical analysis")
        print("4. Explain sensor features")
        print("5. Train models")
        print("6. Evaluate models")
        print("7. Launch dashboard")
        print("8. Run tests")
        print("9. Exit")
        choice = input("Select an option: ").strip()

        try:
            if choice == "1":
                run_full_pipeline()
            elif choice == "2":
                run_audit()
            elif choice == "3":
                run_analysis()
            elif choice == "4":
                run_explain()
            elif choice == "5":
                run_train()
            elif choice == "6":
                run_evaluate()
            elif choice == "7":
                launch_dashboard()
            elif choice == "8":
                run_tests()
            elif choice == "9":
                print("Exiting.")
                break
            else:
                print("Invalid option. Please enter a number from 1 to 9.")
        except Exception as exc:
            print(f"Error: {exc}")
            print("Review the traceback above for details, then rerun the selected task.")


if __name__ == "__main__":
    main()
