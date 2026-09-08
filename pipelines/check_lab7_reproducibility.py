import os
import sys
import json
import subprocess


REPORT_PATH = 'reports/lab7_full_pipeline_evaluation_report.json'


def run_full_pipeline_and_snapshot(run_label):
    print(f"\n{'#'*70}")
    print(f"# {run_label}")
    print(f"{'#'*70}")

    result = subprocess.run(
        [sys.executable, 'pipelines/run_lab7_pipeline.py'],
        capture_output=True, text=True
    )
    print(result.stdout[-1500:])  # tail, full output already shown by run_lab7_pipeline itself

    if result.returncode != 0:
        print(f"[ERROR] {run_label} did not complete successfully (exit code {result.returncode}).")
        return None

    if not os.path.exists(REPORT_PATH):
        print(f"[ERROR] Expected evaluation report not found at {REPORT_PATH}.")
        return None

    with open(REPORT_PATH) as f:
        report = json.load(f)

    return report['metrics']


def run_reproducibility_check():
    print("[INFO] Starting Lab 7 Stage 6: Full-Pipeline Reproducibility Check...")
    print("[INFO] This runs the ENTIRE pipeline twice, back-to-back, and compares the")
    print("[INFO] final evaluation metrics -- extends the Lab 4 idea from 'is the model")
    print("[INFO] reproducible' to 'is the whole pipeline reproducible'.")

    metrics_1 = run_full_pipeline_and_snapshot("EXECUTION 1")
    metrics_2 = run_full_pipeline_and_snapshot("EXECUTION 2")

    if metrics_1 is None or metrics_2 is None:
        print("[ERROR] Could not complete reproducibility check -- one or both runs failed.")
        return False

    is_reproducible = (metrics_1 == metrics_2)

    report = {
        "test_name": "Lab 7 - Full Pipeline Reproducibility Check",
        "scope": "End-to-end: preprocessing -> training -> evaluation, run twice",
        "execution_1_metrics": metrics_1,
        "execution_2_metrics": metrics_2,
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED"
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/pipeline_reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n{'='*70}")
    print("REPRODUCIBILITY CHECK RESULT")
    print('='*70)
    print(f"Execution 1 metrics: {metrics_1}")
    print(f"Execution 2 metrics: {metrics_2}")
    if is_reproducible:
        print("[SUCCESS] Pipeline is fully reproducible across repeated executions.")
    else:
        print("[FAILED] Metrics differ between runs -- pipeline is NOT deterministic.")
    print("Report saved to artifacts/pipeline_reproducibility_report.json")

    return is_reproducible


if __name__ == "__main__":
    passed = run_reproducibility_check()
    sys.exit(0 if passed else 1)