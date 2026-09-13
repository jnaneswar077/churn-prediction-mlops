import os
import sys
import json
import shutil
import subprocess
import pandas as pd

DATA_PATH = 'data/raw/churn.csv'
MODEL_PATH = 'models/churn_full_pipeline.pkl'
PREPROCESS_SCRIPT = 'src/preprocess_pipeline.py'
PIPELINE_LOG = 'logs/pipeline_run_log.json'


# ---------------------------------------------------------------------
# The 5 required failure scenarios. Each "apply" function corrupts the
# real file(s) in place; each scenario is run in isolation (one fault
# at a time) so the root cause is unambiguous.
# ---------------------------------------------------------------------

def apply_missing_data():
    """Scenario 1: missing data -- the raw dataset simply isn't there."""
    os.remove(DATA_PATH)


def apply_invalid_datatypes():
    """Scenario 2: invalid datatypes -- a string where a number belongs.
    Read as dtype=str first, so the value is written into the raw CSV
    as plain text (matching how a real corrupted source file would look)
    rather than being blocked by pandas' in-memory dtype guard."""
    df = pd.read_csv(DATA_PATH, dtype=str)
    df.loc[3, 'MonthlyCharges'] = "One Hundred"
    df.to_csv(DATA_PATH, index=False)


def apply_schema_mismatch():
    """Scenario 3: schema mismatch -- invalid categorical value plus a
    missing required column (two distinct schema violations at once,
    same category the syllabus groups together)."""
    df = pd.read_csv(DATA_PATH)
    df.loc[1, 'Contract'] = 'Lifetime'
    df = df.drop(columns=['gender'])
    df.to_csv(DATA_PATH, index=False)


def apply_preprocessing_inconsistency():
    """Scenario 4: preprocessing inconsistency -- a bug in the pipeline's
    own construction, not the input data. The numeric imputer step is
    accidentally dropped from the sklearn Pipeline, so any NaN values
    (e.g. blank TotalCharges, which the raw schema legitimately allows
    as nullable) reach StandardScaler directly -- which cannot handle
    NaNs and raises an error deep inside preprocessing itself."""
    with open(PREPROCESS_SCRIPT) as f:
        content = f.read()

    target = (
        "    num_pipeline = Pipeline(steps=[\n"
        "        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),\n"
        "        ('scaler', StandardScaler())\n"
        "    ])"
    )
    broken = (
        "    num_pipeline = Pipeline(steps=[\n"
        "        ('scaler', StandardScaler())\n"
        "    ])"
    )
    if target not in content:
        raise RuntimeError("Could not locate the num_pipeline block to corrupt in "
                            "preprocess_pipeline.py -- has the file changed since this "
                            "scenario was written?")
    new_content = content.replace(target, broken)
    with open(PREPROCESS_SCRIPT, 'w') as f:
        f.write(new_content)


def apply_missing_model_artifact():
    """Scenario 5: missing model artifact -- the trained pipeline file
    is gone (e.g. deleted, failed upload, wrong path in a fresh env)."""
    os.remove(MODEL_PATH)


SCENARIOS = {
    "missing_data": (apply_missing_data, None),
    "invalid_datatypes": (apply_invalid_datatypes, None),
    "schema_mismatch": (apply_schema_mismatch, None),
    "preprocessing_inconsistency": (apply_preprocessing_inconsistency, None),
    # This one is deliberately run against predict.py ALONE, not the
    # full orchestrator. Running the full orchestrator would just
    # retrain a fresh model in Stage 2, masking the fault -- unrealistic,
    # since a real deployment does not retrain on every inference call.
    # The real test is: what happens if you try to serve inference with
    # no model artifact and no retraining step available?
    "missing_model_artifact": (apply_missing_model_artifact, "src/predict.py"),
}


def backup_files():
    backups = {}
    for path in [DATA_PATH, MODEL_PATH, PREPROCESS_SCRIPT]:
        if os.path.exists(path):
            backup_path = path + ".failure_test_backup"
            shutil.copy2(path, backup_path)
            backups[path] = backup_path
    return backups


def restore_files(backups):
    for original_path, backup_path in backups.items():
        shutil.copy2(backup_path, original_path)
        os.remove(backup_path)
    # If the corrupted run created the raw file's absence, and there was
    # no backup because it didn't need one, nothing else to restore.


def run_pipeline_and_capture(target_script=None):
    """Runs either the full Lab 7 orchestrator (default) or a single
    target script in isolation, and returns (exit_code, run_log_or_None).

    For the full orchestrator, the structured logs/pipeline_run_log.json
    it writes is used to identify exactly which stage failed. For a
    single target script, there is no such structured log -- its own
    stdout/stderr are captured directly instead.
    """
    script = target_script or 'pipelines/run_lab7_pipeline.py'
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True
    )

    if target_script is None:
        run_log = None
        if os.path.exists(PIPELINE_LOG):
            with open(PIPELINE_LOG) as f:
                run_log = json.load(f)
        return result.returncode, run_log
    else:
        # Wrap the single script's own output in the same shape
        # find_failed_stage() expects, so downstream code doesn't need
        # two separate code paths.
        combined_tail = (result.stdout + result.stderr).strip().splitlines()[-5:]
        synthetic_log = {
            "stages": [{
                "stage": target_script,
                "status": "FAILED" if result.returncode != 0 else "PASSED",
                "stderr_tail": combined_tail,
            }]
        }
        return result.returncode, synthetic_log


def find_failed_stage(run_log):
    if run_log is None:
        return None, []
    for stage in run_log.get("stages", []):
        if stage["status"] == "FAILED":
            return stage["stage"], stage.get("stderr_tail") or stage.get("stdout_tail") or []
    return None, []


def run_scenario(name, apply_fault_fn, target_script=None):
    print(f"\n{'#'*70}")
    print(f"# SCENARIO: {name}" + (f"  (target: {target_script})" if target_script else ""))
    print(f"{'#'*70}")

    backups = backup_files()
    try:
        apply_fault_fn()
        exit_code, run_log = run_pipeline_and_capture(target_script)
        failed_stage, error_lines = find_failed_stage(run_log)

        detected = (exit_code != 0)
        result = {
            "scenario": name,
            "pipeline_exit_code": exit_code,
            "failure_detected": detected,
            "failed_at_stage": failed_stage,
            "error_evidence": error_lines,
        }

        if detected:
            print(f"[RESULT] Failure correctly detected at: {failed_stage}")
        else:
            print(f"[RESULT] WARNING -- pipeline exited 0 despite injected fault '{name}'. "
                  f"This fault was NOT caught.")

        return result

    except Exception as e:
        return {
            "scenario": name,
            "pipeline_exit_code": None,
            "failure_detected": True,
            "failed_at_stage": "harness-level exception (before pipeline even ran)",
            "error_evidence": [str(e)],
        }

    finally:
        restore_files(backups)
        print(f"[INFO] Restored original files after '{name}' scenario.")


def run_all_scenarios():
    print("[INFO] Starting Lab 9 Stage 1-2: Failure Injection and Root-Cause Observation...")
    results = []
    for name, (apply_fn, target_script) in SCENARIOS.items():
        results.append(run_scenario(name, apply_fn, target_script))

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/failure_simulation_report.json", "w") as f:
        json.dump({"scenarios": results}, f, indent=4)

    print(f"\n{'='*70}")
    print("FAILURE SIMULATION SUMMARY")
    print('='*70)
    all_detected = True
    for r in results:
        status = "DETECTED" if r["failure_detected"] else "**NOT DETECTED**"
        print(f"  {r['scenario']:32s} -> {status:14s} (stage: {r['failed_at_stage']})")
        all_detected = all_detected and r["failure_detected"]

    print(f"\nReport saved to artifacts/failure_simulation_report.json")
    return all_detected


if __name__ == "__main__":
    all_ok = run_all_scenarios()
    sys.exit(0 if all_ok else 1)