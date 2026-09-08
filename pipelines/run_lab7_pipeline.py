import os
import sys
import json
import time
import subprocess
from datetime import datetime, timezone

# Ordered pipeline stages. Each is (display_name, script_path).
# Validation runs BOTH before preprocessing (raw schema check) and after
# it (transformed-output check) -- two checkpoints, not one.
STAGES = [
    ("Stage 1 - Pre-Validation (raw schema)", "src/validate_data.py"),
    ("Stage 1 - Preprocessing", "src/preprocess_pipeline.py"),
    ("Stage 1 - Post-Validation (processed outputs)", "src/validate_outputs.py"),
    ("Stage 2 - Unified Pipeline Training", "src/train_full_pipeline.py"),
    ("Stage 4 - End-to-End Evaluation", "src/evaluate_full_pipeline.py"),
    ("Stage 5 - Inference Demo", "src/predict.py"),
]


def run_stage(display_name, script_path):
    print(f"\n{'='*70}")
    print(f"RUNNING: {display_name}  ({script_path})")
    print('='*70)

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True
    )
    duration = round(time.time() - start, 2)

    # Show the real output live, same as if you'd run it yourself
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"--> {display_name}: {status} (exit code {result.returncode}, {duration}s)")

    return {
        "stage": display_name,
        "script": script_path,
        "status": status,
        "exit_code": result.returncode,
        "duration_seconds": duration,
        "stdout_tail": result.stdout.strip().splitlines()[-5:] if result.stdout.strip() else [],
        "stderr_tail": result.stderr.strip().splitlines()[-5:] if result.stderr.strip() else []
    }


def run_pipeline(fail_fast=True):
    """Runs every stage in order. Returns (all_passed, run_log)."""
    run_log = {
        "pipeline": "Lab 7 - Unified ML Pipeline",
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "stages": []
    }

    all_passed = True
    for display_name, script_path in STAGES:
        stage_result = run_stage(display_name, script_path)
        run_log["stages"].append(stage_result)

        if stage_result["status"] == "FAILED":
            all_passed = False
            print(f"\n[HALT] {display_name} failed. ", end="")
            if fail_fast:
                print("Stopping pipeline (fail-fast) -- later stages depend on this one's output.")
                break
            else:
                print("Continuing anyway (fail_fast=False).")

    run_log["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    run_log["overall_status"] = "PASSED" if all_passed else "FAILED"

    os.makedirs("logs", exist_ok=True)
    with open("logs/pipeline_run_log.json", "w") as f:
        json.dump(run_log, f, indent=4)

    print(f"\n{'='*70}")
    print(f"PIPELINE {'COMPLETED SUCCESSFULLY' if all_passed else 'HALTED DUE TO FAILURE'}")
    print(f"Full run log written to logs/pipeline_run_log.json")
    print('='*70)

    return all_passed, run_log


if __name__ == "__main__":
    success, _ = run_pipeline()
    sys.exit(0 if success else 1)