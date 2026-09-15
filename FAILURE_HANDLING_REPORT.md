# Failure Handling and Recovery Report — Lab 9

**Project:** churn-prediction (Telco Customer Churn MLOps Pipeline)
**Lab:** Lab 9 — Pipeline Failure Handling and Recovery

This report documents the 5 required failure scenarios simulated
against the Lab 7 pipeline, the real root cause observed for each, the
one recovery mechanism that was deliberately built, and the reasoning
behind not building recovery for the other four. All results below are
from actual runs (`src/inject_failures.py`), not hypothetical
descriptions.

---

## 1. Failure Scenarios and Observed Root Causes

Each scenario was simulated in isolation — one fault injected at a
time, pipeline run, files restored — so the root cause for each is
unambiguous.

| # | Scenario | Where it was caught | Root cause |
|---|---|---|---|
| 1 | Missing data | Stage 1 — Pre-Validation (raw schema) | `data/raw/churn.csv` deleted; `validate_data.py` fails immediately on file-not-found, before any transformation is attempted |
| 2 | Invalid datatypes | Stage 1 — Pre-Validation (raw schema) | A text value (`"One Hundred"`) written into `MonthlyCharges`; Pandera's schema check rejects it on both the dtype check and the `>= 0` numeric check |
| 3 | Schema mismatch | Stage 1 — Pre-Validation (raw schema) | An invalid categorical value (`Contract = "Lifetime"`) plus a missing required column (`gender` dropped); both violations caught in the same schema validation pass |
| 4 | Preprocessing inconsistency | Stage 1 — Post-Validation (processed outputs) | The numeric imputer step was removed from the sklearn `Pipeline` construction itself (a code bug, not bad input data); NaNs from blank `TotalCharges` values leak through to `X_train`/`X_test`, caught by the post-preprocessing NaN check |
| 5 | Missing model artifact | N/A — recovered, did not fail | `models/churn_full_pipeline.pkl` deleted; `predict.py` detects this and falls back to the legacy `preprocessor.pkl` + `random_forest_baseline.pkl` artifacts |

Scenarios 1–4 all correctly **halt** the pipeline before it can produce
a misleading result. Scenario 5 is the one case where **recovery**,
not halting, was the better-engineered response — covered below.

---

## 2. Why Only One Fallback Mechanism Was Built

The syllabus asks for "fallback mechanisms and failure recovery
workflows," but building recovery for every possible failure is not
automatically good engineering — recovering from the wrong kind of
failure can be worse than stopping. Each of the 5 scenarios was
evaluated on that basis:

- **Missing model artifact → fallback built.** This is a realistic
  deployment situation: someone requests a prediction before a model
  has been (re)trained, or an artifact fails to sync correctly. A
  clear, safe fallback already existed in the repo from earlier labs
  (the separate `preprocessor.pkl` + `random_forest_baseline.pkl`), so
  `predict.py` was updated to use them automatically, with a loud
  `[RECOVERY]` warning marking the degraded mode rather than pretending
  nothing happened.
- **Missing data, invalid datatypes, schema mismatch, preprocessing
  inconsistency → no fallback built, by choice.** These all represent
  corrupted or missing *business data*, or a bug in the pipeline's own
  logic. Silently continuing past any of these — guessing at missing
  values, coercing bad data into shape, or proceeding with a known
  preprocessing bug — would produce a model trained or evaluated on
  data that cannot be trusted. Halting and surfacing the exact error is
  the correct behavior here; a fallback would hide the problem rather
  than solve it. Recovering from a missing raw dataset specifically was
  also considered and rejected on cost/benefit grounds: it would
  require wiring fallback logic into every stage that reads the file,
  for a scenario (losing the source dataset entirely) that is a
  low-probability event for a project of this scope.

This is the central engineering judgment this lab is meant to test:
knowing *which* failures deserve automatic recovery, not building
recovery indiscriminately.

---

## 3. Recovery Mechanism Detail

`src/predict.py` now loads the model through `load_pipeline_with_fallback()`:

1. Check for the primary unified pipeline (`models/churn_full_pipeline.pkl`).
2. If missing, check for both legacy fallback artifacts
   (`models/preprocessor.pkl`, `models/random_forest_baseline.pkl`).
3. If both are present, wrap them in a small adapter that exposes the
   same `.predict()`/`.predict_proba()` interface as the primary
   pipeline, print an explicit `[RECOVERY]` warning naming the degraded
   mode, and continue serving real predictions.
4. If neither the primary nor the fallback artifacts exist, raise a
   clear error — this is the genuine, unrecoverable case.

Real output from a triggered fallback:
```
[WARNING] Primary unified pipeline not found at: models/churn_full_pipeline.pkl
[RECOVERY] Falling back to legacy artifacts: models/preprocessor.pkl + models/random_forest_baseline.pkl
[RECOVERY] Serving inference in DEGRADED MODE (older model, not the current
unified pipeline). Retrain via src/train_full_pipeline.py when possible.
```

The failure-injection harness deliberately tests this scenario against
`predict.py` in isolation rather than the full pipeline orchestrator.
Running the full orchestrator would retrain a fresh model in Stage 2
before inference ever ran, masking the fault entirely — unrealistic,
since a real deployment does not retrain on every inference request.

---

## 4. Repeated-Execution Consistency

`src/inject_failures.py` was run 3 times in a row (not simulated —
actually executed on both the development machine and verified
separately). All 5 scenarios produced identical outcomes across every
run:

| Scenario | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| missing_data | HALT — OK | HALT — OK | HALT — OK |
| invalid_datatypes | HALT — OK | HALT — OK | HALT — OK |
| schema_mismatch | HALT — OK | HALT — OK | HALT — OK |
| preprocessing_inconsistency | HALT — OK | HALT — OK | HALT — OK |
| missing_model_artifact | RECOVER — OK | RECOVER — OK | RECOVER — OK |

No flakiness was observed. After each run, the harness restores every
modified file automatically, and a plain run of
`pipelines/run_lab7_pipeline.py` afterward confirmed the pipeline
returns to `PIPELINE COMPLETED SUCCESSFULLY` — the fault-injection
process itself leaves no residue.

---

## 5. Failure Propagation Analysis

Every one of the 4 halted scenarios was caught at **Stage 1**
(pre-validation or post-validation), before reaching training,
evaluation, or inference. This means no corrupted input ever has a
chance to propagate downstream into a trained model or a served
prediction — the fail-fast design established in Lab 7 (verified there
with a single injected schema fault) held up consistently across all
4 distinct failure types tested here, not just the one case it was
originally built for.

The one exception, by design, is Scenario 5: it is caught at the
inference stage itself, because that is where the fault (a missing
model file) actually lives — there is nothing to catch earlier, since
Stages 1–4 never touch the model file predict.py depends on.

---

## 6. Summary

| Requirement | Evidence |
|---|---|
| Simulate all 5 required failure types | `src/inject_failures.py`, Section 1 table |
| Observe failures and identify root causes across stages | Section 1 — each scenario's exact failing stage and cause |
| Add validation checks, exception handling, fallback mechanisms | `predict.py`'s `load_pipeline_with_fallback()`, Section 3 |
| Test robustness and recovery consistency across repeated executions | Section 4 — 3 identical runs |
| Analyze failure propagation and reliability under corrupted conditions | Section 5 |
| Debugging/recovery/reliability documentation | This report + `artifacts/failure_simulation_report.json` |
