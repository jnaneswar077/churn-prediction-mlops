# Pipeline Engineering Report — Lab 7

**Project:** churn-prediction (Telco Customer Churn MLOps Pipeline)
**Lab:** Lab 7 — ML Pipeline Engineering

This report analyzes the modularity, reliability, scalability, and
maintainability of the unified pipeline built for this lab, chaining
preprocessing, training, validation, evaluation, and inference into a
single orchestrated workflow (`pipelines/run_lab7_pipeline.py`).

---

## 1. Architecture Overview

```
run_lab7_pipeline.py
   │
   ├─► Stage 1  Pre-Validation      src/validate_data.py           (raw schema check)
   ├─► Stage 1  Preprocessing       src/preprocess_pipeline.py     (sklearn ColumnTransformer)
   ├─► Stage 1  Post-Validation     src/validate_outputs.py        (NaN / shape check)
   ├─► Stage 2  Training            src/train_full_pipeline.py     (unified sklearn Pipeline)
   ├─► Stage 4  Evaluation          src/evaluate_full_pipeline.py  (end-to-end, raw input)
   └─► Stage 5  Inference           src/predict.py                 (raw input -> prediction)
```

Each stage is an independent script, invoked as its own subprocess with
its own exit code. The orchestrator treats a non-zero exit code as a
hard failure and halts immediately (fail-fast), rather than continuing
to run stages against data or a model it can no longer trust.

---

## 2. Modularity

Each pipeline stage is a separately runnable, separately testable
script with a single responsibility — preprocessing does not know about
training, training does not know about evaluation, and so on. This was
tested directly: every stage was run and verified in isolation *before*
being wired into the orchestrator, and each still runs standalone
afterward (e.g. `python src/predict.py --input new_customers.csv` works
with no other stage having to run first, as long as
`models/churn_full_pipeline.pkl` already exists).

The training stage goes a step further than the earlier labs: instead
of saving the preprocessor and the classifier as two separate files
that a caller has to remember to apply in order, `train_full_pipeline.py`
bundles both into a single `sklearn.pipeline.Pipeline` object
(`models/churn_full_pipeline.pkl`). This is what makes Stage 5
(inference) possible with zero manual preprocessing steps — raw
customer data goes in, a prediction comes out.

---

## 3. Reliability

Reliability was not just assumed — it was actively tested by
deliberately breaking things and observing what happened:

- **Silent corruption caught, not ignored.** During Stage 3 integration,
  it was discovered that `validate_data.py` and `validate_outputs.py`
  returned a pass/fail boolean internally but always exited with code
  `0` regardless of the result — meaning a subprocess-based orchestrator
  had no way to detect a failed validation. Both scripts were fixed to
  call `sys.exit(0 if passed else 1)`, and the fix was verified against
  both a passing and a deliberately corrupted case before being trusted
  inside the orchestrator.
- **Fail-fast halting, verified with a real fault.** A schema violation
  was deliberately injected into the raw dataset (an invalid `gender`
  value). The pipeline correctly failed at Stage 1 pre-validation and
  **halted before reaching training** — it did not waste time training
  a model on data it had already flagged as invalid, and did not
  silently produce a misleading result.
- **Graceful failure in inference.** `predict.py` and
  `evaluate_full_pipeline.py` were both tested with the model artifact
  deliberately removed. Both printed a clear, actionable error message
  and exited with code `1` — no unhandled Python traceback.

---

## 4. Reproducibility

`pipelines/check_lab7_reproducibility.py` runs the entire pipeline
twice, end to end, and compares the final evaluation metrics between
runs. Both executions produced **identical metrics** (accuracy,
precision, recall, F1, ROC-AUC all matched exactly), confirming that
fixing `random_state=42` at every stage — the split, the imputer, and
the classifier — is sufficient to make the whole pipeline
deterministic, not just the model-training step in isolation (which is
as far as the Lab 4 reproducibility check went).

---

## 5. Scalability

The current pipeline is appropriately scoped for a single-machine,
batch-style workflow of this dataset's size (~7,000 rows) and is not
claimed to be a distributed system. Scalability considerations that
*are* addressed:

- Each stage runs as an independent process, so stages could be moved
  to separate machines or containers (e.g. a CI runner) without code
  changes — this is exactly what Lab 8 will do.
- `predict.py` accepts an arbitrary external CSV via `--input`, so
  inference volume is not hardcoded to the training set size — it
  scales with whatever file is passed in.
- The unified pipeline object is a single serialized artifact, making
  it straightforward to load once and serve many predictions (as Lab
  10's API layer will do), rather than reloading multiple separate
  artifacts per request.

What is *not* addressed at this stage (deliberately, as it belongs to
later labs): distributed/parallel training, batching large inference
workloads, and horizontal scaling of the serving layer.

---

## 6. Maintainability

- Every new artifact this lab introduced (`train_full_pipeline.py`,
  `evaluate_full_pipeline.py`, `predict.py`, the two orchestrator
  scripts) was added alongside the existing Lab 2/4/5/6 scripts rather
  than replacing them — nothing that earlier labs depend on for
  evidence was modified or removed.
- `logs/pipeline_run_log.json` and `reports/lab7_full_pipeline_evaluation_report.json`
  give a structured, inspectable record of any given run, so debugging
  a failure doesn't require re-reading console scrollback.
- Because Stage 2 through Stage 5 are ordinary Python functions
  (`predict_churn()`, `run_full_pipeline_evaluation()`, etc.) and not
  just script-level code, they can be imported and reused directly —
  for example, Lab 10's API service can call `predict_churn()` from
  `src/predict.py` directly instead of shelling out to a subprocess.

---

## 7. Summary

| Quality attribute | Evidence |
|---|---|
| Modularity | 6 independently runnable stages; unified Pipeline artifact enables standalone inference |
| Reliability | Real exit-code bug found and fixed; fail-fast halt verified with an injected schema fault; graceful failure verified with a missing model artifact |
| Reproducibility | Two full end-to-end runs produced identical metrics |
| Scalability | Process-per-stage design; arbitrary-size external input for inference; single artifact for serving |
| Maintainability | No earlier-lab artifacts modified; structured JSON logs/reports; stage logic reusable as functions, not just scripts |
