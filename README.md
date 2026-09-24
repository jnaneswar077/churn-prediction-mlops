# Telco Customer Churn — MLOps Pipeline

> End-to-end Machine Learning Operations project for customer churn prediction, covering reproducible preprocessing, validation, experiment tracking, model registry, lifecycle management, automated ML pipelines, CI with GitHub Actions, and failure-handling/recovery workflows.

**Repository:** `janeswar077/churn-prediction-mlops`

**Current project scope:** Labs 1–9 completed. Lab 10 (FastAPI model serving) is the next stage and is intentionally not documented as completed here.

---

## 1. Project Overview

This project applies MLOps practices to a **Telco Customer Churn Prediction** problem.

The project started as a conventional ML workflow and was progressively transformed into a modular MLOps system. The implementation now separates data loading, validation, preprocessing, training, evaluation, quality gates, model lifecycle management, inference, CI automation, and failure/recovery testing.

The main goal is not only to train a churn model, but to demonstrate how an ML system can be made:

- reproducible
- modular
- traceable
- validated
- versioned
- automated
- failure-aware
- suitable for CI workflows
- ready for the next deployment/API layer

---

## 2. Dataset

The project uses the **Telco Customer Churn** dataset.

The current raw dataset is stored at:

```text
data/raw/churn.csv
```

The pipeline expects the following important fields:

- `customerID` — customer identifier
- `Churn` — prediction target (`Yes` / `No`)
- numerical customer/account attributes such as `tenure`, `MonthlyCharges`, and `TotalCharges`
- categorical customer/account/service attributes such as `Contract`, `InternetService`, `PaymentMethod`, etc.

The current local pipeline run loaded:

```text
7044 rows
21 columns
```

The train/test split is reproducible using:

```yaml
split:
  test_size: 0.2
  random_state: 42
```

Stratification is also used on the churn target.

---

## 3. MLOps Architecture

The current end-to-end pipeline is organized as a sequence of explicit stages:

```text
                    ┌──────────────────────┐
                    │   Raw Churn Dataset  │
                    │ data/raw/churn.csv   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 01. Data Loading     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 02. Raw Validation   │
                    │     Pandera Schema   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 03. Preprocessing    │
                    │ split / impute /     │
                    │ scale / encode       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 04. Output Validation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 05. Training +       │
                    │     MLflow Registry  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 06. Evaluation       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 07. Quality Gate     │
                    │ Recall >= 0.70       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 08. Model Lifecycle  │
                    │ Staging / Production │
                    │ / Archived           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 09. Production       │
                    │     Inference        │
                    └──────────────────────┘

          GitHub Actions automation wraps the pipeline
          for CI execution and artifact collection.

          Lab 9 separately exercises failure injection
          and model-artifact recovery.
```

---

## 4. Repository Structure

```text
churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── churn.csv
│   └── processed/
│       ├── X_train_final.npy
│       ├── X_test_final.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── X_train_raw.csv
│       ├── X_test_raw.csv
│       └── dataset_metadata.json
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validate_data.py
│   ├── validate_outputs.py
│   ├── preprocess_pipeline.py
│   ├── train_mlflow.py
│   ├── train_registry.py
│   ├── automate_lifecycle.py
│   ├── generate_registry_report.py
│   ├── validate_reproducibility.py
│   └── mlops_pipeline/
│       ├── 01_load_data.py
│       ├── 02_validate_data.py
│       ├── 03_preprocessing.py
│       ├── 04_validate_outputs.py
│       ├── 05_train_registry.py
│       ├── 06_model_evaluation.py
│       ├── 07_quality_gate.py
│       ├── 08_automate_lifecycle.py
│       ├── 09_inference.py
│       ├── 10_failure_injection.py
│       └── churn_prediction_pipeline.py
│
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   └── run_lab6_registry.py
│
├── models/
├── artifacts/
├── reports/
├── logs/
├── outputs/
│
├── docs/
│   ├── LAB7_PIPELINE.md
│   ├── LAB8_CI_REPORT.md
│   ├── LAB9_FAILURE_HANDLING_REPORT.md
│   └── MLFLOW_CI_SETUP.md
│
├── notebooks/
│   ├── churn_prediction.ipynb
│   └── project_implementation.ipynb
│
├── .github/
│   └── workflows/
│       ├── lab8_ci.yml
│       └── lab9_failure_tests.yml
│
├── config.yaml
├── requirements.txt
├── GIT_WORKFLOW.md
└── README.md
```

---

# 5. Lab-by-Lab Progress

## Lab 1 — ML Project Initialization and Dataset Preparation

The project was initialized around the Telco churn dataset with a structured ML repository layout.

Work completed includes:

- loading and inspecting the churn dataset
- checking datatypes and missing values
- cleaning `TotalCharges`
- converting the target `Churn` into a binary representation
- separating features and target
- reproducible train/test splitting
- storing processed data under `data/processed/`
- storing metadata describing the processed dataset
- organizing source code, models, outputs, reports, and pipeline scripts

Key artifacts include:

```text
data/processed/dataset_metadata.json
data/processed/X_train_final.npy
data/processed/X_test_final.npy
data/processed/y_train.npy
data/processed/y_test.npy
```

---

## Lab 2 — Baseline Model Development and Evaluation

A baseline Random Forest workflow was implemented before the later MLOps layers were introduced.

The baseline workflow includes:

- preprocessing the dataset
- training a Random Forest classifier
- saving the model with Joblib
- evaluating the held-out test set
- reporting classification metrics
- exporting false-positive and false-negative examples

Baseline model artifacts include files such as:

```text
models/random_forest_baseline.pkl
models/random_forest_model.pkl
outputs/false_negatives.csv
outputs/false_positives.csv
```

The project also contains model/experiment artifacts from the later stages, allowing the baseline work to be connected to the MLOps pipeline.

---

## Lab 3 — Git-Based Version Control and Collaborative ML Workflows

Git was used as a core part of the project lifecycle.

The documented workflow includes:

- repository initialization
- modular feature branches
- separation of preprocessing/training/evaluation concerns
- simulated independent developer changes
- intentional merge-conflict creation
- manual conflict resolution
- faulty commit introduction
- validation-based fault detection
- rollback using `git revert`
- meaningful commit history and traceability

Detailed evidence and history are documented in:

```text
GIT_WORKFLOW.md
```

This demonstrates that version control is being treated as part of the ML engineering process rather than simply as source-code backup.

---

## Lab 4 — Experiment Tracking and Reproducibility

MLflow was introduced to track model experiments.

The experiment workflow records:

- model parameters
- model family
- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- model artifacts
- preprocessing metadata
- diagnostic plots

Tracked diagnostic artifacts include:

```text
artifacts/confusion_matrix.png
artifacts/roc_curve.png
```

The project also includes a reproducibility test using fixed Random Forest parameters and repeated training runs. The reproducibility report records the comparison between repeated executions.

Relevant artifact:

```text
artifacts/reproducibility_report.json
```

---

## Lab 5 — Data Validation and Reproducible ML Pipelines

The project evolved from separate preprocessing scripts into schema-aware and modular preprocessing workflows.

### Dataset validation

Pandera is used to validate the raw Telco dataset using:

- required columns
- expected datatypes
- categorical allowed values
- numerical range checks
- strict schema validation

Examples of validation constraints include:

```text
SeniorCitizen ∈ {0, 1}
tenure >= 0
MonthlyCharges >= 0
Churn ∈ {Yes, No}
```

### Reproducible preprocessing

The modular preprocessing pipeline uses Scikit-learn components:

```text
Numerical features
    ↓
SimpleImputer
    ↓
StandardScaler

Categorical features
    ↓
OneHotEncoder

Numerical + categorical branches
    ↓
ColumnTransformer
```

The preprocessing pipeline is fitted on the training split and then applied to the test split to avoid leakage.

The fitted preprocessing object is stored as:

```text
models/preprocessor.pkl
```

Validation is performed again after preprocessing to check:

- NaNs
- train/test feature dimensionality
- feature/label row alignment
- existence of required raw split artifacts

---

## Lab 6 — Model Registry and Version Management

MLflow Model Registry was introduced to manage model versions.

The registered model name is:

```text
Telco_Churn_Production_Model
```

The lifecycle implemented in the project uses:

```text
None → Staging → Production / Archived
```

The lifecycle manager:

1. finds newly registered versions
2. moves new versions to Staging
3. reads evaluation metrics from MLflow
4. compares candidate and Production models
5. promotes a better candidate when the decision metric improves
6. archives models that are not improvements
7. produces a deployment-oriented registry report

The configured lifecycle decision metric is:

```yaml
registry:
  decision_metric: recall
```

---

# 6. Lab 7 — ML Pipeline Engineering

Lab 7 introduced the integrated end-to-end ML pipeline.

The pipeline is orchestrated by:

```text
src/mlops_pipeline/churn_prediction_pipeline.py
```

Running:

```bash
python src/mlops_pipeline/churn_prediction_pipeline.py
```

executes these stages sequentially:

```text
01 Data Loading
02 Raw Data Validation
03 Preprocessing
04 Processed Output Validation
05 Training + MLflow + Model Registry
06 Model Evaluation
07 Model Quality Gate
08 Automated Model Lifecycle
09 Production Inference
```

Every stage is executed as a separate Python process.

The orchestrator records:

- stage name
- stage status
- exit code
- overall pipeline status

The run log is written to:

```text
logs/churn_prediction_pipeline_run_log.json
```

A stage failure immediately stops the remaining downstream stages.

This creates a fail-fast execution model:

```text
Stage N fails
     ↓
Pipeline HALT
     ↓
Downstream stages do not run
```

### Pipeline outputs

The pipeline produces artifacts across several directories:

```text
artifacts/
reports/
logs/
models/
outputs/
```

Examples:

```text
artifacts/data_load_report.json
artifacts/preprocessing_summary_report.json
reports/evaluation_report.json
reports/quality_gate_report.json
outputs/inference_predictions.csv
```

---

# 7. Model Training and Evaluation

The integrated pipeline currently uses a Random Forest classifier with parameters controlled by `config.yaml`.

Current configuration:

```yaml
model:
  n_estimators: 150
  max_depth: 12
  random_state: 42
  class_weight: balanced
```

The evaluation stage records:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix

A successful local execution produced the following example results on the held-out test split:

```text
Accuracy  : 0.7615
Precision : 0.5374
Recall    : 0.7299
F1 Score  : 0.6190
ROC-AUC   : 0.8406
```

These values are an example from a successful run and are not intended to be hard-coded as permanent project guarantees.

---

# 8. Model Quality Gate

A dedicated quality-gate stage was added before lifecycle promotion.

Current configuration:

```yaml
quality_gate:
  metric: recall
  minimum_value: 0.70
```

The logic is:

```text
Evaluation report
      ↓
Read recall
      ↓
Recall >= 0.70 ?
   ┌───┴───┐
  YES      NO
   ↓        ↓
 PASS     FAIL
   ↓        ↓
Lifecycle  HALT
```

The quality gate report is saved to:

```text
reports/quality_gate_report.json
```

---

# 9. Automated Model Lifecycle

After the quality gate passes, the lifecycle stage manages the registered model versions.

The current logic compares the configured decision metric between candidate and Production models.

For example:

```text
Candidate → Staging
Candidate recall = 0.7299

Production recall = 0.7807

Candidate is not better
        ↓
Production remains unchanged
        ↓
Candidate is archived
```

This prevents an evaluated candidate from automatically replacing a stronger Production version.

> Note: MLflow's legacy Model Registry stage-transition API currently emits deprecation warnings in the local run. The workflow still executes successfully in the current project state.

---

# 10. Production Inference

The final stage demonstrates inference against the Production model registered in MLflow.

The pipeline loads:

```text
models:/Telco_Churn_Production_Model/Production
```

It loads the persisted preprocessing artifact:

```text
models/preprocessor.pkl
```

A sample of production-like input data is transformed and passed to the registered model.

The inference output contains:

```text
Predicted_Churn
Churn_Probability
```

The resulting CSV is saved to:

```text
outputs/inference_predictions.csv
```

---

# 11. Lab 8 — Pipeline Automation and CI with GitHub Actions

Lab 8 wraps the normal ML pipeline in GitHub Actions rather than creating a separate CI-only ML pipeline.

The main workflow is:

```text
.github/workflows/lab8_ci.yml
```

### CI workflow

```text
Git push / Pull Request / Manual trigger
                ↓
       Checkout repository
                ↓
          Setup Python 3.11
                ↓
       Install requirements.txt
                ↓
 run churn_prediction_pipeline.py
                ↓
      upload CI artifacts
```

### Current triggers

The workflow watches changes to:

```text
config.yaml
src/mlops_pipeline/**
requirements.txt
data/raw/churn.csv
.github/workflows/lab8_ci.yml
```

It supports:

- `push`
- `pull_request`
- `workflow_dispatch`

### GitHub runner

The workflow runs on:

```text
ubuntu-latest
```

with:

```text
Python 3.11
```

### CI artifacts

The workflow uploads artifacts from:

```text
artifacts/**
reports/**
logs/**
outputs/**
models/**
mlruns/**
```

### MLflow in CI

The workflow supports a persistent external MLflow tracking server through the secret:

```text
MLFLOW_TRACKING_URI
```

The pipeline first checks the environment variable and then falls back to the value in `config.yaml`.

### Dataset availability in CI

The raw dataset must be available in the GitHub repository checkout because the CI runner starts from a fresh environment.

This was important during CI debugging: the workflow initially failed at the data-loading stage because `data/raw/churn.csv` was excluded by `.gitignore`. After making the dataset available to Git, the GitHub Actions workflow completed successfully.

---

# 12. Lab 8 — Jenkins Comparison

The repository documents the conceptual difference between GitHub Actions and Jenkins.

| Feature | GitHub Actions | Jenkins |
|---|---|---|
| Repository integration | Native to GitHub | Requires integration/setup |
| Workflow definition | YAML | Jenkinsfile / UI |
| Hosted runners | Available | Normally self-managed |
| Setup for this project | Lightweight | More infrastructure |
| Project usage | Implemented | Compared conceptually |

Detailed CI notes are available in:

```text
docs/LAB8_CI_REPORT.md
```

---

# 13. Lab 9 — Pipeline Failure Handling and Recovery

Lab 9 intentionally injects faults into important parts of the pipeline to verify that the system fails safely or recovers correctly.

The failure-injection script is:

```text
src/mlops_pipeline/10_failure_injection.py
```

Run it with:

```bash
python src/mlops_pipeline/10_failure_injection.py
```

The configured number of repetitions is:

```yaml
lab9:
  repetitions: 3
```

### Failure scenarios

| Scenario | Expected behavior | Recovery behavior |
|---|---|---|
| Missing raw dataset | HALT | None; input must be restored |
| Invalid datatype | HALT | None; corrupted input is restored |
| Schema mismatch | HALT | None; corrupted input is restored |
| Corrupted processed output | HALT | None; corrupted artifact is restored |
| Missing local model artifact | Detect failure | Recover Production model from MLflow |

### Recovery path

The model-artifact recovery scenario follows this pattern:

```text
models/churn_model.pkl
       ↓
   intentionally removed
       ↓
model evaluation fails
       ↓
load Production model from MLflow Registry
       ↓
restore models/churn_model.pkl
       ↓
run evaluation again
       ↓
recovery succeeds
```

### Safety principle

Invalid or corrupted input data is not silently accepted. Validation failures stop downstream execution.

The failure test report is written to:

```text
reports/failure_report.json
```

Additional CI artifacts can be collected through:

```text
.github/workflows/lab9_failure_tests.yml
```

Detailed documentation is available in:

```text
docs/LAB9_FAILURE_HANDLING_REPORT.md
```

---

# 14. Reproducibility

Reproducibility is enforced at several levels.

### Fixed split

```yaml
test_size: 0.2
random_state: 42
```

### Fixed model seed

```yaml
random_state: 42
```

### Reusable preprocessing

The preprocessing configuration is built from a deterministic Scikit-learn pipeline and persisted to:

```text
models/preprocessor.pkl
```

### Repeated training validation

`src/validate_reproducibility.py` trains the same model configuration twice and compares the resulting F1 scores.

Report:

```text
artifacts/reproducibility_report.json
```

---

# 15. Configuration-Driven Design

Most pipeline behavior is centralized in:

```text
config.yaml
```

Important sections include:

```yaml
data:
split:
model:
mlflow:
registry:
quality_gate:
lab9:
paths:
```

This avoids hard-coding the same settings across multiple pipeline stages.

For example:

```yaml
registry:
  model_name: Telco_Churn_Production_Model
  decision_metric: recall
```

and:

```yaml
quality_gate:
  metric: recall
  minimum_value: 0.70
```

---

# 16. Running the Project Locally

## Create and activate an environment

Windows PowerShell example:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Make sure the dataset exists at:

```text
data/raw/churn.csv
```

## Run the integrated MLOps pipeline

From the project root:

```powershell
python .\src\mlops_pipeline\churn_prediction_pipeline.py
```

Expected final message:

```text
[SUCCESS] Churn Prediction MLOps Pipeline completed successfully.
```

## Run Lab 9 directly

```powershell
python .\src\mlops_pipeline\10_failure_injection.py
```

## Run the earlier lab pipelines

Lab 3-style baseline pipeline:

```powershell
python .\pipelines\run_lab3_baseline.py
```

Lab 4 experiment tracking:

```powershell
python .\pipelines\run_lab4_tracking.py
```

Lab 5 validation/preprocessing pipeline:

```powershell
python .\pipelines\run_lab5_pipeline.py
```

Lab 6 model registry/lifecycle pipeline:

```powershell
python .\pipelines\run_lab6_registry.py
```

---

# 17. Important Artifacts

| Location | Purpose |
|---|---|
| `data/raw/churn.csv` | Raw Telco churn dataset |
| `data/processed/` | Reproducible processed data and split artifacts |
| `models/preprocessor.pkl` | Persisted preprocessing pipeline |
| `models/churn_model.pkl` | Current locally saved model artifact |
| `artifacts/data_load_report.json` | Data loading metadata |
| `artifacts/preprocessing_summary_report.json` | Processed-output validation report |
| `artifacts/reproducibility_report.json` | Reproducibility validation |
| `reports/evaluation_report.json` | Model evaluation metrics |
| `reports/quality_gate_report.json` | Quality-gate decision |
| `reports/failure_report.json` | Lab 9 failure/recovery results |
| `outputs/inference_predictions.csv` | Production inference output |
| `logs/churn_prediction_pipeline_run_log.json` | Pipeline stage status and exit codes |

---

# 18. Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest

### Data Validation

- Pandera

### MLOps / Tracking

- MLflow
- MLflow Model Registry
- Joblib

### Automation / CI

- Git
- GitHub
- GitHub Actions

### Configuration / Serialization

- YAML
- JSON
- CSV
- NumPy `.npy`
- Joblib `.pkl`

### Current project status

```text
Labs 1–6   ✅
Lab 7      ✅
Lab 8      ✅
Lab 9      ✅
Lab 10     ⏳ Next: FastAPI model serving
```

---

# 19. Design Principles Demonstrated

This project intentionally demonstrates several MLOps engineering principles:

### Fail fast

A failed stage stops downstream pipeline execution.

### Validate before training

Raw data and processed outputs are validated before the model proceeds.

### Reproducibility

Fixed seeds, configuration files, persisted preprocessing, and repeatability checks are used.

### Traceability

Git history, MLflow runs, model versions, configuration, reports, and logs provide a chain of evidence for model development.

### Quality gates

A minimum recall threshold prevents a weak candidate from entering lifecycle promotion.

### Controlled promotion

Production models are compared against candidate versions using a configured metric.

### Failure recovery

Lab 9 verifies both safe failure behavior and registry-based model recovery.

### CI automation

GitHub Actions executes the same pipeline used locally rather than maintaining a separate CI-only implementation.

---

# 20. Evidence and Documentation

Detailed lab-specific documentation is available under `docs/`:

```text
docs/LAB7_PIPELINE.md
docs/LAB8_CI_REPORT.md
docs/LAB9_FAILURE_HANDLING_REPORT.md
docs/MLFLOW_CI_SETUP.md
```

Git workflow evidence:

```text
GIT_WORKFLOW.md
```

---

# 21. Lab 10 — Next Step

The next planned stage is **ML Model Serving and API Engineering using FastAPI**.

The existing MLOps pipeline already produces the key assets required by the serving layer:

```text
registered Production model
persisted preprocessing pipeline
model evaluation report
model quality-gate report
inference workflow
```

The next stage will add an API layer around those assets rather than rebuilding the ML pipeline.

Planned Lab 10 work includes:

- FastAPI application
- health-check endpoint
- prediction endpoint
- metadata endpoint
- batch inference endpoint
- request/response schemas
- input validation and datatype enforcement
- standardized API errors
- API test cases
- latency/throughput measurements
- inference reliability documentation

Lab 10 should be treated as the **serving layer on top of the completed Labs 1–9 MLOps foundation**.

---

# 22. Final Project Status

At the current milestone, the project has moved from a standalone churn-prediction model to a structured MLOps workflow:

```text
Dataset
  ↓
Validation
  ↓
Reproducible preprocessing
  ↓
Training
  ↓
MLflow experiment tracking
  ↓
Model Registry
  ↓
Evaluation
  ↓
Quality Gate
  ↓
Lifecycle management
  ↓
Production inference
  ↓
GitHub Actions CI
  ↓
Failure injection + recovery
  ↓
[ NEXT ] FastAPI serving
```

The project is therefore ready to proceed to **Lab 10: FastAPI Model Serving and API Engineering**.
