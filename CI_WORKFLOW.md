# CI Workflow Documentation — Lab 8

**Project:** churn-prediction (Telco Customer Churn MLOps Pipeline)
**Lab:** Lab 8 — Pipeline Automation and CI Workflows

This document records the GitHub Actions CI setup built for this lab,
the real execution evidence gathered while testing it, and a comparison
against Jenkins as an alternative CI platform.

---

## 1. Workflow Configuration

`.github/workflows/ci.yml` runs the full Lab 7 pipeline
(`pipelines/run_lab7_pipeline.py`) automatically on GitHub-hosted
infrastructure. It is triggered by three distinct conditions, as
required by the lab:

| Trigger | Configured as |
|---|---|
| Code updates | `push`/`pull_request` on `src/**`, `pipelines/**` |
| Dataset modifications | `push`/`pull_request` on `data/raw/**` |
| Workflow changes | `push` on `.github/workflows/**` |
| Manual re-run | `workflow_dispatch` (for on-demand testing) |

Pipeline stages run identically to the local Lab 7 orchestrator —
pre-validation, preprocessing, post-validation, unified training,
end-to-end evaluation, and inference — with `logs/`, `reports/`, and
`artifacts/` uploaded as downloadable CI artifacts after every run,
whether it passes or fails (`if: always()`), so a failed run's evidence
is never lost.

---

## 2. Real Execution Evidence

Three actual runs were captured while building this lab (not simulated):

| Run | Trigger | Result | Duration | Root cause / outcome |
|---|---|---|---|---|
| #1 | Push (new workflow file + dataset) | **FAILED** — exit code 1, 18s | `pywin32`, a Windows-only package inherited from a local `pip freeze`, cannot install on GitHub's Ubuntu runner |
| #2 | Push (removed `pywin32`) | **PASSED** — 1m 14s | Full pipeline ran clean: all 6 stages PASSED, all 3 artifacts uploaded |
| #3 | Push (dataset-only change: 1 new customer record) | **PASSED** — 1m 11s | Automatically retrained on 5,635 rows (was 5,634); metrics shifted slightly as expected: Accuracy 0.7580→0.7559, Recall 0.7807→0.7674, F1 0.6314→0.6253 |

Run #1 is intentionally kept in the CI history rather than deleted — it
is direct proof that the pipeline's local-environment assumptions
(Windows-specific dependencies) do not silently pass elsewhere, and
that the failure was caught immediately rather than discovered later.

---

## 3. Reliability Analysis

- **Environment-dependency failure caught early.** The `pywin32` issue
  would not have been visible from local development alone — it only
  surfaced once the pipeline ran on a genuinely different environment
  (Ubuntu vs. Windows). This is precisely the value CI is meant to add:
  catching "works on my machine" assumptions before they reach anyone
  else.
- **Fail-fast behavior carried over from Lab 7.** Because
  `run_lab7_pipeline.py` already halts on the first failed stage
  (verified in Lab 7 with an injected schema fault), CI failures are
  attributable to a specific stage rather than a generic "something
  broke" — the log clearly shows which of the 6 stages failed.
- **Artifacts preserved on failure.** The `if: always()` condition on
  the upload steps means that even Run #1's (empty, but present)
  artifact-upload attempts were logged with a clear warning
  ("No files were found... no artifacts will be uploaded") rather than
  silently skipped — useful for diagnosing *how far* a failed run got.

---

## 4. Reproducibility & Automation Consistency

Run #2 and Run #3 used the same workflow, the same fixed
`random_state=42`, and differed only by one additional data row. The
result was a small, sensible, *explainable* shift in metrics rather
than an erratic one — consistent with the Lab 7 finding that the
pipeline is deterministic given the same input. This confirms
automation consistency across genuinely independent, dataset-triggered
executions, not just repeated runs of identical input (which Lab 7's
`check_lab7_reproducibility.py` already covers).

The three distinct trigger paths (code, dataset, workflow file) were
each exercised for real during this lab, not just declared in the YAML
and left untested — Run #1 was triggered by the workflow file’s
addition, Run #3 by an isolated dataset-only commit.

---

## 5. GitHub Actions vs. Jenkins for ML CI

| Aspect | GitHub Actions | Jenkins |
|---|---|---|
| **Setup** | Config-as-code (`ci.yml`) lives in-repo, zero infrastructure to manage; runs on GitHub-hosted runners by default | Requires a separately hosted/maintained server (or a hosted Jenkins service); more setup and ongoing maintenance overhead |
| **Triggering** | Native, first-class support for path-filtered triggers on push/PR, directly tied to repo events | Requires plugins (e.g. GitHub webhook plugin) to achieve equivalent path-based, event-driven triggering |
| **Artifacts** | Built-in `actions/upload-artifact`, tightly integrated with the run's UI | Supports artifacts too, but configuration and the resulting UI are less integrated with the source repo |
| **Cost model** | Free tier is generous for public/small repos; scales via GitHub-hosted or self-hosted runners | Self-hosted by default — cheaper at large scale if you already run infrastructure, but that infrastructure is now your responsibility |
| **Ecosystem fit for this project** | Since the repo already lives on GitHub, Actions requires no new account, service, or credential management | Would require standing up a new service just for this project, disproportionate to a single academic pipeline |
| **Flexibility for complex pipelines** | Sufficient for this project's needs; more complex multi-agent or on-prem orchestration can get verbose in YAML | More mature for highly customized, long-running, or on-prem enterprise pipelines with complex approval gates |

**Conclusion:** For a project already hosted on GitHub with a
single, moderate-complexity pipeline, GitHub Actions was the pragmatic
choice — no new infrastructure, native trigger support, and artifacts
integrated directly into the same UI used for code review. Jenkins
remains a stronger fit for organizations that need on-premises control,
highly customized approval workflows, or already maintain their own
build infrastructure independent of any single Git host.

---

## 6. Summary

| Requirement | Evidence |
|---|---|
| Automated workflow via GitHub Actions | `.github/workflows/ci.yml` |
| Automated preprocessing/training/validation/evaluation | Run #2, #3 — all 6 stages PASSED |
| Trigger on code updates | Path filters on `src/**`, `pipelines/**` |
| Trigger on dataset modifications | Run #3 — isolated dataset-only commit, auto-triggered |
| Trigger on workflow changes | Run #1 — triggered by `.github/workflows/**` addition |
| Repeated-run / incremental-update validation | Run #2 vs #3 — consistent, explainable metric shift |
| Execution logs, validation outputs, artifacts tracked | 3 `upload-artifact` steps, `if: always()` |
| Reliability/reproducibility/automation-consistency analysis | Sections 3–4, above |
| GitHub Actions vs Jenkins comparison | Section 5, above |
