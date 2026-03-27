# IE MLOps 1st Group Assignment: Automated Grader

This repository contains the programatic evaluation suite part for the **1st Group Assignment** in the 2026 MLOps course.

It applies the same evidence extraction, thresholds, caps, and score calculations to every repository (Including the professor's one). The grader maps directly to the 9 technical dimensions defined in `2026_1st_group_guidelines.pdf`.

This README reflects the **release configuration used for grading**, including the qualitative-only handling of **Documentation & Clarity**.

---

## Alignment with the assignment guidelines

The grader is designed to evaluate what is **implemented, committed, and verifiable** in the repository at the submission cutoff.

It checks whether your repository evidences the engineering expectations stated in the assignment guidelines, including:

- a clear `main.py` entry point
- modular pipeline logic instead of notebook-only workflows
- reproducible dependency management
- test execution and measurable coverage
- explicit validation and fail-fast checks
- saved pipeline artifacts
- at least one valid notebook that imports modularized code
- collaborative GitHub workflow through Pull Requests and reviews

The grader does **not** depend on one exact filename per module. It looks for logical implementation patterns and repository evidence.

Typical modules or responsibilities expected by the rubric include:

- data loading
- preprocessing or cleaning
- validation
- feature engineering
- training
- evaluation
- inference
- orchestration from `main.py`

---

## Evaluation principles

- every repository is evaluated with the same scoring rules
- scores are based on harvested repository evidence and execution outputs
- missing evidence cannot be inferred from intention
- failing automation steps surface as missing evidence or capped scores where applicable
- the final numeric logic is driven by `config.yaml`

---

## How to run the grader locally

You are encouraged to run the grader on your own repository.

### 1. Setup

```bash
git clone https://github.com/2026-IE-MLOps-Course/Grader_1st_assignment.git
cd Grader_1st_assignment

conda env create -f environment.yml
conda activate grades
```

### 2. GitHub token

The **Version Control & Workflow** dimension uses the GitHub API to inspect Pull Requests and reviews. Export a GitHub token before running the grader.

```bash
export GITHUB_TOKEN="your_token_here"
```

On Windows PowerShell:

```powershell
$env:GITHUB_TOKEN="your_token_here"
```

### 3. Run on your local repository

```bash
python grader.py \
  --config config.yaml \
  --output-dir outputs \
  --local-repo-path /full/path/to/your/repo
```

---

## Read the outputs in this order

### 1. `raw_scores.csv`

This is the main grading table.

Each row is one repository. Each score column corresponds to one rubric dimension.

Important points:

- each automated dimension is scored from `0.0` to `10.0`
- `documentation_clarity` is intentionally left **blank**
- `Documentation & Clarity` is graded manually by the instructor
- `total_raw_score` sums only the **automated** dimensions and therefore **excludes** `documentation_clarity`

This is the file to inspect first if you want to know your numeric baseline.

### 2. `evidence.csv`

This is the audit trail behind the scores.

It contains the raw signals harvested from your repository, such as:

- `main_entry_signal`
- `core_module_count`
- `orchestrated_module_count`
- `ruff_issue_count`
- `pylint_score`
- `radon_average_cc`
- `pytest_pass_rate`
- `coverage_pct`
- `github_prs_found`
- `github_unique_pr_authors`
- `github_unique_reviewers`

If you want to understand **why** a score was assigned, inspect this file second.

### 3. `feedback.csv`

This file translates the evidence into short qualitative comments.

It is meant to help interpretation. The numeric basis still comes from `raw_scores.csv` and `evidence.csv`.

---

## Exact grading logic and calculations

The formulas below describe the released scoring logic.

## 1. Single Entry Point & Modularization

**Goal**

Check whether the notebook workflow has been converted into a modular pipeline orchestrated from `main.py`.

**Score formula**

```text
Score = Entry Signal (2.0)
      + Core Module Points
      + Orchestration Points
      - Stub Penalty (if triggered)
```

**Core Module Points**

- `>= 5` modules  → `3.0`
- `3 to 4` modules → `2.0`
- `1 to 2` modules → `1.0`
- otherwise → `0.0`

**Orchestration Points**

- `>= 7` orchestrated modules → `5.0`
- `6` orchestrated modules → `3.0`
- `4 to 5` orchestrated modules → `2.0`
- `2 to 3` orchestrated modules → `1.0`
- otherwise → `0.0`

**Penalty**

- stub or dummy implementation evidence → `-1.0`

**Maximum score**

- `10.0`

---

## 2. Code Quality & Efficiency

**Goal**

Check style quality, maintainability, and complexity.

**Score formula**

```text
Score = 10.0
      - min(2.0, Ruff Issues × 0.05)
      - max(0, (8.5 - Pylint Score) × 0.3)
      - max(0, (Radon Average CC - 5.0) × 0.2)
```

Where:

- `Ruff Issues` = total issues reported by Ruff
- `Pylint Score` target = `8.5`
- `Radon Average CC` baseline = `5.0`
- Ruff penalty is capped at `2.0`

**Maximum score**

- `10.0`

---

## 3. Documentation & Clarity

**Released grading policy**

This dimension is handled as **qualitative-only**.

Automated evidence checks only whether `README.md` exists.

**Scoring rule**

```text
raw_scores.csv -> documentation_clarity = blank
```

The final score for this dimension is entered manually by the instructor after qualitative review.

---

## 4. Testing & Coverage

**Goal**

Check whether the project includes runnable tests and measurable coverage.

**Score formula**

```text
Score = (Pytest Pass Rate × 4.0)
      + min(6.0, Coverage % / 14.5)
```

Notes:

- a `100%` pytest pass rate yields `4.0`
- coverage is capped at `6.0`
- approximately `87%` coverage reaches the coverage cap

**Environment cap**

If pytest fails due to import or environment issues such as `ModuleNotFoundError`, or if the test run times out, then:

```text
Score = min(Score, 6.0)
```

**Maximum score**

- `10.0`

---

## 5. Dependency Management

**Goal**

Check whether the repository defines a reproducible environment.

**Score formula**

```text
Score = 10.0  if environment.yml exists OR conda.yml exists
Score = 0.0   otherwise
```

This is a strict binary check.

---

## 6. Error Handling & Validation

**Goal**

Check whether the pipeline includes explicit validation and fail-fast safeguards.

**Signals and weights**

- `validation_function_present` → `1.0`
- `validation_raise_present` → `1.0`
- `validation_none_or_type_guard` → `1.0`
- `validation_empty_guard` → `1.0`
- `validation_required_columns_guard` → `2.0`
- `validation_missing_values_guard` → `1.0`
- `validation_target_guard` → `1.5`
- `validation_dtype_guard` → `1.5`
- `validation_range_or_domain_guard` → `1.0`

**Base score formula**

```text
Score = sum(weights of detected validation signals)
```

**Caps**

If no advanced validation is detected across target, dtype, or domain checks:

```text
Score = min(Score, 6.0)
```

If foundation fail-fast evidence is incomplete because either a validation function or explicit raising logic is missing:

```text
Score = min(Score, 6.0)
```

**Maximum score**

- `10.0`

---

## 7. Artifacting & Reproducibility

**Goal**

Check whether the pipeline stores the outputs needed for reproducible downstream use.

**Score formula**

```text
Score = Processed Data Artifact (2.5)
      + Model Artifact (4.0)
      + Prediction Artifact (2.5)
      + Structured Asset Paths (1.0)
```

**Caps**

If the model artifact is missing:

```text
Score = min(Score, 5.0)
```

If the prediction artifact is missing:

```text
Score = min(Score, 7.5)
```

**Maximum score**

- `10.0`

---

## 8. Pipeline Completeness

**Goal**

Check whether at least one valid notebook still exists for experimentation while importing modularized code from `src`.

**Score formula**

```text
Score = 10.0  if at least one valid target notebook imports from src
Score = 0.0   otherwise
```

This is a strict binary check.

---

## 9. Version Control & Workflow

**Goal**

Check whether collaborative GitHub workflow is evidenced through Pull Requests and reviews.

The scorer uses these harvested signals:

- `github_prs_found`
- `github_unique_pr_authors`
- `github_unique_reviewers`

**Tier logic**

```text
10.0 if PR authors >= 5 AND PRs >= 5 AND reviewers >= 3
 8.0 if PR authors >= 4 AND PRs >= 4
 6.0 if PR authors >= 3 AND PRs >= 3
 4.0 if PR authors >= 2 AND PRs >= 2
 2.0 if PR authors >= 1 AND PRs >= 1
 0.0 otherwise
```

If no GitHub token is provided, API evidence may be incomplete.

---

## Final interpretation rules

- `raw_scores.csv` gives the numeric baseline
- `evidence.csv` explains the basis of each score
- `feedback.csv` summarizes the evidence in text
- `config.yaml` contains the weights, thresholds, and caps
- `Documentation & Clarity` is intentionally left for manual grading
