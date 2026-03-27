# MLOps 1st Assignment Grader

This repository contains the official automated grader used for the **1st group assignment** in the 2026 MLOps course.

Its purpose is to apply the same evidence collection rules, score calculations, and output structure to every submitted repository, using the expectations defined in the course rubric and in `2026_1st_group_guidelines.pdf`.

The grader is designed to be transparent, reproducible, and defendable. It does not infer undocumented work, reward intent, or assume functionality that cannot be evidenced from the submitted repository state at the submission cutoff.

## What this grader is aligned to

The grader maps directly to the **9 technical dimensions** used in the assignment rubric:

1. Single Entry Point & Modularization
2. Code Quality & Efficiency
3. Documentation & Clarity
4. Testing & Coverage
5. Dependency Management
6. Error Handling & Validation
7. Artifacting & Reproducibility
8. Pipeline Completeness
9. Version Control & Workflow

This is directly aligned with the assignment guidelines, which expect a modular machine learning pipeline, a clear `main.py` orchestrator, reproducible dependency management, robust validation, stored artifacts, tests, documentation, and proper GitHub collaboration practices.

The guidelines also define the expected pipeline logic students were asked to build, including modules such as:

- `data_loader.py`
- `data_cleaner.py`
- `data_validation.py`
- `preprocessing.py`
- `features.py`
- `train.py`
- `evaluation.py`
- `inference.py`
- `main.py`
- `utils.py` as optional support code
- `config.yaml` for centralized parameters and paths

The grader does not require exact filenames in every case, but it does look for strong evidence that the expected engineering structure and workflow are present.

## Fairness and grading principles

The grader follows these principles:

- Every repository is evaluated with the same scoring rules in `config.yaml`
- The repository is graded at the latest commit on the target branch **at or before** the official submission cutoff
- Scores are based on harvested evidence, not on intent, explanations, or work that exists only in other branches or local machines
- If a dimension is not run, its score column is left blank rather than treated as zero
- If an automation step fails, the outputs surface that limitation instead of silently hiding it
- Evidence is collected conservatively to reduce false positives and make the results easier to defend

This is important because the purpose of the grader is not only to assign marks, but to do so in a way that is consistent across groups and easy to audit afterwards.

## Source of truth

The main sources of truth are:

- `2026_1st_group_guidelines.pdf` for the rubric expectations
- `config.yaml` for the scoring weights, thresholds, caps, and penalties
- `grading_utils.py` for evidence collection and score calculation logic
- `grader.py` for execution flow and CSV output generation

If you want to understand **why** a score was assigned, do not rely on one file alone. Read:

1. `raw_scores.csv`
2. `evidence.csv`
3. `feedback.csv`
4. the submitted repository contents at the cutoff commit

## How to run the grader locally

Students may run the grader on their own repository if they want to preview the automated evidence that will be harvested.

### 1. Clone the grader

```bash
git clone https://github.com/2026-IE-MLOps-Course/Grader_1st_assignment.git
cd Grader_1st_assignment
```

### 2. Install the grading environment

Use the environment setup provided in this repository.

Example:

```bash
conda env create -f environment.yml
conda activate grades
```

If you already have the environment, update it instead:

```bash
conda env update -f environment.yml --prune
conda activate grades
```

### 3. Set a GitHub token for version control checks

The Version Control & Workflow dimension uses the GitHub API to inspect Pull Requests and reviews. Without a token, API access may be rate-limited or incomplete.

Linux or macOS:

```bash
export GITHUB_TOKEN="your_token_here"
```

Windows PowerShell:

```powershell
$env:GITHUB_TOKEN="your_token_here"
```

### 4. Run the grader on a local repository

```bash
python grader.py \
  --config config.yaml \
  --output-dir outputs \
  --local-repo-path /full/path/to/your/repo
```

### 5. Run only selected dimensions

```bash
python grader.py \
  --config config.yaml \
  --output-dir outputs \
  --local-repo-path /full/path/to/your/repo \
  --dimensions modularization testing dependencies
```

### 6. Run the full batch from `repos.txt`

```bash
python grader.py \
  --config config.yaml \
  --repos-file repos.txt \
  --output-dir outputs \
  --dimensions all
```

## Start with `raw_scores.csv`

This is the **primary grading table**.

Each row represents one repository. The file contains:

- repository metadata such as `repo_id`, `repo_url`, `branch`, and `cutoff_commit`
- one score column for each rubric dimension
- `total_raw_score`
- optional `error` information when a repo could not be graded normally

### How to read it

- Every automated dimension is scored on a **0 to 10** scale
- `total_raw_score` is the sum of the numeric dimensions that were actually run
- A blank cell means the dimension was not run, or was intentionally left qualitative only
- A zero means the dimension was run and the harvested evidence did not justify points

### Important note on Documentation & Clarity

In the current grading release, **Documentation & Clarity is intentionally not auto-scored**.

The automation checks only whether `README.md` exists. The `documentation_clarity` column in `raw_scores.csv` is left blank on purpose, and the instructor assigns that dimension separately through qualitative review.

This design choice improves fairness because the rubric expects quality of explanation, not only file presence. A script can verify whether a README exists. It cannot fairly capture, on its own, whether the README is genuinely clear, complete, and useful.

## Then read `evidence.csv`

This is the **audit table** behind the scores.

It contains the raw evidence harvested from each repository, such as:

- whether a valid entry point was detected
- how many core pipeline modules were found
- how many pipeline stages were imported by `main.py`
- linting, maintainability, and complexity signals
- whether dependency files exist
- whether pytest ran, how many tests passed, and what coverage was reported
- whether validation guards were detected
- whether artifacts and prediction outputs were found
- whether a qualifying notebook imports code from `src`
- whether Pull Request and review evidence was found through GitHub

This file is the first place to look if you want to understand **why** a score was high, low, or zero.

## Then read `feedback.csv`

This file converts the raw evidence into short human-readable comments.

It is meant to help interpretation, not replace the evidence.

Examples include statements such as:

- the repo has a clear entry point and partial orchestration
- pytest runs successfully with a given pass count and coverage
- dependency evidence is present and reproducibility is supported
- Pull Request evidence is limited or strong

If there is any perceived discrepancy, `evidence.csv` and the repository contents take priority over the feedback wording.

## Exact grading logic by dimension

The calculations below are the grading logic implemented in the current grader design. The final numeric source of truth remains `config.yaml`.

## 1. Single Entry Point & Modularization

### What the grader checks

This dimension is intended to reflect the rubric progression from a weak notebook-based workflow to a fully orchestrated modular pipeline.

The grader checks for:

- a valid runner file such as `main.py`, `src/main.py`, `run.py`, `run_pipeline.py`, or `pipeline.py`
- a real `if __name__ == "__main__":` guard
- a top-level function called from that guard
- recognizable core pipeline modules across the repository
- pipeline stage imports inside `main.py`
- whether many of the detected modules are merely tiny stubs or placeholders

### Core pipeline stages recognized

The modularization scan looks for evidence of these core stages:

- load
- clean
- validate
- preprocess
- features
- train
- evaluate
- infer

### Evidence fields used

- `main_entry_signal`
- `core_module_count`
- `orchestrated_module_count`
- `stub_penalty_flag`

### Score formula

`score = entry points + core module points + orchestration points - stub penalty`

#### Entry point

- valid entry point detected = `+2.0`

#### Core module thresholds

- `1+` core modules = `+1.0`
- `3+` core modules = `+2.0`
- `5+` core modules = `+3.0`

#### Orchestration thresholds

- `2+` imported stages in `main.py` = `+1.0`
- `4+` imported stages in `main.py` = `+2.0`
- `6+` imported stages in `main.py` = `+3.0`
- `7+` imported stages in `main.py` = `+5.0`

#### Stub penalty

- if `stub_penalty_flag = 1`, subtract `1.0`

### Important rule

Validation is treated as a required pipeline stage for this assignment. If the main orchestration does not import a validation stage, the orchestration count is reduced before scoring.

### Why this is defendable

This dimension does not reward file count alone. It checks whether the repository has a runnable entry point and whether the main script genuinely orchestrates the modular pipeline rather than leaving the work scattered across notebooks.

## 2. Code Quality & Efficiency

### What the grader checks

This dimension combines three signals:

- `ruff` issue count
- `pylint` score
- `radon` average cyclomatic complexity

If a `src/` folder exists, the grader targets `src`. Otherwise it evaluates the repository root.

### Evidence fields used

- `ruff_issue_count`
- `pylint_score`
- `radon_average_cc`

### Score formula

This is a deduction-based score that starts at `10.0`.

`score = 10.0 - ruff penalty - pylint penalty - radon penalty`

#### Ruff penalty

`ruff penalty = min(2.0, 0.05 × ruff_issue_count)`

#### Pylint penalty

`pylint penalty = max(0, (8.5 - pylint_score) × 0.3)`

#### Radon penalty

`radon penalty = max(0, (radon_average_cc - 5.0) × 0.2)`

The final score is clamped to the `0 to 10` range.

### Why this is defendable

It balances style compliance, maintainability, and structural complexity. A repository is not rewarded only for formatting if the code is still hard to maintain.

## 3. Documentation & Clarity

### Current scoring policy

This dimension is treated as **manual qualitative review**.

### Automated evidence used

- `readme_present`

### Raw score behavior

- `documentation_clarity` is intentionally left blank in `raw_scores.csv`
- the dimension is not counted inside `total_raw_score`
- the automation only records whether `README.md` exists

### Why this is defendable

The rubric expects:

- a comprehensive README
- instructions that allow someone else to run the code
- docstrings in functions and files
- comments that explain **why**, not only **what**

Those expectations matter, but they are qualitative. Reducing them to a purely mechanical number would create a less fair result than separating the factual automated check from the final instructor judgment.

## 4. Testing & Coverage

### What the grader checks

This dimension combines:

- whether pytest ran successfully
- how many tests passed out of the total detected
- reported coverage
- whether pytest was compromised by import errors or timeout

### Evidence fields used

- `pytest_passed`
- `pytest_pass_count`
- `pytest_fail_count`
- `pytest_error_count`
- `pytest_total_count`
- `pytest_pass_rate`
- `coverage_pct`
- `pytest_import_error`
- `pytest_timeout`

### Score formula

`score = pytest execution contribution + coverage contribution`

#### Execution contribution

- if total tests are known, `4.0 × pytest_pass_rate`
- otherwise, if pytest passed cleanly, `+4.0`

#### Coverage contribution

`coverage contribution = min(6.0, coverage_pct / 14.5)`

This means coverage reaches the full `6.0` contribution at roughly `87%`.

#### Environment cap

If pytest shows import errors or times out, the whole dimension is capped at `6.0`.

### Why this is defendable

This dimension rewards both test reliability and meaningful coverage. It also avoids over-rewarding repositories whose tests cannot run properly in the grading environment.

## 5. Dependency Management

### What the grader checks

The grader records whether the repository contains dependency and configuration files such as:

- `environment.yml` or `environment.yaml`
- `conda.yml` or `conda.yaml`
- `requirements.txt`, `pyproject.toml`, or `Pipfile`
- `config.yaml`, `config.yml`, or `.env`

### Evidence fields used

- `environment_yml_present`
- `conda_yml_present`
- `requirements_txt_present`
- `config_file_present`
- `dependency_versions_declared`

### Score formula

This dimension currently uses a strict binary score:

- if `environment.yml` or `conda.yml` exists, score = `10.0`
- otherwise, score = `0.0`

### Important clarification

The grader does collect some additional dependency evidence, such as whether versions appear to be pinned. However, in the current scoring configuration, the numeric mark is driven by the presence of a reproducible Conda-style environment file.

### Why this is defendable

The assignment guidelines explicitly asked for reproducible dependency management through `environment.yml` and or `conda.yml`. This makes the check strict, easy to audit, and consistent across all groups.

## 6. Error Handling & Validation

### What the grader checks

This dimension looks for explicit validation and defensive programming patterns, especially in files whose names suggest validation or schema logic.

### Evidence fields used

- `validation_function_present`
- `validation_raise_present`
- `validation_none_or_type_guard`
- `validation_empty_guard`
- `validation_required_columns_guard`
- `validation_missing_values_guard`
- `validation_target_guard`
- `validation_dtype_guard`
- `validation_range_or_domain_guard`

### Score formula

The score is additive, with one weighted contribution per detected signal:

- validation function present = `+1.0`
- explicit raise present = `+1.0`
- none or type guard = `+1.0`
- empty input guard = `+1.0`
- required columns guard = `+2.0`
- missing values guard = `+1.0`
- target guard = `+1.5`
- dtype guard = `+1.5`
- range or domain guard = `+1.0`

### Caps

Two fairness caps apply:

- if **no advanced guards** are detected among target, dtype, and range/domain checks, the score is capped at `6.0`
- if the repository is missing the basic foundation of a validation function and explicit `raise`, the score is capped at `6.0`

### Why this is defendable

This prevents superficial validation from being scored as if it were robust production-style checking. Repositories need both a solid foundation and some breadth of meaningful validation logic.

## 7. Artifacting & Reproducibility

### What the grader checks

This dimension looks for evidence, mainly from `main.py`, that the pipeline saves its outputs in a structured and reusable way.

### Evidence fields used

- `processed_data_artifact_present`
- `model_artifact_present`
- `prediction_artifact_present`
- `structured_asset_paths_present`

### Score formula

- processed data artifact present = `+2.5`
- model artifact present = `+4.0`
- prediction artifact present = `+2.5`
- structured asset paths present = `+1.0`

### Caps

- if the model artifact is missing, the whole dimension is capped at `5.0`
- if prediction artifact output is missing, the whole dimension is capped at `7.5`

### Why this is defendable

The repository should not only train a model. It should also leave behind reusable outputs for downstream work. Missing the saved model is especially serious, which is why it triggers a strong cap.

## 8. Pipeline Completeness

### What the grader checks

This dimension checks whether the repository contains at least one valid target notebook that:

- matches the accepted naming logic for experimentation notebooks
- parses as a valid notebook file
- imports modularized code from `src`

### Evidence fields used

- `valid_target_notebooks_with_src_import`
- `target_notebooks_with_src_import_names`

### Score formula

- at least one valid target notebook importing from `src` = `10.0`
- otherwise = `0.0`

### Why this is defendable

This dimension rewards the intended transition from exploration to modular engineering. A notebook counts only if it actually reuses the modularized project code.

## 9. Version Control & Workflow

### What the grader checks

This dimension uses the GitHub API and local git history to look for collaboration evidence up to the submission cutoff.

### Evidence fields used for scoring

- `github_prs_found`
- `github_unique_pr_authors`
- `github_unique_reviewers`

### Additional audit evidence collected

- `github_approvals_found`
- `github_self_merge_ratio`
- `git_commit_count_before_cutoff`
- `git_unique_contributors`
- `git_top_contributor_share`
- `git_contribution_evenness`

### Score tiers

- `10.0` if unique PR authors `>= 5`, PRs `>= 5`, and reviewers `>= 3`
- `8.0` if unique PR authors `>= 4` and PRs `>= 4`
- `6.0` if unique PR authors `>= 3` and PRs `>= 3`
- `4.0` if unique PR authors `>= 2` and PRs `>= 2`
- `2.0` if unique PR authors `>= 1` and PRs `>= 1`
- `0.0` otherwise

### Important clarification

The current numeric score is driven primarily by Pull Request collaboration evidence. Additional signals such as approvals and self-merges are still harvested and reported in `evidence.csv`, even when they do not directly change the current numeric score.

### Why this is defendable

The assignment explicitly required GitHub-based collaboration practices. The score therefore emphasizes evidence of real Pull Request workflow rather than only the existence of commits.

## Why some evidence is collected but not directly scored

Not every harvested field contributes directly to the numeric mark. This is intentional.

Some evidence is stored for:

- auditability
- feedback generation
- manual review support
- future refinement of the grader

This is why `evidence.csv` may contain more fields than those used in the final formula for a given dimension.

## What the grader does not do

To avoid unfair scoring, the grader does **not**:

- reward work that only exists in notebooks when the rubric expects modularized code
- assume that a file is valid just because it has the expected name
- guess runtime success without actual evidence
- award documentation quality based only on file presence
- give credit for branches, commits, or artifacts that are outside the submitted cutoff state

## Final interpretation guidance

Use the outputs in this order:

1. `raw_scores.csv` to see the dimension-level marks
2. `evidence.csv` to inspect the measurable proof behind those marks
3. `feedback.csv` to read the concise interpretation
4. the repository at `cutoff_commit` to verify anything you want to audit directly

If you want to challenge or understand a mark, start from the evidence rather than from a subjective impression. That is the point of making the grader public and the logic explicit.
