# IE MLOps Final Group Assignment: Automated Grader

This repository contains the programmatic evaluation suite for the **Final Group Assignment** in the 2026 MLOps course.

The same evidence extraction, thresholds, caps, and scoring logic applies to every repository, including the instructor's benchmark. The grader maps to the 12 production-oriented dimensions defined in `2026_Final_guidelines_v01.pdf`, and carries forward all 9 dimensions from the 1st assignment.

---

## How to use this grader

This is a **diagnostic tool**, not a final grade predictor.

- it shows what is observable and verifiable in your repository at the submission cutoff
- it cannot infer intent, README descriptions, or manual steps
- if something is not implemented in code and committed, it will not count
- the instructor applies cross-dimension weights and manual review (documentation, video) outside this tool

Run it early. Use `evidence.csv` to understand gaps. Iterate.

---

## What this grader checks

**Carried forward from the 1st assignment**

Single entry point and modularization, code quality, documentation (qualitative only), testing, dependency management, error handling, artifacting, pipeline completeness, version control and workflow.

**New for the final assignment**

Configuration and reproducibility, security and secrets, logging and observability, experiment tracking, model registry, API serving, containerization, CI/CD, monitoring, deployment, GitHub workflow discipline, release discipline.

---

## How to run it on your own repository

Run this only against your own repository. Bulk evaluation across other groups is not supported and not intended.

### 1. Setup

```bash
git clone https://github.com/2026-IE-MLOps-Course/Grader_final_assignment.git
cd Grader_final_assignment

conda env create -f environment.yml
conda activate grades
```

### 2. Export your GitHub token

Several dimensions use the GitHub API: workflow discipline, release discipline, and version control. Without a token those dimensions will return incomplete evidence and scores may be under-reported.

```bash
export GITHUB_TOKEN="your_token_here"
```

On Windows PowerShell:

```powershell
$env:GITHUB_TOKEN="your_token_here"
```

A classic token with `repo` scope is sufficient.

### 3. Run the final assignment dimensions

```bash
python grader.py \
  --config config.yaml \
  --output-dir outputs \
  --dimension-group final_assignment \
  --local-repo-path /full/path/to/your/repo
```

To run all dimensions (1st assignment + final) in one pass:

```bash
python grader.py \
  --config config.yaml \
  --output-dir outputs \
  --dimension-group all \
  --local-repo-path /full/path/to/your/repo
```

### 4. Deployment check

The `deployment` dimension makes a live HTTP request to your Render URL. For this to work, add a `deployment_urls.csv` file in the grader directory:

```
repo_id,deployment_url
your-repo-name,https://your-app.onrender.com
```

Then run with:

```bash
python grader.py \
  --config config.yaml \
  --output-dir outputs \
  --dimension-group final_assignment \
  --deployment-urls-file deployment_urls.csv \
  --local-repo-path /full/path/to/your/repo
```

If your app is in cold-start sleep on Render's free tier, wake it manually before running. The grader retries with backoff, but a full cold-start timeout will surface as `deployment_timeout_flag=1` in evidence.

---

## Read the outputs in this order

### 1. `raw_scores.csv`

The main grading table. One row per repository. Each score column is one rubric dimension, `0.0` to `10.0`.

- `documentation_clarity` is intentionally **blank** — graded manually by the instructor
- dimensions not in the selected group are left blank, not zero
- `total_raw_score` is the automated subtotal only; final grades include instructor weights applied separately

### 2. `evidence.csv`

The audit trail behind every score. This is the file to inspect when a score is lower than expected. It contains the raw signals collected from your repository: file presence, AST inspection results, regex matches, GitHub API responses, and live HTTP probe outcomes.

If you disagree with a score, start here.

### 3. `feedback.csv`

Short qualitative comments translating evidence into plain language. Useful for a quick read; the numeric basis always comes from `raw_scores.csv` and `evidence.csv`.

---

## How each dimension is evaluated

All weights, thresholds, and caps live in `config.yaml`. What follows is the plain-language version, with enough detail to understand your result and close gaps.

---

### 1. Configuration and reproducibility

The grader checks whether runtime settings are centralized in `config.yaml`, the environment is reproducible via `conda-lock.yml`, and production code reads config rather than embedding values directly.

Key evidence: `config.yaml` presence and key count, `conda-lock.yml` presence, `main.py` reading config, `.env` exclusion from Git, absence of hardcoded paths and hyperparameters in production code.

Scoring notes: `config.yaml` and `conda-lock.yml` carry the most weight. Hardcoded runtime values generate penalties. Missing `config.yaml` caps the score at `4.0`; missing `conda-lock.yml` caps at `8.0`.

---

### 2. Security and secrets

The grader checks whether secrets stay in `.env`, are excluded from Git and Docker build context, and are absent from tracked source files.

Key evidence: `.gitignore` and `.dockerignore` exclude `.env`, no secret-like literals in committed code, `.env` not tracked by Git.

Scoring notes: `.gitignore` exclusion carries the most weight. A committed `.env` file or secret literals in tracked code both cap the score severely.

---

### 3. Logging and observability

The grader checks whether `print()` is absent from production code, a dedicated logger module exists at `src/logger.py`, and it writes to both console and a local file.

Key evidence: print statement count in production files, `src/logger.py` presence, file handler and stream handler configured, logger actually imported and used in other production modules.

Scoring notes: being print-free, having dual output, and using the logger in other modules are the three heaviest signals. Missing the logger module caps at `4.0`; missing dual output caps at `6.0`; logger not used elsewhere caps at `8.0`.

Up to 3 print statements are tolerated. Zero is the goal, but this threshold accommodates edge cases like startup banners.

---

### 4. Experiment tracking

The grader checks whether W&B is initialized in `main.py` and used to log config, run metadata, evaluation metrics, and a model artifact.

Key evidence: `wandb.init()` in `main.py`, config passed at init, evaluation metrics logged, model artifact logged via `wandb.Artifact`.

Scoring notes: evaluation metrics and model artifact logging carry the most weight. No `wandb.init()` in `main.py` caps at `4.0`; no evaluation metrics caps at `5.0`; no model artifact caps at `8.0`.

---

### 5. Model registry

The grader checks whether inference loads the model from the W&B registry using the `prod` alias, rather than from an unmanaged local file.

Key evidence: `use_artifact()` or equivalent registry-backed loading in serving code, `prod` alias referenced, absence of local-only `.joblib` loading as the sole path.

Scoring notes: registry-backed serving and the `prod` alias together make up the full score. Local-only model loading with no registry path caps the score at `0.0`.

---

### 6. API serving

The grader checks whether FastAPI serves predictions with a Pydantic-enforced request contract, a `/health` endpoint, and a `/predict` endpoint that delegates to the inference layer.

Key evidence: FastAPI app in `src/api.py`, Pydantic model as request contract on `/predict`, `/health` and `/predict` route handlers present, `/predict` calling serving-layer functions rather than raw model calls, Uvicorn configured.

Scoring notes: no FastAPI app caps at `0.0`; no `/predict` endpoint caps at `5.0`; a predict handler that calls `model.predict()` directly instead of delegating to an inference function caps at `7.0`. The check rewards proper separation of concerns: `api.py` should orchestrate, not implement ML logic.

---

### 7. Containerization and portability

The grader checks whether a Dockerfile serves the API, a strict `.dockerignore` excludes development noise, and the install is reproducible via a lock file.

Key evidence: `Dockerfile` present, CMD or ENTRYPOINT starts the API server, `.dockerignore` present and excluding noise categories (tests, notebooks, data, wandb, caches, reports, `.github`), lock file referenced in Dockerfile.

Scoring notes: the serving entrypoint and `.dockerignore` quality carry the most weight. No Dockerfile caps at `0.0`; no serving entrypoint caps at `5.0`; no `.dockerignore` caps at `6.0`.

---

### 8. CI/CD

The grader checks whether `ci.yml` validates Pull Requests and runs at least one quality gate (tests, lint). A release-triggered deploy workflow is rewarded as an additional signal — it is not required for full marks on the CI side.

Key evidence: `ci.yml` present and triggering on `pull_request`, validation step confirmed, deploy workflow (`deploy.yml`, `cd.yml`, `release.yml`, or `publish.yml`) present and triggered by release only.

Scoring notes: CI signals cover the majority of the score. A deploy workflow gated correctly on release adds points. A deploy workflow that also triggers on push generates a penalty. If CI does not trigger on `pull_request` at all, the score is capped at `5.0`.

---

### 9. Monitoring

The grader checks whether there is operational traceability through local logs, API request logging, and W&B inference telemetry — all verified through code and config signals, not live runtime inspection.

Key evidence: log file path configured and wired to the API, per-request trace signal in `api.py` (timestamp, route, or request id), structured logging from the API, W&B telemetry during inference, healthcheck endpoint present.

Scoring notes: each signal contributes roughly equally. The full score requires evidence across all five.

---

### 10. Deployment

The grader makes a live HTTP request to your Render URL and checks whether the app responds correctly to a real inference payload.

Key evidence: public URL provided, service reachable, `/predict` accepts a valid JSON payload, response contains a valid prediction.

Scoring notes: reachability and valid prediction together make up most of the score. If the OpenAPI docs example is malformed but the grader can repair it and still get a valid response, a small penalty applies. The cleanest path is a correct `/docs` example.

---

### 11. GitHub workflow discipline

The grader checks whether Pull Requests are the path to `main`, CI checks pass before merge, and the branch list is clean.

Key evidence: at least one merged PR to `main` before the cutoff, CI check runs present on merged PRs (requires GitHub token), fewer than 6 non-main/non-dev branches remaining.

Scoring notes: having at least one merged PR to `main` carries the most weight. No merged PR to `main` caps the score at `3.0`. CI check evidence and branch hygiene add the remaining points.

---

### 12. Release discipline

The grader checks whether a formal GitHub Release exists, was published before the submission cutoff, and targets `main`.

Key evidence: non-draft release found via GitHub API, `published_at` at or before the cutoff datetime, `target_commitish` equals `main`.

Scoring notes: release found and release targeting `main` split the score equally. A release published after the cutoff does not count. A draft release does not count.

---

## Evaluation principles

- every repository is evaluated with the same config, thresholds, and scoring logic
- scores are based on harvested evidence — missing evidence cannot be inferred from intention
- the grader is deterministic: same repository at the same commit produces the same outputs
- all weights and caps live in `config.yaml` — nothing is hidden

---

## Common reasons for lower-than-expected scores

Use `evidence.csv` to confirm which signal is missing before drawing a conclusion.

**Configuration and reproducibility**
- `conda-lock.yml` not committed: biggest single contributor to this dimension
- config loaded through a custom utility the grader does not recognise: check `main_reads_config_signal` in evidence

**Logging and observability**
- logger present but not imported in other production modules: `log_logger_usage_signal = 0`
- print statements remaining in `src/api.py` or inference files

**Model registry**
- `use_artifact()` present but `prod` alias not referenced: `reg_serving_prod_alias_used = 0`
- model loaded from a local path as the only option: `reg_serving_local_only = 1`

**API serving**
- `/predict` calls `model.predict()` directly instead of delegating to `run_inference()` or equivalent
- Pydantic model defined but not used as the endpoint parameter type annotation

**CI/CD**
- `deploy.yml` triggers on `push` in addition to `release`: penalty applied
- workflow file not named `deploy.yml`, `cd.yml`, `release.yml`, or `publish.yml`: check `cd_workflow_file` in evidence to confirm it was detected

**Deployment**
- app in cold-start sleep: wake it before running the grader
- OpenAPI docs example has wrong field types or missing required fields: grader attempts repair but applies a small penalty; fix the source

**Release discipline**
- GitHub Release created as a draft: excluded
- release `target_commitish` is a commit SHA rather than the branch name `main`: `release_targets_main = 0`
- release published after the submission cutoff: does not count
