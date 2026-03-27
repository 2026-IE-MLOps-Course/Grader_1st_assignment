# Grading run instructions

## Files
- `grader.py`
- `grading_utils.py`
- `repos.txt`

## Recommended local environment
Use a clean disposable Python environment, for example Conda or `venv`

## Install dependencies
```bash
pip install pandas requests pytest pytest-cov ruff pylint radon
```

## GitHub token is required
The grader calls the GitHub API for Pull Request and review evidence. Without a token, rate limits are too low for 16 repos

Linux or macOS:
```bash
export GITHUB_TOKEN="your_token_here"
```

Windows PowerShell:
```powershell
$env:GITHUB_TOKEN="your_token_here"
```

## Optional pytest timeout override
Default timeout is 180 seconds per repo. To change it:

Linux or macOS:
```bash
export GRADER_PYTEST_TIMEOUT="180"
```

Windows PowerShell:
```powershell
$env:GRADER_PYTEST_TIMEOUT="180"
```

## Run the grader
From the folder where the three files live:

```bash
python grader.py \
  --repos-file repos.txt \
  --workdir grading_workspace \
  --output-dir outputs \
  --cutoff "2026-03-09 23:59:59" \
  --timezone "Europe/Madrid" \
  --include-benchmark
```

## Outputs
The script creates:
- `outputs/raw_scores.csv`
- `outputs/evidence.csv`
- `outputs/feedback.csv`

## Important grading rules implemented
- Each repo is frozen at the latest commit on `main` at or before the cutoff
- The grader is mostly static by design
- It executes only the students' own `pytest` suite
- It does not execute the full pipeline
- Notebook checks look only for notebook filenames containing `vExp` or `sandbox`
- Other notebooks are ignored for the notebook requirement
- It does not normalize grades
- It writes raw evidence so you can recalibrate later in Excel

## Pytest safety notes
- Pytest runs with a reduced environment to limit plugin noise and accidental local side effects
- Test execution is still arbitrary student code, so run this in a disposable environment
- If pytest fails because of dependency mismatch or timeout, the grader records that explicitly in `evidence.csv`
- In those cases, coverage should be interpreted cautiously rather than as a clean zero

## How to read the outputs
- `raw_scores.csv`
  - first-pass 0 to 10 proxy score for each of the 9 rubric dimensions
  - one total raw score column
- `evidence.csv`
  - raw measurable checks such as coverage, contributors, PR counts, approvals, docstring ratios, module coverage, and tool outputs
  - use this file for normalization and manual calibration
- `feedback.csv`
  - short first-pass comments per dimension and one overall comment
  - edit these manually after reviewing borderline cases

## Important cautions
- `Version Control & Workflow` is inferred from observable Git and GitHub behavior, not hidden branch settings
- `direct commits to main/dev` cannot be proven perfectly from public clone data alone
- If a local tool is not installed, the script continues and records partial evidence
- Repos with broken tests or unusual layouts still get partial grading instead of stopping the full run
