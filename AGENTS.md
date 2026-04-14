# Repository Guidelines

## Project Purpose
This repository contains a deterministic Python grader for the MLOps course assignments. The current baseline supports the original 9 grading dimensions and writes 3 audit outputs: `raw_scores.csv`, `evidence.csv`, and `feedback.csv`. Current work is a controlled extension for the final assignment rubric, not a rewrite.

## Core Working Rules
- Extend the grader one new dimension at a time
- Preserve backward compatibility with the original 9 dimensions
- Keep scoring deterministic, explicit, and audit-friendly
- Do not introduce hidden heuristics or subjective logic
- Do not redesign the architecture unless strictly necessary
- Keep all new evidence visible in `evidence.csv`
- Keep all new scoring traceable to `config.yaml`

## Project Structure
- `grader.py`: command-line entry point and CSV writing
- `grading_utils.py`: evidence collection, scoring, and feedback logic
- `config.yaml`: scoring thresholds, weights, caps, and penalties
- `repos.txt`: repository list for batch grading
- `outputs/`: generated CSV outputs
- `grading_workspace/clones/`: temporary cloned repositories
- Rubric PDFs in project context define grading expectations and must guide implementation

## Editing Expectations
When implementing a new dimension, make minimal surgical edits only in the files required for that dimension, usually `grader.py`, `grading_utils.py`, and `config.yaml`. Do not break existing CLI behavior, argparse dimension selection, or output schemas unless explicitly requested.

## Development Commands
- `python grader.py --help`
- `python grader.py --config config.yaml --dimensions <dimension> --local-repo-path /path/to/repo --output-dir outputs`
- `python grader.py --config config.yaml --dimensions <dimension> --repos-file repos.txt --output-dir outputs`

## Coding Style
Use small focused functions, `snake_case`, `pathlib.Path`, and explicit readable logic. Prefer stable file checks, AST inspection, regex, config parsing, subprocess output, Git metadata, and HTTP checks only when the rubric truly requires them.