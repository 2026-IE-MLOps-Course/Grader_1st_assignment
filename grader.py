from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from grading_utils import (
    DIMENSION_TO_SCORE_COLUMN,
    RepoSpec,
    checkout_commit,
    collect_evidence,
    compute_proxy_scores,
    load_grading_config,
    ensure_clone,
    git_commit_metadata,
    make_dimension_comments,
    parse_repos_file,
    resolve_deployment_urls_file,
    slim_evidence_for_selected_dimensions,
    resolve_cutoff_commit,
    round2,
    write_csv,
)


DEFAULT_CUTOFF = None
DEFAULT_TIMEZONE = None
DEFAULT_BENCHMARK_URL = "https://github.com/2026-IE-MLOps-Course/2-mlops-modularized-repo"
DEFAULT_BENCHMARK_COMMIT = "e30e9a963bab3feb2ae3d3f3cfa53aba42ad7d48"

DIMENSION_CHOICES = [
    "all",
    "modularization",
    "code_quality",
    "documentation",
    "testing",
    "dependencies",
    "config_reproducibility",
    "security_secrets",
    "error_handling",
    "artifacting",
    "pipeline",
    "version_control",
    "logging_observability",
    "experiment_tracking",
    "model_registry",
    "api_serving",
    "containerization",
    "ci_cd",
    "monitoring",
    "deployment",
    "github_workflow_discipline",
]

ALL_DIMENSIONS = [d for d in DIMENSION_CHOICES if d != "all"]


def documentation_score_is_qualitative_only(config: dict[str, Any]) -> bool:
    return bool(config.get("scoring", {}).get("documentation", {}).get("qualitative_only", True))


def normalize_local_repo_id(repo_dir: Path) -> str:
    repo_id = repo_dir.name
    parent_parts = repo_dir.parent.parts
    if len(parent_parts) < 2 or parent_parts[-2:] != ("grading_workspace", "clones"):
        return repo_id

    parts = repo_id.split("_")
    if len(parts) < 3:
        return repo_id

    for split_index in range(1, len(parts)):
        left = parts[:split_index]
        right = parts[split_index:]
        if left and left == right:
            return "_".join(left)

    if len(parts) >= 5:
        for split_index in range(2, len(parts) - 1):
            left = parts[:split_index]
            right = parts[split_index:]
            if left[1:] == right:
                return "_".join(left)

    return repo_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade public student repos for the 1st MLOps assignment")
    parser.add_argument("--repos-file", default="repos.txt",
                        help="Path to repos.txt")
    parser.add_argument("--workdir", default="grading_workspace",
                        help="Working directory for cloned repos")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for CSV outputs")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                        help="Submission cutoff in local time: YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE,
                        help="IANA timezone for the cutoff")
    parser.add_argument("--include-benchmark", action="store_true",
                        help="Also grade the instructor benchmark repo")
    parser.add_argument("--benchmark-url", default=DEFAULT_BENCHMARK_URL,
                        help="Instructor benchmark repo URL")
    parser.add_argument("--benchmark-branch", default="main",
                        help="Instructor benchmark repo branch")
    parser.add_argument(
        "--benchmark-commit", default=DEFAULT_BENCHMARK_COMMIT, help="Fixed benchmark commit")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["all"],
        choices=DIMENSION_CHOICES,
        help="Run all checks or only selected rubric dimensions",
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to grader scoring config YAML")
    parser.add_argument(
        "--deployment-urls-file",
        default=None,
        help="CSV mapping repo_id to public deployment URLs",
    )
    parser.add_argument(
        "--local-repo-path",
        default=None,
        help="Run grader on one already-cloned local repo path and skip repos.txt",
    )
    return parser.parse_args()

# def validate_runtime_requirements() -> None:
#     token = os.getenv("GITHUB_TOKEN", "").strip()
#     if not token:
#         raise SystemExit(
#             "GITHUB_TOKEN is required for reliable Pull Request and review evidence across public repos. "
#             "Set it before running the grader"
#         )


def normalize_dimensions(raw_dimensions: list[str]) -> set[str]:
    if "all" in raw_dimensions:
        return set(ALL_DIMENSIONS)
    return set(raw_dimensions)


def prepare_repo_specs(args: argparse.Namespace) -> list[RepoSpec]:
    repo_specs = parse_repos_file(Path(args.repos_file))
    if args.include_benchmark:
        repo_specs.append(
            RepoSpec(
                repo_id="benchmark_repo",
                repo_url=args.benchmark_url,
                branch=args.benchmark_branch,
                fixed_commit=args.benchmark_commit,
                is_benchmark=True,
            )
        )
    return repo_specs


def prepare_local_repo_spec(local_repo_path: str) -> tuple[RepoSpec, Path]:
    repo_dir = Path(local_repo_path).resolve()
    if not repo_dir.exists() or not repo_dir.is_dir():
        raise SystemExit(
            f"Local repo path does not exist or is not a directory: {repo_dir}")

    repo_id = normalize_local_repo_id(repo_dir)
    repo = RepoSpec(
        repo_id=repo_id,
        repo_url=f"local://{repo_id}",
        branch="main",
        fixed_commit=None,
        is_benchmark=False,
    )
    return repo, repo_dir


def empty_score_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for score_col in DIMENSION_TO_SCORE_COLUMN.values():
        payload[score_col] = ""
    payload["total_raw_score"] = ""
    return payload


def grade_single_repo(
    repo: RepoSpec,
    args: argparse.Namespace,
    clones_dir: Path,
    selected_dimensions: set[str],
    config: dict[str, Any],
    deployment_urls_file: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo_dir = ensure_clone(repo, clones_dir)
    effective_cutoff = args.cutoff
    effective_timezone = args.timezone

    commit = repo.fixed_commit or resolve_cutoff_commit(
        repo_dir, repo.branch, effective_cutoff, effective_timezone)
    checkout_commit(repo_dir, commit)

    metadata = git_commit_metadata(repo_dir, commit)
    evidence = collect_evidence(
        repo_dir=repo_dir,
        repo=repo,
        cutoff_str=effective_cutoff,
        timezone_name=effective_timezone,
        selected_dimensions=selected_dimensions,
        deployment_urls_file=deployment_urls_file,
    )
    evidence.update(metadata)
    evidence.update(
        {
            "repo_id": repo.repo_id,
            "repo_url": repo.repo_url,
            "branch": repo.branch,
            "cutoff_commit": commit,
            "is_benchmark": int(repo.is_benchmark),
            "dimensions_run": ",".join(sorted(selected_dimensions)),
        }
    )
    evidence = slim_evidence_for_selected_dimensions(
        evidence, selected_dimensions, config)

    scores = compute_proxy_scores(
        evidence=evidence, selected_dimensions=selected_dimensions, config=config)
    comments = make_dimension_comments(
        evidence=evidence,
        scores=scores,
        selected_dimensions=selected_dimensions,
        repo_dir=repo_dir,
    )

    score_row: dict[str, Any] = {
        "repo_id": repo.repo_id,
        "repo_url": repo.repo_url,
        "branch": repo.branch,
        "cutoff_commit": commit,
        "is_benchmark": int(repo.is_benchmark),
        "dimensions_run": ",".join(sorted(selected_dimensions)),
    }

    total = 0.0
    ran_any = False
    docs_qualitative_only = documentation_score_is_qualitative_only(config)
    for dimension, score_col in DIMENSION_TO_SCORE_COLUMN.items():
        if dimension in selected_dimensions:
            if dimension == "documentation" and docs_qualitative_only:
                score_row[score_col] = ""
                continue
            value = round2(scores.get(score_col, 0.0))
            score_row[score_col] = value
            total += value
            ran_any = True
        else:
            score_row[score_col] = ""

    score_row["total_raw_score"] = round2(total) if ran_any else ""

    feedback_row: dict[str, Any] = {
        "repo_id": repo.repo_id,
        "repo_url": repo.repo_url,
        "branch": repo.branch,
        "cutoff_commit": commit,
        "dimensions_run": ",".join(sorted(selected_dimensions)),
        **comments,
    }

    return score_row, evidence, feedback_row


def error_rows(
    repo: RepoSpec,
    exc: Exception,
    selected_dimensions: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    message = f"{type(exc).__name__}: {exc}"

    score_row = {
        "repo_id": repo.repo_id,
        "repo_url": repo.repo_url,
        "branch": repo.branch,
        "cutoff_commit": "",
        "is_benchmark": int(repo.is_benchmark),
        "dimensions_run": ",".join(sorted(selected_dimensions)),
        **empty_score_payload(),
        "error": message,
    }

    evidence_row = {
        "repo_id": repo.repo_id,
        "repo_url": repo.repo_url,
        "branch": repo.branch,
        "cutoff_commit": "",
        "dimensions_run": ",".join(sorted(selected_dimensions)),
        "error": message,
        "traceback": traceback.format_exc(limit=3),
    }

    feedback_row = {
        "repo_id": repo.repo_id,
        "repo_url": repo.repo_url,
        "branch": repo.branch,
        "cutoff_commit": "",
        "dimensions_run": ",".join(sorted(selected_dimensions)),
        "overall_comment": "The repo could not be fully graded because one or more automation steps failed",
        "error": message,
    }

    return score_row, evidence_row, feedback_row


def main() -> None:
    # validate_runtime_requirements()
    args = parse_args()
    selected_dimensions = normalize_dimensions(args.dimensions)
    config = load_grading_config(Path(args.config))

    submission_cfg = config.get(
        "submission", {}) if isinstance(config, dict) else {}

    effective_cutoff = args.cutoff or submission_cfg.get("cutoff_datetime")
    effective_timezone = args.timezone or submission_cfg.get("timezone")

    if not effective_cutoff:
        raise SystemExit(
            "Missing submission.cutoff_datetime in config.yaml and no --cutoff provided")

    if not effective_timezone:
        raise SystemExit(
            "Missing submission.timezone in config.yaml and no --timezone provided")

    args.cutoff = effective_cutoff
    args.timezone = effective_timezone
    deployment_urls_file = resolve_deployment_urls_file(args.deployment_urls_file)

    clones_dir = Path(args.workdir) / "clones"
    output_dir = Path(args.output_dir)
    clones_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_score_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []

    print(f"Selected dimensions: {', '.join(sorted(selected_dimensions))}")

    if args.local_repo_path:
        repo, repo_dir = prepare_local_repo_spec(args.local_repo_path)
        repo_items = [(repo, repo_dir)]
    else:
        repo_specs = prepare_repo_specs(args)
        repo_items = [(repo, None) for repo in repo_specs]

    for repo, local_repo_dir in repo_items:
        print(f"Grading {repo.repo_id} -> {repo.repo_url}")
        try:
            if local_repo_dir is not None:
                commit = resolve_cutoff_commit(
                    repo_dir=local_repo_dir,
                    branch=repo.branch,
                    cutoff_local=args.cutoff,
                    timezone_name=args.timezone,
                )
                checkout_commit(local_repo_dir, commit)

                metadata = git_commit_metadata(local_repo_dir, commit)
                evidence = collect_evidence(
                    repo_dir=local_repo_dir,
                    repo=repo,
                    cutoff_str=args.cutoff,
                    timezone_name=args.timezone,
                    selected_dimensions=selected_dimensions,
                    deployment_urls_file=deployment_urls_file,
                )
                evidence.update(metadata)
                evidence.update(
                    {
                        "repo_id": repo.repo_id,
                        "repo_url": repo.repo_url,
                        "branch": repo.branch,
                        "cutoff_commit": commit,
                        "is_benchmark": int(repo.is_benchmark),
                        "dimensions_run": ",".join(sorted(selected_dimensions)),
                    }
                )
                evidence = slim_evidence_for_selected_dimensions(
                    evidence, selected_dimensions, config)
                scores = compute_proxy_scores(
                    evidence=evidence,
                    selected_dimensions=selected_dimensions,
                    config=config,
                )
                comments = make_dimension_comments(
                    evidence=evidence,
                    scores=scores,
                    selected_dimensions=selected_dimensions,
                    repo_dir=local_repo_dir,
                )

                score_row = {
                    "repo_id": repo.repo_id,
                    "repo_url": repo.repo_url,
                    "branch": repo.branch,
                    "cutoff_commit": commit,
                    "is_benchmark": int(repo.is_benchmark),
                    "dimensions_run": ",".join(sorted(selected_dimensions)),
                }
                total = 0.0
                ran_any = False
                docs_qualitative_only = documentation_score_is_qualitative_only(config)
                for dimension, score_col in DIMENSION_TO_SCORE_COLUMN.items():
                    if dimension in selected_dimensions:
                        if dimension == "documentation" and docs_qualitative_only:
                            score_row[score_col] = ""
                            continue
                        value = round2(scores.get(score_col, 0.0))
                        score_row[score_col] = value
                        total += value
                        ran_any = True
                    else:
                        score_row[score_col] = ""

                score_row["total_raw_score"] = round2(total) if ran_any else ""

                evidence_row = {
                    "repo_id": repo.repo_id,
                    "repo_url": repo.repo_url,
                    "branch": repo.branch,
                    "cutoff_commit": commit,
                    **evidence,
                }
                feedback_row = {
                    "repo_id": repo.repo_id,
                    "repo_url": repo.repo_url,
                    "branch": repo.branch,
                    "cutoff_commit": commit,
                    "dimensions_run": ",".join(sorted(selected_dimensions)),
                    **comments,
                }
            else:
                score_row, evidence_row, feedback_row = grade_single_repo(
                    repo=repo,
                    args=args,
                    clones_dir=clones_dir,
                    selected_dimensions=selected_dimensions,
                    config=config,
                    deployment_urls_file=deployment_urls_file,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to grade {repo.repo_id}: {exc}")
            score_row, evidence_row, feedback_row = error_rows(
                repo=repo,
                exc=exc,
                selected_dimensions=selected_dimensions,
            )

        raw_score_rows.append(score_row)
        evidence_rows.append(evidence_row)
        feedback_rows.append(feedback_row)

    raw_scores_path = output_dir / "raw_scores.csv"
    evidence_path = output_dir / "evidence.csv"
    feedback_path = output_dir / "feedback.csv"

    write_csv(raw_score_rows, raw_scores_path)
    write_csv(evidence_rows, evidence_path)
    write_csv(feedback_rows, feedback_path)

    print("Saved outputs:")
    print(f"- {raw_scores_path}")
    print(f"- {evidence_path}")
    print(f"- {feedback_path}")

    try:
        df = pd.read_csv(raw_scores_path)
        print("\nRaw score preview:")
        print(df.head(10).to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        print(f"Could not preview raw_scores.csv: {exc}")


if __name__ == "__main__":
    main()
