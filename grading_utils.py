from __future__ import annotations

import ast
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
import yaml


RUBRIC_DIMENSIONS = [
    "single_entry_modularization",
    "code_quality_efficiency",
    "documentation_clarity",
    "testing_coverage",
    "dependency_management",
    "error_handling_validation",
    "artifacting_reproducibility",
    "pipeline_completeness",
    "version_control_workflow",
]

DIMENSION_TO_SCORE_COLUMN = {
    "modularization": "single_entry_modularization",
    "code_quality": "code_quality_efficiency",
    "documentation": "documentation_clarity",
    "testing": "testing_coverage",
    "dependencies": "dependency_management",
    "config_reproducibility": "config_reproducibility_score",
    "security_secrets": "security_secrets_score",
    "logging_observability": "logging_observability_score",
    "experiment_tracking": "experiment_tracking_score",
    "model_registry": "model_registry_score",
    "api_serving": "api_serving_score",
    "error_handling": "error_handling_validation",
    "artifacting": "artifacting_reproducibility",
    "pipeline": "pipeline_completeness",
    "version_control": "version_control_workflow",
}

EVIDENCE_FIELDS_BY_DIMENSION = {
    "modularization": [
        "main_entry_signal",
        "core_module_count",
        "orchestrated_module_count",
        "stub_penalty_flag",
    ],
    "code_quality": [
        "ruff_issue_count",
        "pylint_score",
        "radon_average_cc",
    ],
    "documentation": [
        "readme_present",
    ],
    "testing": [
        "pytest_passed",
        "pytest_pass_count",
        "pytest_fail_count",
        "pytest_error_count",
        "pytest_total_count",
        "pytest_pass_rate",
        "coverage_pct",
        "pytest_import_error",
        "pytest_timeout",
    ],
    "dependencies": [
        "environment_yml_present",
        "conda_yml_present",
    ],
    "config_reproducibility": [
        "config_yaml_present",
        "config_yaml_has_keys",
        "runtime_config_keys_found",
        "env_contract_signal",
        "gitignore_excludes_env",
        "environment_yml_present",
        "conda_lock_yml_present",
        "main_reads_config_signal",
        "dotenv_usage_signal",
        "code_hardcoded_path_hits",
        "code_hardcoded_hyperparam_hits",
        "secret_like_literal_hits",
        "config_reproducibility_cap_reason",
    ],
    "security_secrets": [
        "sec_gitignore_excludes_env",
        "sec_dockerignore_present",
        "sec_dockerignore_excludes_env",
        "sec_env_file_present",
        "sec_env_file_tracked_by_git",
        "sec_env_example_present",
        "sec_dotenv_usage_signal",
        "sec_secret_literal_hits",
        "sec_secret_literal_files",
        "sec_tracked_env_like_files",
        "security_secrets_cap_reason",
    ],
    "logging_observability": [
        "log_print_statement_hits",
        "log_print_free",
        "log_print_hit_files",
        "log_logger_module_present",
        "log_logger_module_path",
        "log_logger_module_name",
        "log_logger_module_fallback_used",
        "log_file_handler_present",
        "log_stream_handler_present",
        "log_dual_output_signal",
        "log_logfile_path_present",
        "log_logger_usage_signal",
        "log_logger_usage_count",
        "logging_observability_cap_reason",
    ],
    "experiment_tracking": [
        "wandb_import_present",
        "wandb_init_in_main",
        "wandb_config_logged",
        "wandb_run_metadata_logged",
        "wandb_eval_metrics_logged",
        "wandb_rich_eval_tracking_logged",
        "wandb_model_artifact_logged",
        "wandb_cap_reason",
    ],
    "model_registry": [
        "reg_serving_path_registry_backed",
        "reg_serving_prod_alias_used",
        "reg_production_registry_selected",
        "reg_serving_local_fallback_present",
        "reg_serving_local_only",
        "reg_model_registry_cap_reason",
    ],
    "api_serving": [
        "api_fastapi_app_present",
        "api_pydantic_contract_present",
        "api_health_endpoint_present",
        "api_predict_endpoint_present",
        "api_uvicorn_serving_present",
        "api_predict_calls_inference_logic",
        "api_serving_cap_reason",
    ],
    "error_handling": [
        "validation_function_present",
        "validation_raise_present",
        "validation_none_or_type_guard",
        "validation_empty_guard",
        "validation_required_columns_guard",
        "validation_missing_values_guard",
        "validation_target_guard",
        "validation_dtype_guard",
        "validation_range_or_domain_guard",
    ],
    "artifacting": [
        "processed_data_artifact_present",
        "model_artifact_present",
        "prediction_artifact_present",
        "structured_asset_paths_present",
    ],
    "pipeline": [
        "all_notebooks_found",
        "target_notebooks_found",
        "valid_target_notebooks",
        "valid_target_notebooks_with_src_import",
        "target_notebook_names",
        "target_notebooks_with_src_import_names",
        "notebook_target_rule",
    ],
    "version_control": [
        "git_unique_contributors",
        "github_prs_found",
        "github_unique_pr_authors",
        "github_unique_reviewers",
        "github_api_authenticated",
    ],
}

EVIDENCE_METADATA_FIELDS = [
    "repo_id",
    "repo_url",
    "branch",
    "cutoff_commit",
    "dimensions_run",
    "is_benchmark",
    "error",
    "traceback",
]

CORE_MODULE_MAP = {
    "load": ["load", "loader", "ingest", "read_data"],
    "clean": ["clean", "cleaner"],
    "validate": ["validate", "validation", "schema"],
    "preprocess": ["preprocess", "preprocessing", "transform", "encode", "scale"],
    "features": ["feature", "features"],
    "train": ["train", "trainer", "fit_model"],
    "evaluate": ["evaluate", "evaluation", "metrics"],
    "infer": ["infer", "inference", "predict"],
    "main": ["main"],
}

BUSINESS_KEYWORDS = {
    "client": ["client", "industry"],
    "business_unit": ["business unit", "department"],
    "maturity": ["maturity", "tools", "processes", "people", "strategy"],
    "goal": ["goal", "objective", "kpi", "baseline"],
    "problem": ["problem", "pain point"],
    "solution": ["solution", "functionalit"],
    "scalability": ["scalability", "scale", "other use cases"],
    "benefit": ["benefit", "competitiveness", "opportunit"],
    "cost": ["cost", "talent", "infrastructure", "licenses", "time"],
    "risk": ["risk", "challenge", "mitigation", "security"],
}

RUN_SECTION_KEYWORDS = ["install", "setup", "run", "pytest", "test", "usage"]
NOTEBOOK_EXCLUDE_PARTS = {".ipynb_checkpoints"}
NOTEBOOK_TARGET_TOKENS = ("exp", "experiment", "sandbox", "sanbox", "sand", "model_main", "modular")
NOTEBOOK_EXCLUDE_NAME_TOKENS = ("legacy",)
PYTEST_TIMEOUT_SECONDS = int(os.getenv("GRADER_PYTEST_TIMEOUT", "180"))


def default_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")


def load_grading_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or default_config_path()
    if not path.exists():
        raise GraderError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise GraderError(f"Invalid YAML config: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise GraderError(f"Config file must contain a YAML mapping at the top level: {path}")

    return payload


def cfg_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@dataclass
class RepoSpec:
    repo_id: str
    repo_url: str
    branch: str = "main"
    fixed_commit: str | None = None
    is_benchmark: bool = False


class GraderError(Exception):
    """Raised when a critical grading step fails."""


def _is_selected(selected_dimensions: set[str], dimension: str) -> bool:
    return dimension in selected_dimensions


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def round2(value: Any) -> float:
    return round(safe_float(value), 2)


def slugify_repo_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def canonical_repo_url(repo_url: str) -> str:
    url = repo_url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


def parse_repos_file(repos_file: Path) -> list[RepoSpec]:
    specs: list[RepoSpec] = []
    seen: set[str] = set()

    with repos_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 1:
                repo_url = parts[0]
                repo_id = default_repo_id_from_url(repo_url, line_number)
                branch = "main"
                fixed_commit = None
            elif len(parts) == 2:
                repo_id, repo_url = parts
                branch = "main"
                fixed_commit = None
            elif len(parts) == 3:
                repo_id, repo_url, branch = parts
                fixed_commit = None
            else:
                repo_id, repo_url, branch, fixed_commit = parts[:4]

            repo_url = canonical_repo_url(repo_url)
            if repo_url in seen:
                continue
            seen.add(repo_url)

            specs.append(
                RepoSpec(
                    repo_id=repo_id,
                    repo_url=repo_url,
                    branch=branch or "main",
                    fixed_commit=fixed_commit or None,
                )
            )

    return specs


def default_repo_id_from_url(repo_url: str, line_number: int) -> str:
    owner, repo = parse_owner_repo(repo_url)
    return f"{line_number:02d}_{owner}_{repo}"


def parse_owner_repo(repo_url: str) -> tuple[str, str]:
    stripped = repo_url.strip()
    if stripped.endswith(".git"):
        stripped = stripped[:-4]

    if stripped.startswith("git@github.com:"):
        path = stripped.split(":", 1)[1]
        owner, repo = path.split("/", 1)
        return owner, repo

    parsed = urlparse(stripped)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise GraderError(f"Could not parse owner/repo from URL: {repo_url}")
    owner, repo = parts[0], parts[1]
    return owner, repo


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": shlex.join(command),
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": f"Command not found: {command[0]}",
            "command": shlex.join(command),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": f"Timed out after {timeout}s",
            "command": shlex.join(command),
        }


def ensure_clone(repo: RepoSpec, clones_dir: Path) -> Path:
    owner, repo_name = parse_owner_repo(repo.repo_url)
    target = clones_dir / slugify_repo_name(f"{repo.repo_id}_{owner}_{repo_name}")

    if not target.exists():
        result = run_command(["git", "clone", repo.repo_url, str(target)], timeout=240)
        if not result["ok"]:
            raise GraderError(f"git clone failed for {repo.repo_url}: {result['stderr']}")

    fetch_result = run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=target, timeout=240)
    if not fetch_result["ok"]:
        raise GraderError(f"git fetch failed for {repo.repo_url}: {fetch_result['stderr']}")

    return target


def resolve_cutoff_commit(repo_dir: Path, branch: str, cutoff_local: str, timezone_name: str) -> str:
    local_dt = datetime.strptime(cutoff_local, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(timezone_name))
    cutoff_text = local_dt.strftime("%Y-%m-%d %H:%M:%S %z")
    candidates = [branch, f"origin/{branch}", f"refs/remotes/origin/{branch}"]

    for ref in candidates:
        result = run_command(["git", "rev-list", "-n", "1", f"--before={cutoff_text}", ref], cwd=repo_dir)
        commit = result["stdout"].strip()
        if commit:
            return commit

    raise GraderError(f"No commit found on branch {branch} before cutoff {cutoff_text}")


def checkout_commit(repo_dir: Path, commit: str) -> None:
    reset_result = run_command(["git", "reset", "--hard"], cwd=repo_dir)
    if not reset_result["ok"]:
        raise GraderError(f"git reset failed: {reset_result['stderr']}")

    clean_result = run_command(["git", "clean", "-fd"], cwd=repo_dir)
    if not clean_result["ok"]:
        raise GraderError(f"git clean failed: {clean_result['stderr']}")

    checkout_result = run_command(["git", "checkout", "--force", commit], cwd=repo_dir, timeout=120)
    if not checkout_result["ok"]:
        raise GraderError(f"git checkout failed for commit {commit}: {checkout_result['stderr']}")


def git_commit_metadata(repo_dir: Path, commit: str) -> dict[str, Any]:
    result = run_command(["git", "show", "-s", "--format=%H%n%cI%n%s", commit], cwd=repo_dir)
    lines = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    return {
        "resolved_commit": commit,
        "commit_datetime": lines[1] if len(lines) > 1 else "",
        "commit_subject": lines[2] if len(lines) > 2 else "",
    }


def list_repo_files(repo_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_dir.rglob("*"):
        if ".git" in path.parts:
            continue
        files.append(path)
    return files


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")
    except OSError:
        return ""


def python_files(repo_dir: Path) -> list[Path]:
    return [p for p in repo_dir.rglob("*.py") if ".git" not in p.parts]


def is_notebook_path(path: Path) -> bool:
    return path.suffix == ".ipynb" and not any(part in NOTEBOOK_EXCLUDE_PARTS for part in path.parts)


def count_lines_of_code(repo_dir: Path) -> int:
    total = 0
    for path in python_files(repo_dir):
        try:
            total += sum(1 for _ in path.open("r", encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return total


def find_readme(repo_dir: Path) -> Path | None:
    candidates = [repo_dir / "README.md", repo_dir / "readme.md", repo_dir / "README.MD"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for path in repo_dir.glob("README*"):
        if path.is_file():
            return path
    return None


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = stem.replace("data_", "").replace("_data", "")
    stem = stem.replace("ml_", "")
    return stem


def scan_repo_structure(repo_dir: Path) -> dict[str, Any]:
    src_dir = repo_dir / "src"
    tests_dir = repo_dir / "tests"
    notebooks = [p for p in repo_dir.rglob("*.ipynb") if is_notebook_path(p)]
    py_files = python_files(repo_dir)
    return {
        "src_dir_present": int(src_dir.exists()),
        "tests_dir_present": int(tests_dir.exists()),
        "repo_notebook_count": len(notebooks),
        "python_module_count": len(py_files),
        "lines_of_code": count_lines_of_code(repo_dir),
    }


def scan_readme(repo_dir: Path) -> dict[str, Any]:
    readme_path = find_readme(repo_dir)
    if not readme_path:
        return {
            "readme_present": 0,
            "readme_word_count": 0,
            "readme_run_keywords_found": 0,
            "readme_run_instructions_present": 0,
            "business_sections_found": 0,
            "business_section_ratio": 0.0,
            "readme_has_badges": 0,
            "readme_path": "",
        }

    text = read_text(readme_path)
    text_lower = text.lower()
    business_hits = sum(any(keyword in text_lower for keyword in keywords) for keywords in BUSINESS_KEYWORDS.values())
    run_hits = sum(keyword in text_lower for keyword in RUN_SECTION_KEYWORDS)

    return {
        "readme_present": 1,
        "readme_word_count": len(re.findall(r"\b\w+\b", text)),
        "readme_run_keywords_found": run_hits,
        "readme_run_instructions_present": int(run_hits >= 2),
        "business_sections_found": business_hits,
        "business_section_ratio": round(business_hits / max(len(BUSINESS_KEYWORDS), 1), 3),
        "readme_has_badges": int("img.shields.io" in text_lower or "badge" in text_lower),
        "readme_path": str(readme_path.relative_to(repo_dir)),
    }

def _is_target_experiment_notebook(path: Path) -> bool:
    stem = path.stem.lower()
    has_target = any(token in stem for token in NOTEBOOK_TARGET_TOKENS)
    has_excluded_name = any(token in stem for token in NOTEBOOK_EXCLUDE_NAME_TOKENS)
    return has_target and not has_excluded_name

def _notebook_imports_from_src(payload: dict[str, Any]) -> bool:
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if re.search(r"^\s*from\s+src(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s+import\s+", source, flags=re.MULTILINE):
            return True

        if re.search(r"^\s*import\s+src(?:\.[A-Za-z_][A-Za-z0-9_]*)*", source, flags=re.MULTILINE):
            return True

    return False

def scan_notebooks(repo_dir: Path) -> dict[str, Any]:
    all_notebooks = [path for path in repo_dir.rglob("*.ipynb") if is_notebook_path(path)]
    target_notebooks = [
        path for path in all_notebooks if _is_target_experiment_notebook(path)
    ]

    valid_count = 0
    valid_target_notebooks_with_src_import = 0
    code_cells = 0
    markdown_cells = 0
    selected_names: list[str] = []
    target_notebooks_with_src_import_names: list[str] = []

    for notebook in target_notebooks:
        try:
            payload = json.loads(read_text(notebook))
            valid_count += 1
            if _notebook_imports_from_src(payload):
                valid_target_notebooks_with_src_import += 1
                target_notebooks_with_src_import_names.append(str(notebook.relative_to(repo_dir)))
            selected_names.append(str(notebook.relative_to(repo_dir)))
            for cell in payload.get("cells", []):
                if cell.get("cell_type") == "code":
                    code_cells += 1
                elif cell.get("cell_type") == "markdown":
                    markdown_cells += 1
        except json.JSONDecodeError:
            continue

    return {
        "all_notebooks_found": len(all_notebooks),
        "target_notebooks_found": len(target_notebooks),
        "valid_target_notebooks": valid_count,
        "notebook_code_cells": code_cells,
        "notebook_markdown_cells": markdown_cells,
        "notebook_present": int(len(target_notebooks) > 0),
        "notebook_valid": int(valid_count > 0),
        "target_notebook_names": " | ".join(selected_names),
        "notebook_target_rule": "name contains exp, experiment, sandbox, sanbox, sand, model_main, or modular, excluding legacy, and notebook imports from src",
        "valid_target_notebooks_with_src_import": valid_target_notebooks_with_src_import,
        "target_notebooks_with_src_import_names": " | ".join(target_notebooks_with_src_import_names),
    }

def _validation_candidate_paths(repo_dir: Path) -> list[Path]:
    return [
        path for path in python_files(repo_dir)
        if any(token in path.stem.lower() for token in ["valid", "schema"])
    ]

def _clean_python_for_detection(text: str) -> str:
    text = re.sub(r'"""[\s\S]*?"""', " ", text)
    text = re.sub(r"'''[\s\S]*?'''", " ", text)
    text = re.sub(r"#.*", " ", text)
    return text.lower()


def _is_test_or_setup_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    lowered_name = path.name.lower()
    lowered_stem = path.stem.lower()
    return (
        "tests" in lowered_parts
        or lowered_name == "conftest.py"
        or lowered_name == "setup.py"
        or lowered_stem.startswith("test_")
        or lowered_stem.endswith("_test")
    )


def production_python_files(repo_dir: Path) -> list[Path]:
    return [path for path in python_files(repo_dir) if not _is_test_or_setup_path(path)]


def _is_runtime_owner_python_path(repo_dir: Path, path: Path) -> bool:
    if _is_test_or_setup_path(path) or is_notebook_path(path):
        return False

    try:
        rel_path = path.relative_to(repo_dir)
    except ValueError:
        rel_path = path

    rel_text = str(rel_path).lower()
    stem = path.stem.lower()

    if rel_text in {"main.py", "src/main.py"}:
        return True

    return any(token in stem for token in ["train", "trainer", "pipeline", "run"])


def runtime_owner_python_files(repo_dir: Path) -> list[Path]:
    return [path for path in production_python_files(repo_dir) if _is_runtime_owner_python_path(repo_dir, path)]


def _count_yaml_leaf_keys(value: Any, depth: int = 0, max_depth: int = 4) -> int:
    if depth > max_depth:
        return 0
    if isinstance(value, dict):
        total = 0
        for child in value.values():
            if isinstance(child, dict):
                total += _count_yaml_leaf_keys(child, depth + 1, max_depth)
            elif isinstance(child, list):
                total += _count_yaml_leaf_keys(child, depth + 1, max_depth)
            else:
                total += 1
        return total
    if isinstance(value, list):
        total = 0
        for child in value:
            total += _count_yaml_leaf_keys(child, depth + 1, max_depth)
        return total
    return 0


def _string_constant_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _runtime_path_literal(value: str | None) -> bool:
    if not value:
        return False
    literal = value.strip().replace("\\", "/")
    path_re = re.compile(
        r"^(?:data|models|artifacts|outputs|reports)/[^'\"]+\.(?:csv|parquet|joblib|pkl|pickle)$",
        flags=re.IGNORECASE,
    )
    return bool(path_re.search(literal))


def _path_literal_hit_count(tree: ast.AST) -> int:
    hits = 0
    for node in ast.walk(tree):
        literal_nodes: list[ast.AST] = []

        if isinstance(node, ast.Assign):
            literal_nodes.append(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            literal_nodes.append(node.value)
        elif isinstance(node, ast.Call):
            literal_nodes.extend(
                keyword.value for keyword in node.keywords if keyword.arg and keyword.value is not None
            )

        for literal_node in literal_nodes:
            literal = _string_constant_value(literal_node)
            if _runtime_path_literal(literal):
                hits += 1

    return hits


def _name_contains_config_token(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ["config", "cfg", "settings", "params"])


def _value_is_config_derived(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return _name_contains_config_token(node.id)

    if isinstance(node, ast.Subscript):
        return _value_is_config_derived(node.value)

    if isinstance(node, ast.Attribute):
        if _name_contains_config_token(node.attr):
            return True
        return _value_is_config_derived(node.value)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "getenv"
            )
        if isinstance(node.func, ast.Name):
            return _name_contains_config_token(node.func.id)

    return False


def _numeric_constant_value(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        value = _numeric_constant_value(node.operand)
        return -value if value is not None else None
    return None


def _hyperparameter_hit_count(tree: ast.AST) -> int:
    target_names = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "num_leaves",
        "min_samples_split",
        "min_samples_leaf",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "random_state",
        "test_size",
        "val_size",
        "batch_size",
        "epochs",
        "dropout",
    }
    hits = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if _value_is_config_derived(node.value):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id.lower() in target_names
                    and _numeric_constant_value(node.value) is not None
                ):
                    hits += 1
        elif isinstance(node, ast.AnnAssign):
            if node.value is None or _value_is_config_derived(node.value):
                continue
            if (
                isinstance(node.target, ast.Name)
                and node.target.id.lower() in target_names
                and _numeric_constant_value(node.value) is not None
            ):
                hits += 1
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg and keyword.arg.lower() in target_names:
                    if _value_is_config_derived(keyword.value):
                        continue
                    if _numeric_constant_value(keyword.value) is not None:
                        hits += 1

    return hits


def _secret_like_literal_hit_count(cleaned_text: str) -> int:
    pattern = re.compile(
        r"\b(?:api[_-]?key|token|secret|password|passwd|client_secret|access_key)\b\s*=\s*['\"][a-z0-9_\-\/+=]{8,}['\"]"
    )
    return len(pattern.findall(cleaned_text))


def _scan_ignore_for_env(ignore_file_path: Path) -> int:
    if not ignore_file_path.exists():
        return 0
    for raw_line in read_text(ignore_file_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line in {".env", ".env*", "*.env"}:
            return 1
    return 0


def _git_ls_files(repo_dir: Path) -> list[str]:
    result = run_command(["git", "ls-files"], cwd=repo_dir, timeout=120)
    if not result["ok"]:
        return []
    return sorted(line.strip() for line in result["stdout"].splitlines() if line.strip())


def _tracked_env_like_files(repo_dir: Path) -> list[str]:
    safe_names = {".env.example", ".env.sample", ".env.template"}
    risky: list[str] = []
    for rel_path in _git_ls_files(repo_dir):
        name = Path(rel_path).name.lower()
        if name.startswith(".env") and name not in safe_names:
            risky.append(rel_path)
    return sorted(risky)


def _tracked_source_like_files(repo_dir: Path) -> list[Path]:
    tracked_files = _git_ls_files(repo_dir)
    production_python_relpaths = {
        str(path.relative_to(repo_dir)).replace("\\", "/")
        for path in production_python_files(repo_dir)
    }

    selected: list[Path] = []
    for rel_path in tracked_files:
        rel_path_obj = Path(rel_path)
        rel_text = rel_path.replace("\\", "/")
        lowered = rel_text.lower()

        if lowered in {"environment.yml", "conda-lock.yml"}:
            continue
        if any(part in lowered for part in ["/tests/", "/notebooks/", "/outputs/", "/feedback/", "/grading_workspace/"]):
            continue
        if lowered.startswith(("tests/", "notebooks/", "outputs/", "feedback/", "grading_workspace/")):
            continue
        if lowered.endswith((".lock", ".ipynb")):
            continue

        suffix = rel_path_obj.suffix.lower()
        if suffix == ".py":
            if rel_text not in production_python_relpaths:
                continue
        elif suffix not in {".yaml", ".yml", ".sh"}:
            continue

        selected.append(repo_dir / rel_path_obj)

    return selected


def _strip_non_python_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _is_secret_like_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in {
        "api_key",
        "api-secret",
        "token",
        "secret",
        "password",
        "passwd",
        "client_secret",
        "access_key",
        "private_key",
    }


def _is_placeholder_secret_value(value: str) -> bool:
    lowered = value.strip().lower()
    return (
        lowered.startswith("secrets.")
        or lowered in {"your_token_here", "change_me", "todo"}
        or "[" in value
        or "]" in value
    )


def _is_env_reference_value(value: str) -> bool:
    stripped = value.strip()
    return bool(
        re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", stripped)
        or re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", stripped)
    )


def _is_env_var_name_value(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{5,}", value.strip()))


def _looks_like_real_secret_literal(value: str) -> bool:
    stripped = value.strip().strip("'\"")
    if len(stripped) < 12:
        return False
    if _is_placeholder_secret_value(stripped):
        return False
    if _is_env_reference_value(stripped):
        return False
    if _is_env_var_name_value(stripped):
        return False
    if stripped.lower().startswith("secrets."):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_./+=-]{12,}", stripped))


def _python_secret_literal_match_count(text: str) -> int:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0

    hits = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _string_constant_value(node.value)
            if value is None or _is_placeholder_secret_value(value):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_secret_like_name(target.id):
                    hits += 1
        elif isinstance(node, ast.AnnAssign):
            value = _string_constant_value(node.value) if node.value is not None else None
            if value is None or _is_placeholder_secret_value(value):
                continue
            if isinstance(node.target, ast.Name) and _is_secret_like_name(node.target.id):
                hits += 1
        elif isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                key = _string_constant_value(key_node) if key_node is not None else None
                value = _string_constant_value(value_node)
                if key and _is_secret_like_name(key) and value is not None and not _is_placeholder_secret_value(value):
                    hits += 1

    return hits


def _secret_literal_match_count(text: str) -> int:
    text = re.sub(r"\$\{\{\s*secrets\.[A-Za-z0-9_]+\s*\}\}", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsecrets\.[A-Za-z0-9_]+\b", " ", text, flags=re.IGNORECASE)
    pattern = re.compile(
        r"^(?:export\s+)?(?P<key>[A-Za-z0-9_\"'-]*(?:api[_-]?key|api-secret|token|secret|password|passwd|client_secret|access_key|private_key)[A-Za-z0-9_\"'-]*)"
        r"\s*(?:=|:)\s*(?P<value>.+?)$",
        flags=re.IGNORECASE,
    )
    hits = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue
        key_text = match.group("key").strip().strip("'\"").lower()
        if key_text.endswith(("_env", "_env_name", "_var")):
            continue
        value_text = match.group("value").strip()
        if not _looks_like_real_secret_literal(value_text):
            continue
        hits += 1
    return hits


def _count_print_calls(tree: ast.AST) -> int:
    hits = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            hits += 1
    return hits


def _scan_logger_module(logger_path: Path) -> dict[str, int]:
    cleaned = _clean_python_for_detection(read_text(logger_path))

    file_handler_present = int(
        "filehandler(" in cleaned
        or "rotatingfilehandler(" in cleaned
        or "timedrotatingfilehandler(" in cleaned
    )
    stream_handler_present = int("streamhandler(" in cleaned)
    logfile_path_present = int(
        bool(re.search(r'["\'][^"\']*\.log["\']', cleaned))
        or bool(re.search(r'path\s*\([^)]*\.log', cleaned))
    )

    return {
        "log_file_handler_present": file_handler_present,
        "log_stream_handler_present": stream_handler_present,
        "log_logfile_path_present": logfile_path_present,
    }


def scan_security_secrets(repo_dir: Path) -> dict[str, Any]:
    evidence = {
        "sec_gitignore_excludes_env": 0,
        "sec_dockerignore_present": 0,
        "sec_dockerignore_excludes_env": 0,
        "sec_env_file_present": 0,
        "sec_env_file_tracked_by_git": 0,
        "sec_env_example_present": 0,
        "sec_dotenv_usage_signal": 0,
        "sec_secret_literal_hits": 0,
        "sec_secret_literal_files": "",
        "sec_tracked_env_like_files": "",
        "security_secrets_cap_reason": "",
    }

    gitignore_path = repo_dir / ".gitignore"
    dockerignore_path = repo_dir / ".dockerignore"
    tracked_files = _git_ls_files(repo_dir)
    tracked_env_like_files = _tracked_env_like_files(repo_dir)

    evidence["sec_gitignore_excludes_env"] = _scan_ignore_for_env(gitignore_path)
    evidence["sec_dockerignore_present"] = int(dockerignore_path.exists())
    evidence["sec_dockerignore_excludes_env"] = _scan_ignore_for_env(dockerignore_path)
    evidence["sec_env_file_present"] = int((repo_dir / ".env").exists())
    evidence["sec_env_file_tracked_by_git"] = int(".env" in tracked_files)
    evidence["sec_env_example_present"] = int((repo_dir / ".env.example").exists())
    evidence["sec_tracked_env_like_files"] = " | ".join(tracked_env_like_files)

    dotenv_hits = 0
    for path in production_python_files(repo_dir):
        cleaned = _clean_python_for_detection(read_text(path))
        if "load_dotenv(" in cleaned or re.search(r"\bos\.getenv\s*\(", cleaned):
            dotenv_hits += 1
    evidence["sec_dotenv_usage_signal"] = int(dotenv_hits > 0)

    secret_files: list[str] = []
    secret_hits = 0
    for path in _tracked_source_like_files(repo_dir):
        text = read_text(path)
        if path.suffix.lower() == ".py":
            hits = _python_secret_literal_match_count(text)
        else:
            hits = _secret_literal_match_count(_strip_non_python_comments(text))
        if hits > 0:
            secret_hits += hits
            secret_files.append(str(path.relative_to(repo_dir)).replace("\\", "/"))

    evidence["sec_secret_literal_hits"] = secret_hits
    evidence["sec_secret_literal_files"] = " | ".join(sorted(secret_files))
    return evidence


def scan_logging_observability(repo_dir: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "log_print_statement_hits": 0,
        "log_print_free": 0,
        "log_print_hit_files": "",
        "log_logger_module_present": 0,
        "log_logger_module_path": "",
        "log_logger_module_name": "",
        "log_logger_module_fallback_used": 0,
        "log_file_handler_present": 0,
        "log_stream_handler_present": 0,
        "log_dual_output_signal": 0,
        "log_logfile_path_present": 0,
        "log_logger_usage_signal": 0,
        "log_logger_usage_count": 0,
        "logging_observability_cap_reason": "",
    }

    total_print_hits = 0
    print_hit_files: list[str] = []
    for path in production_python_files(repo_dir):
        text = read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        hits = _count_print_calls(tree)
        if hits > 0:
            total_print_hits += hits
            print_hit_files.append(str(path.relative_to(repo_dir)).replace("\\", "/"))

    evidence["log_print_statement_hits"] = total_print_hits
    allowed_print_calls = 3
    evidence["log_print_free"] = int(total_print_hits <= allowed_print_calls)
    evidence["log_print_hit_files"] = " | ".join(print_hit_files)

    logger_path = repo_dir / "src" / "logger.py"
    fallback_logger_path = repo_dir / "src" / "logging.py"
    selected_logger_path: Path | None = None
    if logger_path.exists():
        selected_logger_path = logger_path
        evidence["log_logger_module_present"] = 1
        evidence["log_logger_module_path"] = "src/logger.py"
        evidence["log_logger_module_name"] = "logger.py"
        evidence["log_logger_module_fallback_used"] = 0
    elif fallback_logger_path.exists():
        selected_logger_path = fallback_logger_path
        evidence["log_logger_module_present"] = 1
        evidence["log_logger_module_path"] = "src/logging.py"
        evidence["log_logger_module_name"] = "logging.py"
        evidence["log_logger_module_fallback_used"] = 1

    if selected_logger_path is not None:
        evidence.update(_scan_logger_module(selected_logger_path))
        evidence["log_dual_output_signal"] = int(
            evidence["log_file_handler_present"]
            and evidence["log_stream_handler_present"]
        )

    usage_count = 0
    for path in production_python_files(repo_dir):
        if selected_logger_path is not None and path.resolve() == selected_logger_path.resolve():
            continue
        cleaned = _clean_python_for_detection(read_text(path))
        usage_patterns = ["get_logger(", "setup_logger(", "getlogger("]
        if selected_logger_path == logger_path:
            usage_patterns.extend(["from src.logger import", "import src.logger"])
        elif selected_logger_path == fallback_logger_path:
            usage_patterns.extend(["from src.logging import", "import src.logging"])
        if any(pattern in cleaned for pattern in usage_patterns):
            usage_count += 1

    evidence["log_logger_usage_count"] = usage_count
    evidence["log_logger_usage_signal"] = int(usage_count >= 1)
    return evidence


def scan_experiment_tracking(repo_dir: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "wandb_import_present": 0,
        "wandb_init_in_main": 0,
        "wandb_config_logged": 0,
        "wandb_run_metadata_logged": 0,
        "wandb_eval_metrics_logged": 0,
        "wandb_rich_eval_tracking_logged": 0,
        "wandb_model_artifact_logged": 0,
        "wandb_cap_reason": "",
    }

    production_cleaned: list[str] = []
    for path in production_python_files(repo_dir):
        cleaned = _clean_python_for_detection(read_text(path))
        production_cleaned.append(cleaned)

    if any(
        "import wandb" in cleaned
        or "from wandb import" in cleaned
        or 'importlib.import_module("wandb")' in cleaned
        or "importlib.import_module('wandb')" in cleaned
        for cleaned in production_cleaned
    ):
        evidence["wandb_import_present"] = 1

    main_path = _find_main_path(repo_dir)
    main_cleaned = ""
    if main_path and main_path.exists():
        main_cleaned = _clean_python_for_detection(read_text(main_path))
        if "wandb.init(" in main_cleaned or re.search(r"\b\w*wandb\w*\.init\s*\(", main_cleaned):
            evidence["wandb_init_in_main"] = 1
        if evidence["wandb_init_in_main"]:
            idx = main_cleaned.find("wandb.init(")
            if idx != -1:
                window = main_cleaned[idx:idx + 800]
                if "config=" in window:
                    evidence["wandb_config_logged"] = 1

    metadata_patterns = [
        r"\braw_rows\b",
        r"\braw_cols\b",
        r"\bclean_rows\b",
        r"\bclean_cols\b",
        r"\btrain_rows\b",
        r"\btrain_cols\b",
        r"\btest_rows\b",
        r"\btest_cols\b",
        r"\bval_rows\b",
        r"\bval_cols\b",
        r"\btrain_size\b",
        r"\btest_size\b",
        r"\bval_size\b",
        r"\bsplit(?:_| )sizes?\b",
        r"\bselected_model_name\b",
        r"\bentrypoint\b",
        r"\bmodel_artifact_path\b",
    ]
    eval_namespace_patterns = [
        r"metrics/",
        r"\bval_",
        r"\btest_",
    ]
    eval_metric_name_patterns = [
        r"\brmse\b",
        r"\bmae\b",
        r"\baccuracy\b",
        r"\bf1\b",
        r"\bprecision\b",
        r"\brecall\b",
        r"\bauc\b",
        r"\br2\b",
    ]
    eval_context_patterns = [
        r"\beval",
        r"\bevaluate",
        r"\bevaluation\b",
        r"\bmetric\b",
        r"\bmetrics\b",
        r"\bvalidation\b",
        r"\btest\b",
        r"\bval\b",
    ]
    rich_eval_tokens = [
        "comparison_table",
        "comparison table",
        "confusion_matrix",
        "confusion matrix",
        "roc_curve",
        "roc curve",
        "pr_curve",
        "pr curve",
        "calibration_table",
        "calibration table",
    ]
    model_context_tokens = [
        "type=\"model\"",
        "type='model'",
        "model_artifact",
        "model artifact",
        "model_path",
        "model.pkl",
        "model.joblib",
        "model.pt",
    ]

    for path in production_python_files(repo_dir):
        text = read_text(path)
        cleaned = _clean_python_for_detection(text)

        has_artifact_signal = (
            "wandb.artifact(" in cleaned
            or re.search(r"\b\w*wandb\w*\.artifact\s*\(", cleaned)
            or "log_artifact(" in cleaned
            or "wandb.log_artifact(" in cleaned
            or re.search(r"\b\w*wandb\w*\.log_artifact\s*\(", cleaned)
            or re.search(r"\b\w*run\.log_artifact\s*\(", cleaned)
        )
        has_model_context = any(token in cleaned for token in model_context_tokens)
        if has_artifact_signal and has_model_context:
            evidence["wandb_model_artifact_logged"] = 1

        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        assignments = _collect_name_assignments(tree, text)
        parent_map = _build_parent_map(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_wandb_log_context(node):
                is_main_file = path == main_path
                call_context = _call_context_text(path, text, node, assignments, parent_map)

                if _matches_any_pattern(call_context, metadata_patterns):
                    evidence["wandb_run_metadata_logged"] = 1

                if is_main_file:
                    has_eval_namespace = _matches_any_pattern(call_context, eval_namespace_patterns)
                    has_eval_metric_name = _matches_any_pattern(call_context, eval_metric_name_patterns)
                    has_eval_context = _matches_any_pattern(call_context, eval_context_patterns)
                    has_evaluate_model_assignment = "evaluate_model(" in call_context

                    if has_eval_namespace or (has_eval_metric_name and has_eval_context) or has_evaluate_model_assignment:
                        evidence["wandb_eval_metrics_logged"] = 1

                if (
                    not _is_config_gated_rich_tracking(node, parent_map, text)
                    and (
                    "wandb.table(" in call_context
                    or "wandb.plot." in call_context
                    or any(token in call_context for token in rich_eval_tokens)
                    )
                ):
                    evidence["wandb_rich_eval_tracking_logged"] = 1

            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Subscript) and _is_wandb_summary_context(target) for target in node.targets):
                    summary_text = _node_text(text, node)
                    if _matches_any_pattern(summary_text, metadata_patterns):
                        evidence["wandb_run_metadata_logged"] = 1

            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Subscript) and _is_wandb_summary_context(node.target):
                    summary_text = _node_text(text, node)
                    if _matches_any_pattern(summary_text, metadata_patterns):
                        evidence["wandb_run_metadata_logged"] = 1

            if isinstance(node, ast.Call):
                call_text = _node_text(text, node)
                if (
                    not _is_config_gated_rich_tracking(node, parent_map, text)
                    and ("wandb.table(" in call_text or "wandb.plot." in call_text)
                ):
                    evidence["wandb_rich_eval_tracking_logged"] = 1

    return evidence


def _attribute_chain_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _wandb_log_context_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _attribute_chain_name(node.func).lower()
    if isinstance(node, ast.Subscript):
        return _attribute_chain_name(node.value).lower()
    return _attribute_chain_name(node).lower()


def _is_wandb_log_context(node: ast.AST) -> bool:
    context_name = _wandb_log_context_name(node)
    if not context_name.endswith(".log"):
        return False
    root = context_name.split(".", 1)[0]
    return "wandb" in context_name or root == "run" or root.endswith("_run")


def _is_wandb_summary_context(node: ast.AST) -> bool:
    context_name = _wandb_log_context_name(node)
    if not context_name.endswith(".summary"):
        return False
    root = context_name.split(".", 1)[0]
    return "wandb" in context_name or root == "run" or root.endswith("_run")


def _node_text(text: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    return (ast.get_source_segment(text, node) or "").lower()


def _collect_name_assignments(tree: ast.AST, text: str) -> dict[str, list[tuple[int, str]]]:
    assignments: dict[str, list[tuple[int, str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value_text = _node_text(text, node.value)
            if not value_text:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append((node.lineno, value_text))
        elif isinstance(node, ast.AnnAssign):
            value_text = _node_text(text, node.value)
            if value_text and isinstance(node.target, ast.Name):
                assignments.setdefault(node.target.id, []).append((node.lineno, value_text))

    for entries in assignments.values():
        entries.sort(key=lambda item: item[0])
    return assignments


def _resolve_name_assignment(
    name: str,
    lineno: int,
    assignments: dict[str, list[tuple[int, str]]],
) -> str:
    latest = ""
    for assignment_lineno, assignment_text in assignments.get(name, []):
        if assignment_lineno >= lineno:
            break
        latest = assignment_text
    return latest


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    return parent_map


def _enclosing_function_name(node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> str:
    current = parent_map.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name.lower()
        current = parent_map.get(current)
    return ""


def _is_config_gated_rich_tracking(
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
    text: str,
) -> bool:
    current = parent_map.get(node)
    while current is not None:
        if isinstance(current, ast.If):
            test_text = _node_text(text, current.test)
            if "cfg" in test_text or "config" in test_text:
                return True
        current = parent_map.get(current)
    return False


def _call_context_text(
    path: Path,
    text: str,
    node: ast.Call,
    assignments: dict[str, list[tuple[int, str]]],
    parent_map: dict[ast.AST, ast.AST],
) -> str:
    parts = [path.stem.lower(), _enclosing_function_name(node, parent_map), _node_text(text, node)]

    if node.args and isinstance(node.args[0], ast.Name):
        parts.append(_resolve_name_assignment(node.args[0].id, node.lineno, assignments))

    for keyword in node.keywords:
        if isinstance(keyword.value, ast.Name):
            parts.append(_resolve_name_assignment(keyword.value.id, node.lineno, assignments))

    return " ".join(part for part in parts if part)


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def scan_config_reproducibility(repo_dir: Path) -> dict[str, Any]:
    evidence = {
        "config_yaml_present": 0,
        "config_yaml_has_keys": 0,
        "runtime_config_keys_found": 0,
        "env_contract_signal": 0,
        "gitignore_excludes_env": 0,
        "conda_lock_yml_present": 0,
        "main_reads_config_signal": 0,
        "dotenv_usage_signal": 0,
        "code_hardcoded_path_hits": 0,
        "code_hardcoded_hyperparam_hits": 0,
        "secret_like_literal_hits": 0,
        "config_reproducibility_cap_reason": "",
    }

    config_candidates = [repo_dir / "config.yaml", repo_dir / "config.yml"]
    config_path = next((path for path in config_candidates if path.exists()), None)
    if config_path:
        evidence["config_yaml_present"] = 1
        try:
            payload = yaml.safe_load(read_text(config_path))
        except yaml.YAMLError:
            payload = None

        if isinstance(payload, dict) and payload:
            evidence["config_yaml_has_keys"] = 1
            evidence["runtime_config_keys_found"] = _count_yaml_leaf_keys(payload)

    gitignore_path = repo_dir / ".gitignore"
    if gitignore_path.exists():
        gitignore_text = read_text(gitignore_path)
        for raw_line in gitignore_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            if line in {".env", ".env*", "*.env"}:
                evidence["gitignore_excludes_env"] = 1
                break

    evidence["conda_lock_yml_present"] = int(any(path.is_file() for path in repo_dir.rglob("conda-lock.yml")))

    main_path = _find_main_path(repo_dir)
    if main_path and main_path.exists():
        main_text = read_text(main_path)
        main_clean = _clean_python_for_detection(main_text)
        if (
            "yaml.safe_load" in main_clean
            or re.search(r"\byaml\.load\s*\(", main_clean)
            or re.search(r"\bopen\s*\([^\n]{0,120}config[^\n]{0,120}\)", main_clean)
            or re.search(r"\bload_config\s*\(", main_clean)
            or re.search(r"\bget_config\s*\(", main_clean)
            or re.search(r"\bread_config\s*\(", main_clean)
            or re.search(r"\bconfig\s*=\s*(?:yaml\.(?:safe_load|load)|load_config|get_config|read_config)\s*\(", main_clean)
        ):
            evidence["main_reads_config_signal"] = 1

    runtime_owner_paths = set(runtime_owner_python_files(repo_dir))

    for path in production_python_files(repo_dir):
        text = read_text(path)
        cleaned = _clean_python_for_detection(text)

        if "load_dotenv(" in cleaned or re.search(r"\bos\.getenv\s*\(", cleaned):
            evidence["dotenv_usage_signal"] += 1

        try:
            tree = ast.parse(text)
        except SyntaxError:
            evidence["secret_like_literal_hits"] += _secret_like_literal_hit_count(cleaned)
            continue

        evidence["secret_like_literal_hits"] += _secret_like_literal_hit_count(cleaned)

        if path in runtime_owner_paths:
            evidence["code_hardcoded_path_hits"] += _path_literal_hit_count(tree)
            evidence["code_hardcoded_hyperparam_hits"] += _hyperparameter_hit_count(tree)

    evidence["dotenv_usage_signal"] = int(evidence["dotenv_usage_signal"] > 0)
    evidence["env_contract_signal"] = int(
        evidence["gitignore_excludes_env"] and evidence["dotenv_usage_signal"]
    )
    return evidence

def scan_validation_breadth(repo_dir: Path) -> dict[str, Any]:
    validation_paths = _validation_candidate_paths(repo_dir)

    signals = {
        "validation_function_present": 0,
        "validation_raise_present": 0,
        "validation_none_or_type_guard": 0,
        "validation_empty_guard": 0,
        "validation_required_columns_guard": 0,
        "validation_missing_values_guard": 0,
        "validation_target_guard": 0,
        "validation_dtype_guard": 0,
        "validation_range_or_domain_guard": 0,
    }

    for path in validation_paths:
        raw_text = read_text(path)
        code = _clean_python_for_detection(raw_text)

        if re.search(r"\bdef\s+\w*valid\w*\s*\(", code):
            signals["validation_function_present"] = 1

        if re.search(r"\braise\s+(valueerror|typeerror|exception|runtimeerror|assertionerror)\b", code):
            signals["validation_raise_present"] = 1

        if (
            re.search(r"\bdf\s+is\s+none\b", code)
            or re.search(r"not\s+isinstance\s*\(\s*df\s*,\s*(pd\.)?dataframe\s*\)", code)
        ):
            signals["validation_none_or_type_guard"] = 1

        if (
            ".empty" in code
            or re.search(r"len\s*\(\s*df\s*\)\s*==\s*0", code)
        ):
            signals["validation_empty_guard"] = 1

        if (
            any(token in code for token in ["required_columns", "required_cols"])
            and (
                re.search(r"not\s+in\s+df\.columns", code)
                or re.search(r"\bmissing\s*=\s*\[", code)
                or re.search(r"set\s*\(\s*required_columns\s*\)", code)
                or re.search(r"set\s*\(\s*required_cols\s*\)", code)
            )
        ):
            signals["validation_required_columns_guard"] = 1

        if (
            re.search(r"\.isna\s*\(\)\.any\s*\(\)", code)
            or re.search(r"\.isnull\s*\(\)\.any\s*\(\)", code)
            or re.search(r"\.notna\s*\(\)\.any\s*\(\)", code)
            or re.search(r"\.notnull\s*\(\)\.any\s*\(\)", code)
        ):
            signals["validation_missing_values_guard"] = 1

        if (
            re.search(r"\btarget_column\b", code)
            and (
                re.search(r"target_column\s+not\s+in\s+df\.columns", code)
                or re.search(r"df\s*\[\s*target_column\s*\]", code)
                or re.search(r"\btarget_allowed_values\b", code)
                or re.search(r"\ballowed_values\b", code)
            )
        ):
            signals["validation_target_guard"] = 1

        if (
            re.search(r"\bis_numeric_dtype\s*\(", code)
            or re.search(r"\bis_string_dtype\s*\(", code)
            or re.search(r"\bis_bool_dtype\s*\(", code)
            or re.search(r"\bis_datetime64_any_dtype\s*\(", code)
            or re.search(r"\.dtype\b", code)
            or re.search(r"\.dtypes\b", code)
            or re.search(r"\bselect_dtypes\s*\(", code)
        ):
            signals["validation_dtype_guard"] = 1

        if (
            re.search(r"\(df\s*\[[^\]]+\]\s*<\s*-?\d+(\.\d+)?\)\.any\s*\(\)", code)
            or re.search(r"\(df\s*\[[^\]]+\]\s*>\s*-?\d+(\.\d+)?\)\.any\s*\(\)", code)
            or re.search(r"\(df\s*\[[^\]]+\]\s*<=\s*-?\d+(\.\d+)?\)\.any\s*\(\)", code)
            or re.search(r"\(df\s*\[[^\]]+\]\s*>=\s*-?\d+(\.\d+)?\)\.any\s*\(\)", code)
            or re.search(r"\.between\s*\(", code)
            or re.search(r"\.isin\s*\(", code)
            or re.search(r"\bnumeric_non_negative_cols\b", code)
        ):
            signals["validation_range_or_domain_guard"] = 1

    return signals

def scan_python_files(repo_dir: Path) -> dict[str, Any]:
    module_count = 0
    parseable_count = 0
    module_docstring_count = 0
    function_count = 0
    function_docstring_count = 0
    class_count = 0
    comment_count = 0

    for path in python_files(repo_dir):
        text = read_text(path)

        comment_count += len(re.findall(r"^\s*#", text, flags=re.MULTILINE))

        module_count += 1
        try:
            tree = ast.parse(text)
            parseable_count += 1
        except SyntaxError:
            continue

        if ast.get_docstring(tree):
            module_docstring_count += 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                if ast.get_docstring(node):
                    function_docstring_count += 1

    loc = count_lines_of_code(repo_dir)
    comment_ratio = comment_count / max(loc, 1)
    validation_signals = scan_validation_breadth(repo_dir)

    return {
        "python_module_count": module_count,
        "python_parseable_count": parseable_count,
        "module_docstring_ratio": round(module_docstring_count / max(module_count, 1), 3),
        "function_count": function_count,
        "function_docstring_ratio": round(function_docstring_count / max(function_count, 1), 3),
        "comment_count": comment_count,
        "comment_density_per_100_loc": round((comment_count / max(loc, 1)) * 100, 2),
        "comment_ratio": round(comment_ratio, 4),
        "class_count": class_count,
        **validation_signals,
    }


def _is_main_guard_test(node: ast.AST) -> bool:
    """
    Detect: if __name__ == "__main__":
    """
    if not isinstance(node, ast.Compare):
        return False
    if not isinstance(node.left, ast.Name) or node.left.id != "__name__":
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    if len(node.comparators) != 1:
        return False

    comparator = node.comparators[0]
    if isinstance(comparator, ast.Constant):
        return comparator.value == "__main__"
    if isinstance(comparator, ast.Str):
        return comparator.s == "__main__"
    return False


def _called_function_names_in_nodes(nodes: list[ast.stmt]) -> set[str]:
    """
    Return simple function names called inside a list of AST nodes.
    Example:
    - main()
    - run_pipeline()
    """
    called: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    called.add(func.id)
    return called


def scan_main_entry(repo_dir: Path) -> dict[str, Any]:
    """
    A valid entry point for this rubric means:
    - a runner file exists
    - it has a main guard
    - the main guard calls at least one top-level function defined in the same file

    This avoids false negatives when the function is named run_pipeline(),
    run(), execute_pipeline(), etc. instead of main()
    """
    candidates = [
        repo_dir / "main.py",
        repo_dir / "src" / "main.py",
        repo_dir / "run_pipeline.py",
        repo_dir / "src" / "run_pipeline.py",
        repo_dir / "run.py",
        repo_dir / "src" / "run.py",
        repo_dir / "pipeline.py",
        repo_dir / "src" / "pipeline.py",
    ]
    entry_path = next((path for path in candidates if path.exists()), None)

    if not entry_path:
        return {
            "main_py_present": 0,
            "main_guard_present": 0,
            "main_function_present": 0,
            "main_import_count": 0,
            "main_path": "",
        }

    text = read_text(entry_path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "main_py_present": 1,
            "main_guard_present": 0,
            "main_function_present": 0,
            "main_import_count": 0,
            "main_path": str(entry_path.relative_to(repo_dir)),
        }

    top_level_function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    main_guard_present = 0
    main_guard_calls_local_function = 0

    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard_test(node.test):
            main_guard_present = 1
            called_names = _called_function_names_in_nodes(node.body)
            if any(name in top_level_function_names for name in called_names):
                main_guard_calls_local_function = 1
            break

    import_count = len(
        [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    )

    return {
        "main_py_present": 1,
        "main_guard_present": main_guard_present,
        "main_function_present": main_guard_calls_local_function,
        "main_import_count": import_count,
        "main_path": str(entry_path.relative_to(repo_dir)),
    }


def scan_pipeline_modules(repo_dir: Path) -> dict[str, Any]:
    files = [normalized_stem(path) for path in python_files(repo_dir)]
    matched: dict[str, int] = {}

    for module_name, patterns in CORE_MODULE_MAP.items():
        matched[module_name] = int(any(any(pattern in stem for pattern in patterns) for stem in files))

    return {
        **{f"module_{name}": value for name, value in matched.items()},
        "core_module_count": sum(matched.values()),
    }


def _find_main_path(repo_dir: Path) -> Path | None:
    candidates = [repo_dir / "main.py", repo_dir / "src" / "main.py"]
    return next((path for path in candidates if path.exists()), None)

def scan_artifact_contract(repo_dir: Path) -> dict[str, Any]:
    main_path = _find_main_path(repo_dir)
    if not main_path or not main_path.exists():
        return {
            "processed_data_artifact_present": 0,
            "model_artifact_present": 0,
            "prediction_artifact_present": 0,
            "structured_asset_paths_present": 0,
        }

    code = _clean_python_for_detection(read_text(main_path))

    csv_save_calls = re.findall(
        r"save_csv\([^\n]*\)|\b\w+\.to_csv\([^\n]*\)",
        code,
    )

    processed_path_present = bool(
        re.search(r"\b\w*(?:clean|processed)\w*_path\b", code)
        or re.search(r'\["\w*(?:clean|processed)\w*"\]', code)
        or re.search(r"\['\w*(?:clean|processed)\w*'\]", code)
        or "data/processed" in code
        or "clean.csv" in code
        or "processed.csv" in code
    )

    model_path_present = bool(
        re.search(r"\b\w*model\w*_path\b", code)
        or re.search(r'\["\w*model\w*"\]', code)
        or re.search(r"\['\w*model\w*'\]", code)
        or "models" in code
        or ".joblib" in code
        or ".pkl" in code
        or ".pickle" in code
    )

    prediction_path_present = bool(
        re.search(r"\b\w*(?:pred|prediction|report|artifact)\w*_path\b", code)
        or re.search(r'\["\w*(?:pred|prediction|report|artifact)\w*"\]', code)
        or re.search(r"\['\w*(?:pred|prediction|report|artifact)\w*'\]", code)
        or "reports" in code
        or "artifacts" in code
        or "predictions.csv" in code
        or "prediction.csv" in code
    )

    processed_data_artifact_present = int(
        bool(csv_save_calls) and processed_path_present
    )

    model_save_present = bool(
        re.search(r"\b(?:joblib\.dump|pickle\.dump|save_model)\s*\(", code)
        or ("train_model(" in code and "model_path" in code)
    )

    model_artifact_present = int(
        model_save_present and model_path_present
    )

    inference_call_present = bool(
        re.search(r"\b\w*infer\w*\s*\(", code)
    )

    explicit_prediction_save = any(
        re.search(r"(?:pred|prediction|report|artifact)", call)
        for call in csv_save_calls
    )

    inference_saves_directly = bool(
        re.search(r"\b\w*infer\w*\s*\([^\)]*(?:save_path|output_path)\s*=", code, re.S)
    )

    prediction_artifact_present = int(
        inference_call_present
        and (
            explicit_prediction_save
            or inference_saves_directly
        )
    )

    structured_asset_paths_present = int(
        sum([
            processed_path_present,
            model_path_present,
            prediction_path_present,
        ]) >= 2
    )

    return {
        "processed_data_artifact_present": processed_data_artifact_present,
        "model_artifact_present": model_artifact_present,
        "prediction_artifact_present": prediction_artifact_present,
        "structured_asset_paths_present": structured_asset_paths_present,
    }


def _find_api_path(repo_dir: Path) -> Path | None:
    candidates = [repo_dir / "api.py", repo_dir / "src" / "api.py"]
    return next((path for path in candidates if path.exists()), None)


def _repo_config_text(repo_dir: Path) -> str:
    config_path = repo_dir / "config.yaml"
    if not config_path.exists():
        return ""
    return read_text(config_path)


def _repo_config_payload(repo_dir: Path) -> dict[str, Any]:
    config_path = repo_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml.safe_load(read_text(config_path)) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _module_reference_to_path(repo_dir: Path, module_name: str) -> Path | None:
    cleaned = module_name.strip().lstrip(".")
    if not cleaned:
        return None
    parts = cleaned.split(".")
    if parts[0] == "src":
        candidate = repo_dir.joinpath(*parts).with_suffix(".py")
        if candidate.exists():
            return candidate
        if len(parts) > 1:
            candidate = repo_dir.joinpath(*parts[1:]).with_suffix(".py")
            if candidate.exists():
                return candidate
    candidate = repo_dir.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate
    if (repo_dir / "src").exists():
        candidate = repo_dir.joinpath("src", *parts).with_suffix(".py")
        if candidate.exists():
            return candidate
    return None


def _api_helper_paths(repo_dir: Path, api_path: Path) -> list[Path]:
    try:
        tree = ast.parse(read_text(api_path))
    except SyntaxError:
        return []

    called_names: set[str] = set()
    called_modules: set[str] = set()
    import_map: dict[str, Path] = {}
    module_alias_map: dict[str, Path] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if node.level and api_path.parent.name == "src" and not module_name.startswith("src"):
                module_name = f"src.{module_name}" if module_name else "src"
            module_path = _module_reference_to_path(repo_dir, module_name)
            if not module_path:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                import_map[(alias.asname or alias.name).lower()] = module_path
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_path = _module_reference_to_path(repo_dir, alias.name)
                if not module_path:
                    continue
                module_alias_map[(alias.asname or alias.name.split(".")[-1]).lower()] = module_path

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called_names.add(func.id.lower())
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                called_modules.add(func.value.id.lower())

    helper_paths: list[Path] = []
    for name, path in import_map.items():
        if name in called_names and path not in helper_paths and path != api_path:
            helper_paths.append(path)
    for alias, path in module_alias_map.items():
        if alias in called_modules and path not in helper_paths and path != api_path:
            helper_paths.append(path)
    return helper_paths


def _registry_serving_signal(text: str) -> bool:
    return bool(
        re.search(r"\b\w+\.use_artifact\s*\(", text)
        or re.search(r"\buse_artifact\s*\(", text)
        or re.search(r"\b\w+\.artifact\s*\(", text)
    )


def _local_serving_signal(text: str) -> bool:
    has_deserialize = bool(re.search(r"\b(?:joblib|pickle)\.load\s*\(", text))
    if not has_deserialize:
        return False
    local_cues = [
        "local_path",
        "local_model_path",
        "model_source=local",
        "source == \"local\"",
        "source == 'local'",
        "loading model from local",
        "using local model artifact",
        "falling back to local",
        "fallback to local",
    ]
    if any(cue in text for cue in local_cues):
        return True
    if re.search(r"os\.(?:getenv|environ\.get)\(\s*['\"]model_source['\"]\s*,\s*['\"]local['\"]\s*\)", text):
        return True
    return False


def _prod_alias_signal(text: str, config_text: str) -> bool:
    if ":prod" in text or re.search(r'["\']prod["\']', text):
        return True
    alias_used = bool(
        re.search(r"\bartifact_(?:alias|ref(?:erence)?)\b", text)
        or re.search(r"\balias\b", text)
        or re.search(r"production_alias", text)
        or re.search(r"artifact_alias", text)
        or re.search(r"wandb_model_alias", text)
    )
    if not alias_used:
        return False
    return bool(re.search(r"(?im)^\s*(?:production_alias|artifact_alias)\s*:\s*[\"']?prod[\"']?\s*$", config_text))


def _default_source_from_code_and_config(
    combined_cleaned: str,
    config_payload: dict[str, Any],
) -> str | None:
    inference_cfg = config_payload.get("inference")
    if isinstance(inference_cfg, dict):
        source = str(inference_cfg.get("source", "")).strip().lower()
        if source in {"wandb", "local"}:
            return source

    default_patterns = [
        r"os\.getenv\(\s*['\"]model_source['\"]\s*,\s*['\"](wandb|local)['\"]\s*\)",
        r"os\.environ\.get\(\s*['\"]model_source['\"]\s*,\s*['\"](wandb|local)['\"]\s*\)",
        r"getenv\(\s*['\"]model_source['\"]\s*,\s*['\"](wandb|local)['\"]\s*\)",
        r"get\(\s*['\"]source['\"]\s*,\s*['\"](wandb|local)['\"]\s*\)",
    ]
    for pattern in default_patterns:
        match = re.search(pattern, combined_cleaned)
        if match:
            return match.group(1)
    return None


def _strip_non_code_comments(text: str) -> str:
    stripped_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        if "#" in line:
            line = line.split("#", 1)[0]
        if line.strip():
            stripped_lines.append(line)
    return "\n".join(stripped_lines)


def _api_candidate_paths(repo_dir: Path) -> list[Path]:
    candidates = [repo_dir / "src" / "api.py", repo_dir / "api.py"]
    return [path for path in candidates if path.exists()]


def _load_api_module(repo_dir: Path) -> tuple[Path | None, ast.AST | None]:
    for path in _api_candidate_paths(repo_dir):
        try:
            return path, ast.parse(read_text(path))
        except SyntaxError:
            continue
    return None, None


def _assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _call_is_fastapi(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) == "FastAPI"


def _function_returns_fastapi(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    fastapi_bound_names: set[str] = set()
    for node in function_node.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value if isinstance(node, ast.AnnAssign) else node.value
            if value is None or not _call_is_fastapi(value):
                continue
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                assigned = _assigned_name(target)
                if assigned:
                    fastapi_bound_names.add(assigned)
        elif isinstance(node, ast.Return):
            if _call_is_fastapi(node.value):
                return True
            if isinstance(node.value, ast.Name) and node.value.id in fastapi_bound_names:
                return True
    return False


def _fastapi_app_present(tree: ast.AST) -> bool:
    create_app_returns_fastapi = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "create_app":
            create_app_returns_fastapi = _function_returns_fastapi(node)
            break

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets):
            continue
        if _call_is_fastapi(node.value):
            return True
        if (
            isinstance(node.value, ast.Call)
            and _call_name(node.value.func) == "create_app"
            and create_app_returns_fastapi
        ):
            return True
    return False


def _route_literal_matches(node: ast.Call, route_path: str) -> bool:
    literal_args = [
        arg.value
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]
    if route_path in literal_args:
        return True
    for keyword in node.keywords:
        if keyword.arg == "path" and isinstance(keyword.value, ast.Constant) and keyword.value.value == route_path:
            return True
    return False


def _route_decorator_matches(decorator: ast.AST, route_path: str) -> bool:
    if not isinstance(decorator, ast.Call):
        return False
    if not isinstance(decorator.func, ast.Attribute):
        return False
    if not isinstance(decorator.func.value, ast.Name) or decorator.func.value.id != "app":
        return False
    if decorator.func.attr not in {"get", "post", "put", "patch", "delete", "route", "api_route"}:
        return False
    return _route_literal_matches(decorator, route_path)


def _find_route_handlers(
    tree: ast.AST,
    route_path: str,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    handlers: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(_route_decorator_matches(decorator, route_path) for decorator in node.decorator_list):
            handlers.append(node)
    return handlers


def _base_model_subclass_names(tree: ast.AST) -> set[str]:
    subclass_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseModel":
                subclass_names.add(node.name)
            elif isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                subclass_names.add(node.name)
    return subclass_names


def _module_reference_to_local_path(repo_dir: Path, api_path: Path, module_name: str, level: int = 0) -> Path | None:
    module = module_name.strip()
    if level:
        relative_parts = list(api_path.relative_to(repo_dir).parts[:-1])
        if level <= len(relative_parts):
            base_parts = relative_parts[: len(relative_parts) - level + 1]
        else:
            base_parts = []
        if module:
            parts = base_parts + module.split(".")
        else:
            parts = base_parts
    else:
        if not module:
            return None
        parts = module.split(".")

    candidate = repo_dir.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate

    package_init = repo_dir.joinpath(*parts, "__init__.py")
    if package_init.exists():
        return package_init
    return None


def _local_imported_base_model_names(repo_dir: Path, api_path: Path, tree: ast.AST) -> set[str]:
    imported_model_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_path = _module_reference_to_local_path(repo_dir, api_path, node.module or "", node.level)
        if module_path is None or module_path == api_path:
            continue
        try:
            imported_tree = ast.parse(read_text(module_path))
        except SyntaxError:
            continue
        available_base_models = _base_model_subclass_names(imported_tree)
        if not available_base_models:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            exported_name = alias.name
            local_name = alias.asname or alias.name
            if exported_name in available_base_models:
                imported_model_names.add(local_name)
    return imported_model_names


def _annotation_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Subscript):
        names = _annotation_names(node.value)
        slice_node = node.slice
        if isinstance(slice_node, ast.Tuple):
            for elt in slice_node.elts:
                names.update(_annotation_names(elt))
        else:
            names.update(_annotation_names(slice_node))
        return names
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for elt in node.elts:
            names.update(_annotation_names(elt))
        return names
    return set()


def _predict_uses_pydantic_contract(
    predict_handlers: list[ast.FunctionDef | ast.AsyncFunctionDef],
    base_model_names: set[str],
) -> bool:
    if not base_model_names:
        return False
    for handler in predict_handlers:
        positional_args = list(handler.args.posonlyargs) + list(handler.args.args)
        for arg in positional_args + list(handler.args.kwonlyargs):
            names = _annotation_names(arg.annotation)
            if names & base_model_names:
                return True
    return False


def _infer_import_signals(tree: ast.AST) -> tuple[set[str], set[str]]:
    infer_function_names: set[str] = set()
    infer_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").lower()
            if "infer" not in module_name and "inference" not in module_name:
                continue
            for alias in node.names:
                if alias.name != "*":
                    infer_function_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.lower()
                if "infer" in module_name or "inference" in module_name:
                    infer_module_aliases.add(alias.asname or alias.name.split(".")[-1])
    return infer_function_names, infer_module_aliases


def _predict_calls_inference_logic(
    predict_handlers: list[ast.FunctionDef | ast.AsyncFunctionDef],
    infer_function_names: set[str],
    infer_module_aliases: set[str],
) -> bool:
    for handler in predict_handlers:
        for node in ast.walk(handler):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in infer_function_names.union({"run_inference"}):
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in infer_module_aliases
            ):
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"predict", "predict_proba"}:
                return True
    return False


def _api_serving_signal_files(repo_dir: Path) -> list[Path]:
    candidates = [
        repo_dir / "Dockerfile",
        repo_dir / "render.yaml",
        repo_dir / "render.yml",
        repo_dir / "Procfile",
        repo_dir / "src" / "main.py",
        repo_dir / "main.py",
    ]
    return [path for path in candidates if path.exists()]


def _uvicorn_serving_present(repo_dir: Path) -> bool:
    patterns = [
        r"\buvicorn\s+(?:src\.api:app|api:app)\b",
        r"[\[\(\{,\s\"']uvicorn[\"']\s*[, \]]+[\s\S]{0,200}[\"'](?:src\.api:app|api:app)[\"']",
        r"\bconda\s+run\b[\s\S]{0,200}\buvicorn\b[\s\S]{0,200}\b(?:src\.api:app|api:app)\b",
        r"\bgunicorn\b[\s\S]{0,200}\buvicorn\.workers\.[\w]+worker\b[\s\S]{0,200}\b(?:src\.api:app|api:app)\b",
    ]
    for path in _api_serving_signal_files(repo_dir):
        if path.suffix == ".py":
            try:
                tree = ast.parse(read_text(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "uvicorn"
                    and node.func.attr == "run"
                ):
                    return True
            continue

        stripped = _strip_non_code_comments(read_text(path))
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns):
            return True
    return False


def scan_api_serving(repo_dir: Path) -> dict[str, Any]:
    evidence = {
        "api_fastapi_app_present": 0,
        "api_pydantic_contract_present": 0,
        "api_health_endpoint_present": 0,
        "api_predict_endpoint_present": 0,
        "api_uvicorn_serving_present": 0,
        "api_predict_calls_inference_logic": 0,
        "api_serving_cap_reason": "",
    }

    api_path, tree = _load_api_module(repo_dir)
    if api_path is None or tree is None:
        evidence["api_uvicorn_serving_present"] = int(_uvicorn_serving_present(repo_dir))
        evidence["api_serving_cap_reason"] = "no_fastapi_app"
        return evidence

    evidence["api_fastapi_app_present"] = int(_fastapi_app_present(tree))
    health_handlers = _find_route_handlers(tree, "/health")
    predict_handlers = _find_route_handlers(tree, "/predict")
    evidence["api_health_endpoint_present"] = int(bool(health_handlers))
    evidence["api_predict_endpoint_present"] = int(bool(predict_handlers))

    base_model_names = _base_model_subclass_names(tree)
    base_model_names.update(_local_imported_base_model_names(repo_dir, api_path, tree))
    evidence["api_pydantic_contract_present"] = int(
        _predict_uses_pydantic_contract(predict_handlers, base_model_names)
    )

    infer_function_names, infer_module_aliases = _infer_import_signals(tree)
    evidence["api_predict_calls_inference_logic"] = int(
        _predict_calls_inference_logic(
            predict_handlers,
            infer_function_names,
            infer_module_aliases,
        )
    )
    evidence["api_uvicorn_serving_present"] = int(_uvicorn_serving_present(repo_dir))

    if not evidence["api_fastapi_app_present"]:
        evidence["api_serving_cap_reason"] = "no_fastapi_app"
    elif not evidence["api_predict_endpoint_present"]:
        evidence["api_serving_cap_reason"] = "missing_predict_endpoint"
    elif not evidence["api_predict_calls_inference_logic"]:
        evidence["api_serving_cap_reason"] = "predict_without_inference_logic"

    return evidence


def _production_signal_files(repo_dir: Path) -> list[Path]:
    candidates = [
        repo_dir / "Dockerfile",
        repo_dir / "render.yaml",
        repo_dir / "render.yml",
        repo_dir / "config.yaml",
        repo_dir / "config.yml",
    ]

    workflow_dir = repo_dir / ".github" / "workflows"
    if workflow_dir.exists():
        candidates.extend(sorted(workflow_dir.glob("*.yml")))
        candidates.extend(sorted(workflow_dir.glob("*.yaml")))

    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            existing.append(path)
            seen.add(path)
    return existing


def _production_source_signal(text: str) -> bool:
    patterns = [
        r"(?im)\bmodel_source\s*[:=]\s*[\"']?wandb[\"']?\b",
        r"(?im)\bsource\s*:\s*wandb\b",
        r"(?im)\binference\s*:\s*\n(?:[ \t].*\n)*?[ \t]+source\s*:\s*wandb\b",
        r"(?im)\benv\s+model_source\s*=\s*[\"']?wandb[\"']?\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _production_prod_alias_signal(text: str) -> bool:
    patterns = [
        r"(?im)\bwandb_model_alias\s*[:=]\s*[\"']?prod[\"']?\b",
        r"(?im)\bmodel_alias\s*:\s*[\"']?prod[\"']?\b",
        r"(?im)\bproduction_alias\s*:\s*[\"']?prod[\"']?\b",
        r"(?im)\bartifact_alias\s*:\s*[\"']?prod[\"']?\b",
        r"(?im)\benv\s+wandb_model_alias\s*=\s*[\"']?prod[\"']?\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _production_local_signal(text: str) -> bool:
    patterns = [
        r"(?im)\bmodel_source\s*[:=]\s*[\"']?local[\"']?\b",
        r"(?im)\bsource\s*:\s*local\b",
        r"(?im)\binference\s*:\s*\n(?:[ \t].*\n)*?[ \t]+source\s*:\s*local\b",
        r"(?im)\benv\s+model_source\s*=\s*[\"']?local[\"']?\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _deployment_api_start_signal(text: str) -> bool:
    patterns = [
        r"(?im)\bstartcommand\s*:\s*.*(?:uvicorn|gunicorn).*(?:src\.)?api:app\b",
        r"(?im)\bcmd\b.*(?:uvicorn|gunicorn).*(?:src\.)?api:app\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _wandb_runtime_env_signal(text: str) -> bool:
    keys = ["wandb_api_key", "wandb_entity", "wandb_project"]
    return all(key in text.lower() for key in keys)


def _runtime_registry_prod_wiring_signal(combined_cleaned: str, config_text: str) -> bool:
    source_switch_present = bool(
        re.search(r"os\.(?:getenv|environ\.get)\(\s*['\"]model_source['\"]", combined_cleaned)
        and re.search(r"if\s+model_source\s*==\s*['\"]wandb['\"]", combined_cleaned)
    )
    return source_switch_present and _prod_alias_signal(combined_cleaned, config_text)


def _runtime_registry_first_prod_signal(combined_cleaned: str, config_text: str) -> bool:
    registry_first_patterns = [
        r"try:\s*[\s\S]{0,400}use_artifact\s*\(",
        r"try:\s*[\s\S]{0,400}wandb\.api\(\)\s*[\s\S]{0,200}artifact\s*\(",
        r"attempting to load model from w&b artifact \(prod\)",
    ]
    registry_first = any(re.search(pattern, combined_cleaned, re.I) for pattern in registry_first_patterns)
    local_fallback = bool(re.search(r"fall(?:ing)? back to local", combined_cleaned, re.I))
    return registry_first and local_fallback and _prod_alias_signal(combined_cleaned, config_text)


def _production_registry_selected(
    repo_dir: Path,
    combined_cleaned: str,
    config_text: str,
) -> bool:
    env_keys_used = {
        key
        for key in ["model_source", "wandb_model_alias", "artifact_alias", "production_alias", "model_alias"]
        if key in combined_cleaned
    }

    source_selected = False
    alias_selected = False
    local_selected = False
    has_runtime_container = False
    deployment_starts_api = False
    deployment_has_wandb_runtime = False

    for path in _production_signal_files(repo_dir):
        sanitized_text = _strip_non_code_comments(read_text(path))
        lower_text = sanitized_text.lower()

        if path.name == ".env.example":
            if not env_keys_used:
                continue
            if ".env.example" not in combined_cleaned and ".env.example" not in lower_text:
                continue

        if "workflow" in path.parts and not env_keys_used.intersection({"model_source", "wandb_model_alias", "artifact_alias", "production_alias", "model_alias"}):
            continue

        if path.name == "Dockerfile":
            has_runtime_container = True
        if _production_source_signal(sanitized_text):
            source_selected = True
        if _production_prod_alias_signal(sanitized_text):
            alias_selected = True
        if path.name != "ci.yml" and _production_local_signal(sanitized_text):
            local_selected = True
        if path.suffix in {".yaml", ".yml"} and _deployment_api_start_signal(sanitized_text):
            deployment_starts_api = True
        if path.suffix in {".yaml", ".yml"} and _wandb_runtime_env_signal(sanitized_text):
            deployment_has_wandb_runtime = True

    if not alias_selected and _production_prod_alias_signal(config_text):
        alias_selected = True

    if not source_selected and has_runtime_container and not local_selected and _runtime_registry_prod_wiring_signal(combined_cleaned, config_text):
        source_selected = True
        alias_selected = True

    if (
        not source_selected
        and deployment_starts_api
        and deployment_has_wandb_runtime
        and _runtime_registry_first_prod_signal(combined_cleaned, config_text)
    ):
        source_selected = True
        alias_selected = True

    return source_selected and alias_selected


def scan_model_registry(repo_dir: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "reg_serving_path_registry_backed": 0,
        "reg_serving_prod_alias_used": 0,
        "reg_production_registry_selected": 0,
        "reg_serving_local_fallback_present": 0,
        "reg_serving_local_only": 0,
        "reg_model_registry_cap_reason": "",
    }

    api_path = _find_api_path(repo_dir)
    if not api_path or not api_path.exists():
        evidence["reg_model_registry_cap_reason"] = "no_registry_serving_path"
        return evidence

    relevant_paths = [api_path, *_api_helper_paths(repo_dir, api_path)]
    relevant_payloads: list[tuple[Path, str, str]] = []
    for path in relevant_paths:
        raw_text = read_text(path)
        relevant_payloads.append((path, raw_text, _clean_python_for_detection(raw_text)))

    api_raw_text = relevant_payloads[0][1]
    combined_cleaned = "\n".join(cleaned for _, _, cleaned in relevant_payloads)
    config_payload = _repo_config_payload(repo_dir)
    config_text = _repo_config_text(repo_dir)

    registry_any = any(_registry_serving_signal(cleaned) for _, _, cleaned in relevant_payloads)
    local_any = any(_local_serving_signal(cleaned) for _, _, cleaned in relevant_payloads)
    default_source = _default_source_from_code_and_config(combined_cleaned, config_payload)

    registry_first_fallback = bool(
        registry_any
        and local_any
        and "except" in api_raw_text
        and "try" in api_raw_text
        and "wandb" in combined_cleaned
        and "local" in combined_cleaned
    )

    if default_source == "wandb" and registry_any:
        evidence["reg_serving_path_registry_backed"] = 1
    elif registry_first_fallback:
        evidence["reg_serving_path_registry_backed"] = 1
        evidence["reg_serving_local_fallback_present"] = 1
    elif registry_any and not local_any:
        evidence["reg_serving_path_registry_backed"] = 1
    elif default_source == "local" and registry_any:
        evidence["reg_serving_local_fallback_present"] = 1
    elif local_any and not registry_any:
        evidence["reg_serving_local_only"] = 1

    if default_source == "local" and registry_any and local_any:
        evidence["reg_serving_local_fallback_present"] = 1

    if evidence["reg_serving_local_only"] == 0 and default_source == "local" and local_any and not registry_any:
        evidence["reg_serving_local_only"] = 1

    if evidence["reg_serving_path_registry_backed"]:
        active_registry_text = combined_cleaned
        evidence["reg_serving_prod_alias_used"] = int(_prod_alias_signal(active_registry_text, config_text))
        evidence["reg_production_registry_selected"] = int(
            _production_registry_selected(repo_dir, combined_cleaned, config_text)
        )

    cap_reasons: list[str] = []
    if registry_any and len(relevant_paths) == 1 and api_path == relevant_paths[0]:
        pass
    elif not registry_any and any(_registry_serving_signal(_clean_python_for_detection(read_text(path))) for path in production_python_files(repo_dir) if path not in relevant_paths):
        cap_reasons.append("registry_helper_not_used")
    if not evidence["reg_serving_path_registry_backed"]:
        cap_reasons.append("no_registry_serving_path")
    if evidence["reg_serving_path_registry_backed"] and not evidence["reg_serving_prod_alias_used"]:
        cap_reasons.append("missing_prod_alias")
    if evidence["reg_serving_local_fallback_present"] and not evidence["reg_production_registry_selected"]:
        cap_reasons.append("default_or_fallback_local")
    if evidence["reg_serving_local_only"]:
        cap_reasons.append("local_only")
    evidence["reg_model_registry_cap_reason"] = "|".join(cap_reasons)
    return evidence

def _extract_import_targets(tree: ast.AST) -> list[str]:
    targets: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.lower()
                if module_name.startswith("src."):
                    targets.append(module_name)
                    if alias.asname:
                        targets.append(alias.asname.lower())
        elif isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").lower()
            if module_name.startswith("src"):
                targets.append(module_name)
                for alias in node.names:
                    if alias.name:
                        targets.append(alias.name.lower())
                    if alias.asname:
                        targets.append(alias.asname.lower())

    return targets


def _extract_called_names(tree: ast.AST) -> list[str]:
    called: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.append(func.id.lower())
            elif isinstance(func, ast.Attribute):
                called.append(func.attr.lower())

    return called


def scan_main_orchestration(repo_dir: Path) -> dict[str, Any]:
    main_path = _find_main_path(repo_dir)
    if not main_path:
        return {
            "main_core_import_hits": 0,
            "main_core_call_hits": 0,
            "main_orchestration_signal": 0,
            "main_imported_core_modules": "",
            "main_called_pipeline_terms": "",
        }

    text = read_text(main_path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "main_core_import_hits": 0,
            "main_core_call_hits": 0,
            "main_orchestration_signal": 0,
            "main_imported_core_modules": "",
            "main_called_pipeline_terms": "",
        }

    import_targets = _extract_import_targets(tree)
    called_names = _extract_called_names(tree)

    core_hits: list[str] = []
    for module_name, patterns in CORE_MODULE_MAP.items():
        if module_name == "main":
            continue
        if any(any(pattern in target for pattern in patterns) for target in import_targets):
            core_hits.append(module_name)

    pipeline_terms = [
        "load",
        "read",
        "ingest",
        "clean",
        "validate",
        "preprocess",
        "transform",
        "feature",
        "train",
        "fit",
        "evaluate",
        "predict",
        "infer",
        "save",
        "dump",
    ]

    call_hits = sorted({term for term in pipeline_terms if any(term in call for call in called_names)})

    return {
        "main_core_import_hits": len(core_hits),
        "main_core_call_hits": len(call_hits),
        "main_orchestration_signal": int(len(core_hits) >= 2 and len(call_hits) >= 2),
        "main_imported_core_modules": " | ".join(sorted(core_hits)),
        "main_called_pipeline_terms": " | ".join(call_hits),
    }


def scan_module_stubness(repo_dir: Path) -> dict[str, Any]:
    stub_candidates = 0
    tiny_module_count = 0
    real_module_count = 0

    excluded_names = {"__init__", "main", "setup", "conftest"}

    for path in python_files(repo_dir):
        stem = path.stem.lower()
        if stem in excluded_names:
            continue

        text = read_text(path)
        if not text.strip():
            continue

        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        function_defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        class_defs = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        executable_defs = len(function_defs) + len(class_defs)

        line_count = len([line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")])

        if executable_defs > 0:
            stub_candidates += 1

            only_pass = True
            for fn in function_defs:
                fn_body = [
                    n for n in fn.body
                    if not (
                        isinstance(n, ast.Expr)
                        and isinstance(getattr(n, "value", None), ast.Constant)
                        and isinstance(n.value.value, str)
                    )
                ]
                if not fn_body:
                    continue
                if not all(isinstance(n, (ast.Pass, ast.Expr)) for n in fn_body):
                    only_pass = False
                    break

            if line_count <= 12 or only_pass:
                tiny_module_count += 1
            else:
                real_module_count += 1

    stub_ratio = round(tiny_module_count / max(stub_candidates, 1), 3) if stub_candidates else 0.0

    return {
        "stub_candidate_module_count": stub_candidates,
        "tiny_stub_module_count": tiny_module_count,
        "real_module_count": real_module_count,
        "tiny_stub_ratio": stub_ratio,
    }





def derive_modularization_evidence(
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    stub_threshold = safe_float(cfg_get(config, "scoring.modularization.stub_penalty_threshold", 0.5))

    main_entry_signal = int(
        evidence.get("main_py_present", 0)
        and evidence.get("main_guard_present", 0)
        and evidence.get("main_function_present", 0)
    )

    orchestrated_module_count = int(evidence.get("main_core_import_hits", 0) or 0)
    imported_modules = set(filter(None, (evidence.get("main_imported_core_modules", "") or "").split(" | ")))

    # Validation is a required pipeline stage for this course.
    # If main.py does not import a validate stage, the pipeline is not fully orchestrated.
    if "validate" not in imported_modules and orchestrated_module_count > 0:
        orchestrated_module_count -= 1

    stub_penalty_flag = int(
        float(evidence.get("tiny_stub_ratio", 0.0) or 0.0) >= stub_threshold
    )

    return {
        "main_entry_signal": main_entry_signal,
        "core_module_count": int(evidence.get("core_module_count", 0) or 0),
        "orchestrated_module_count": orchestrated_module_count,
        "stub_penalty_flag": stub_penalty_flag,
    }


def slim_evidence_for_selected_dimensions(
    evidence: dict[str, Any],
    selected_dimensions: set[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    derived = dict(evidence)
    if "modularization" in selected_dimensions:
        derived.update(derive_modularization_evidence(derived, config))

    keep = set(EVIDENCE_METADATA_FIELDS)
    for dimension in selected_dimensions:
        keep.update(EVIDENCE_FIELDS_BY_DIMENSION.get(dimension, []))

    return {key: derived[key] for key in derived if key in keep}


def scan_dependency_files(repo_dir: Path) -> dict[str, Any]:
    env_candidates = [repo_dir / "environment.yml", repo_dir / "environment.yaml"]
    conda_candidates = [repo_dir / "conda.yml", repo_dir / "conda.yaml"]
    requirements_candidates = [repo_dir / "requirements.txt", repo_dir / "pyproject.toml", repo_dir / "Pipfile"]
    config_candidates = [repo_dir / "config.yaml", repo_dir / "config.yml", repo_dir / ".env"]

    env_path = next((path for path in env_candidates if path.exists()), None)
    conda_path = next((path for path in conda_candidates if path.exists()), None)
    req_path = next((path for path in requirements_candidates if path.exists()), None)
    config_path = next((path for path in config_candidates if path.exists()), None)

    dependency_versions = 0
    if env_path:
        text = read_text(env_path)
        dependency_versions = int(bool(re.search(r"[<>=]=?", text)))
    elif conda_path:
        text = read_text(conda_path)
        dependency_versions = int(bool(re.search(r"[<>=]=?", text)))
    elif req_path:
        text = read_text(req_path)
        dependency_versions = int(bool(re.search(r"==|>=|<=|~=", text)))

    return {
        "environment_yml_present": int(env_path is not None),
        "conda_yml_present": int(conda_path is not None),
        "requirements_txt_present": int(req_path is not None),
        "config_file_present": int(config_path is not None),
        "dependency_versions_declared": dependency_versions,
        "environment_file": str(env_path.relative_to(repo_dir)) if env_path else "",
        "conda_file": str(conda_path.relative_to(repo_dir)) if conda_path else "",
        "requirements_file": str(req_path.relative_to(repo_dir)) if req_path else "",
        "config_file": str(config_path.relative_to(repo_dir)) if config_path else "",
    }



def tool_available(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def run_ruff(repo_dir: Path) -> dict[str, Any]:
    if not tool_available("ruff"):
        return {"ruff_available": 0, "ruff_issue_count": None, "ruff_returncode": None}

    result = run_command(["ruff", "check", ".", "--output-format", "json"], cwd=repo_dir, timeout=240)
    issue_count = None

    if result["stdout"].strip():
        try:
            payload = json.loads(result["stdout"])
            issue_count = len(payload)
        except json.JSONDecodeError:
            issue_count = None
    elif result["ok"]:
        issue_count = 0

    return {
        "ruff_available": 1,
        "ruff_issue_count": issue_count,
        "ruff_returncode": result["returncode"],
    }


PYLINT_SCORE_RE = re.compile(r"rated at\s+([-0-9.]+)/10")


def run_pylint(repo_dir: Path) -> dict[str, Any]:
    if not tool_available("pylint"):
        return {"pylint_available": 0, "pylint_score": None, "pylint_returncode": None}

    target = "src" if (repo_dir / "src").exists() else "."
    cmd = ["pylint", target, "--rcfile", "/home/idiazl/2026_MLOps/Grades/.pylintrc"]
    result = run_command(cmd, cwd=repo_dir, timeout=300)
    match = PYLINT_SCORE_RE.search(result["stdout"] + "\n" + result["stderr"])
    score = float(match.group(1)) if match else None

    return {
        "pylint_available": 1,
        "pylint_score": score,
        "pylint_returncode": result["returncode"],
    }


def run_radon(repo_dir: Path) -> dict[str, Any]:
    if not tool_available("radon"):
        return {"radon_available": 0, "radon_average_cc": None, "radon_files_analyzed": None}

    target = "src" if (repo_dir / "src").exists() else "."
    result = run_command(["radon", "cc", target, "-j", "-s"], cwd=repo_dir, timeout=240)

    if not result["stdout"].strip():
        return {"radon_available": 1, "radon_average_cc": None, "radon_files_analyzed": 0}

    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"radon_available": 1, "radon_average_cc": None, "radon_files_analyzed": None}

    complexities: list[float] = []
    for records in payload.values():
        for item in records:
            complexities.append(float(item.get("complexity", 0)))

    average = sum(complexities) / len(complexities) if complexities else 0.0
    return {
        "radon_available": 1,
        "radon_average_cc": round(average, 3),
        "radon_files_analyzed": len(payload),
    }


def run_pytest(repo_dir: Path) -> dict[str, Any]:
    tests_dir = repo_dir / "tests"
    test_files = [p for p in tests_dir.rglob("test_*.py")] if tests_dir.exists() else []

    def empty_payload(pytest_available: int, pytest_ran: int) -> dict[str, Any]:
        return {
            "pytest_available": pytest_available,
            "pytest_ran": pytest_ran,
            "pytest_passed": 0,
            "pytest_pass_count": 0,
            "pytest_fail_count": 0,
            "pytest_error_count": 0,
            "pytest_total_count": 0,
            "pytest_pass_rate": None,
            "coverage_pct": None,
            "tests_collected": 0,
            "pytest_returncode": None,
            "pytest_timeout": 0,
            "pytest_import_error": 0,
            "pytest_collection_error": 0,
            "pytest_environment_warning": 0,
            "pytest_stdout_tail": "",
            "pytest_stderr_tail": "",
            "pytest_command": "",
        }

    if not test_files:
        return empty_payload(int(tool_available("pytest")), 0)

    if not tool_available("pytest"):
        payload = empty_payload(0, 0)
        payload["tests_collected"] = None
        return payload

    with tempfile.TemporaryDirectory(prefix="grader_pytest_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        temp_home = tmp_root_path / "home"
        temp_cache = tmp_root_path / "cache"
        temp_pycache = tmp_root_path / "pycache"
        temp_basetemp = tmp_root_path / "pytest_tmp"
        coverage_json = tmp_root_path / "coverage.json"
        junit_xml = tmp_root_path / "pytest_junit.xml"
        coverage_file = tmp_root_path / ".coverage"

        temp_home.mkdir(parents=True, exist_ok=True)
        temp_cache.mkdir(parents=True, exist_ok=True)
        temp_pycache.mkdir(parents=True, exist_ok=True)
        temp_basetemp.mkdir(parents=True, exist_ok=True)

        cov_target = "src" if (repo_dir / "src").exists() else "."

        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "--basetemp",
            str(temp_basetemp),
            "-p",
            "no:cacheprovider",
            "-p",
            "pytest_cov",
            f"--cov={cov_target}",
            f"--cov-report=json:{coverage_json}",
            f"--junitxml={junit_xml}",
        ]

        env = {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": str(temp_pycache),
            "HOME": str(temp_home),
            "XDG_CACHE_HOME": str(temp_cache),
            "MPLCONFIGDIR": str(temp_cache),
            "COVERAGE_FILE": str(coverage_file),
        }

        result = run_command(command, cwd=repo_dir, timeout=PYTEST_TIMEOUT_SECONDS, env=env)
        combined = (result["stdout"] or "") + "\n" + (result["stderr"] or "")
        tests_collected = None
        pass_count = 0
        fail_count = 0
        error_count = 0
        skipped_count = 0

        if junit_xml.exists():
            try:
                root = ET.fromstring(read_text(junit_xml))
                suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")

                tests_collected = sum(int(float(s.attrib.get("tests", 0))) for s in suites)
                fail_count = sum(int(float(s.attrib.get("failures", 0))) for s in suites)
                error_count = sum(int(float(s.attrib.get("errors", 0))) for s in suites)
                skipped_count = sum(int(float(s.attrib.get("skipped", 0))) for s in suites)

                pass_count = max(tests_collected - fail_count - error_count - skipped_count, 0)
            except Exception:
                tests_collected = None
                pass_count = 0
                fail_count = 0
                error_count = 0
                skipped_count = 0

        if tests_collected is None:
            collected_match = re.search(r"collected\s+(\d+)\s+items?", combined)
            tests_collected = int(collected_match.group(1)) if collected_match else None

            summary_matches = re.findall(
                r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)",
                combined,
            )

            summary_counts = {}
            for count, label in summary_matches:
                summary_counts[label] = summary_counts.get(label, 0) + int(count)

            pass_count = summary_counts.get("passed", 0)
            fail_count = summary_counts.get("failed", 0)
            error_count = summary_counts.get("error", 0) + summary_counts.get("errors", 0)
            skipped_count = summary_counts.get("skipped", 0)

        total_count = pass_count + fail_count + error_count
        pass_rate = round(pass_count / total_count, 6) if total_count > 0 else None

        if tests_collected is None and (total_count + skipped_count) > 0:
            tests_collected = total_count + skipped_count

        coverage_pct = None
        if coverage_json.exists():
            try:
                payload = json.loads(read_text(coverage_json))
                totals = payload.get("totals", {})
                coverage_pct = safe_float(totals.get("percent_covered"), None)
            except json.JSONDecodeError:
                coverage_pct = None

        import_error = int(bool(re.search(r"(ModuleNotFoundError|ImportError):", combined)))
        collection_error = int(bool(re.search(r"ERROR\s+collecting|collected\s+0\s+items\s*/\s*\d+\s+errors", combined)))
        timeout_flag = int(result["returncode"] == 124)
        environment_warning = int(import_error or timeout_flag)

        return {
            "pytest_available": 1,
            "pytest_ran": 1,
            "pytest_passed": int(result["returncode"] == 0),
            "pytest_pass_count": pass_count,
            "pytest_fail_count": fail_count,
            "pytest_error_count": error_count,
            "pytest_total_count": total_count,
            "pytest_pass_rate": pass_rate,
            "coverage_pct": coverage_pct,
            "tests_collected": tests_collected,
            "pytest_returncode": result["returncode"],
            "pytest_timeout": timeout_flag,
            "pytest_import_error": import_error,
            "pytest_collection_error": collection_error,
            "pytest_environment_warning": environment_warning,
            "pytest_stdout_tail": "\n".join((result["stdout"] or "").splitlines()[-10:]),
            "pytest_stderr_tail": "\n".join((result["stderr"] or "").splitlines()[-10:]),
            "pytest_command": shlex.join(command),
        }


EDGE_CASE_PATTERNS = ["empty", "missing", "invalid", "nan", "null", "error", "raise", "edge", "shape", "type"]


def analyze_tests(repo_dir: Path) -> dict[str, Any]:
    tests_dir = repo_dir / "tests"
    test_files = [p for p in tests_dir.rglob("test_*.py")] if tests_dir.exists() else []
    edge_case_hits = 0
    total_asserts = 0

    for path in test_files:
        text = read_text(path).lower()
        total_asserts += len(re.findall(r"\bassert\b", text))
        edge_case_hits += sum(pattern in text for pattern in EDGE_CASE_PATTERNS)

    return {
        "test_file_count": len(test_files),
        "test_assert_count": total_asserts,
        "edge_case_keyword_hits": edge_case_hits,
    }


def scan_git_history(repo_dir: Path) -> dict[str, Any]:
    result = run_command(["git", "log", "--pretty=format:%an|%ae"], cwd=repo_dir, timeout=180)
    authors = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    counter = Counter(authors)
    total = sum(counter.values())
    top_share = (counter.most_common(1)[0][1] / total) if total else 0.0
    evenness = 1.0 - top_share if total else 0.0

    return {
        "git_commit_count_before_cutoff": total,
        "git_unique_contributors": len(counter),
        "git_top_contributor_share": round(top_share, 3),
        "git_contribution_evenness": round(evenness, 3),
    }


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_get(url: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    response = requests.get(url, headers=github_headers(), params=params, timeout=30)
    meta = {
        "status_code": response.status_code,
        "remaining": response.headers.get("X-RateLimit-Remaining"),
    }
    if response.status_code >= 400:
        return None, meta
    try:
        return response.json(), meta
    except ValueError:
        return None, meta


def list_pull_requests(owner: str, repo: str, cutoff_iso: str) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    page = 1

    while page <= 5:
        payload, _ = github_get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls",
            {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100, "page": page},
        )
        if not isinstance(payload, list):
            break
        if not payload:
            break

        for pr in payload:
            created_at = pr.get("created_at")
            if created_at and created_at <= cutoff_iso:
                collected.append(pr)

        if len(payload) < 100:
            break
        page += 1

    return collected


def list_reviews(owner: str, repo: str, pr_number: int, cutoff_iso: str) -> list[dict[str, Any]]:
    payload, _ = github_get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews")
    if not isinstance(payload, list):
        return []

    filtered: list[dict[str, Any]] = []
    for review in payload:
        submitted_at = review.get("submitted_at")
        if submitted_at and submitted_at <= cutoff_iso:
            filtered.append(review)

    return filtered


def scan_github_workflow(
    repo: RepoSpec,
    cutoff_str: str,
    timezone_name: str,
) -> dict[str, Any]:
    if repo.repo_url.startswith("local://"):
        return {
            "github_prs_found": 0,
            "github_unique_pr_authors": 0,
            "github_unique_reviewers": 0,
            "github_approvals_found": 0,
            "github_self_merge_ratio": None,
            "github_api_used": 0,
            "github_api_authenticated": int(bool(os.getenv("GITHUB_TOKEN", "").strip())),
        }

    owner, repo_name = parse_owner_repo(repo.repo_url)

    cutoff_dt = datetime.strptime(cutoff_str, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=ZoneInfo(timezone_name)
    )
    cutoff_iso = cutoff_dt.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    prs = list_pull_requests(owner, repo_name, cutoff_iso)
    if not prs:
        return {
            "github_prs_found": 0,
            "github_unique_pr_authors": 0,
            "github_unique_reviewers": 0,
            "github_approvals_found": 0,
            "github_self_merge_ratio": None,
            "github_api_used": 1,
            "github_api_authenticated": int(bool(os.getenv("GITHUB_TOKEN", "").strip())),
        }

    pr_authors: set[str] = set()
    reviewers: set[str] = set()
    approvals_found = 0
    self_merges = 0

    for pr in prs:
        author_login = (pr.get("user") or {}).get("login")
        if author_login:
            pr_authors.add(author_login.lower())

        merged_by = (pr.get("merged_by") or {}).get("login")
        if merged_by and author_login and merged_by.lower() == author_login.lower():
            self_merges += 1

        number = pr.get("number")
        if number is not None:
            reviews = list_reviews(owner, repo_name, int(number), cutoff_iso)
            for review in reviews:
                reviewer_login = (review.get("user") or {}).get("login")
                if reviewer_login:
                    reviewers.add(reviewer_login.lower())
                if review.get("state") == "APPROVED":
                    approvals_found += 1

    return {
        "github_prs_found": len(prs),
        "github_unique_pr_authors": len(pr_authors),
        "github_unique_reviewers": len(reviewers),
        "github_approvals_found": approvals_found,
        "github_self_merge_ratio": round(self_merges / max(len(prs), 1), 3),
        "github_api_used": 1,
        "github_api_authenticated": int(bool(os.getenv("GITHUB_TOKEN", "").strip())),
    }


def collect_evidence(
    repo_dir: Path,
    repo: RepoSpec,
    cutoff_str: str,
    timezone_name: str,
    selected_dimensions: set[str],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}

    structural_needed = {
        "modularization",
        "documentation",
        "dependencies",
        "config_reproducibility",
        "error_handling",
        "artifacting",
        "pipeline",
        "testing",
    }

    if selected_dimensions & structural_needed:
        evidence.update(scan_repo_structure(repo_dir))
        evidence.update(scan_readme(repo_dir))
        evidence.update(scan_notebooks(repo_dir))
        evidence.update(scan_python_files(repo_dir))
        evidence.update(scan_main_entry(repo_dir))
        evidence.update(scan_pipeline_modules(repo_dir))
        evidence.update(scan_main_orchestration(repo_dir))
        evidence.update(scan_module_stubness(repo_dir))
        evidence.update(scan_artifact_contract(repo_dir))
        evidence.update(analyze_tests(repo_dir))

    if (
        _is_selected(selected_dimensions, "dependencies")
        or _is_selected(selected_dimensions, "config_reproducibility")
    ):
        evidence.update(scan_dependency_files(repo_dir))

    if _is_selected(selected_dimensions, "config_reproducibility"):
        evidence.update(scan_config_reproducibility(repo_dir))

    if _is_selected(selected_dimensions, "security_secrets"):
        evidence.update(scan_security_secrets(repo_dir))

    if _is_selected(selected_dimensions, "logging_observability"):
        evidence.update(scan_logging_observability(repo_dir))

    if _is_selected(selected_dimensions, "experiment_tracking"):
        evidence.update(scan_experiment_tracking(repo_dir))

    if _is_selected(selected_dimensions, "model_registry"):
        evidence.update(scan_model_registry(repo_dir))

    if _is_selected(selected_dimensions, "api_serving"):
        evidence.update(scan_api_serving(repo_dir))

    if _is_selected(selected_dimensions, "code_quality"):
        evidence.update(run_ruff(repo_dir))
        evidence.update(run_pylint(repo_dir))
        evidence.update(run_radon(repo_dir))

    if _is_selected(selected_dimensions, "testing"):
        evidence.update(run_pytest(repo_dir))

    if _is_selected(selected_dimensions, "version_control"):
        evidence.update(scan_git_history(repo_dir))
        evidence.update(
            scan_github_workflow(
                repo=repo,
                cutoff_str=cutoff_str,
                timezone_name=timezone_name,
            )
        )

    return evidence


def compute_proxy_scores(
    evidence: dict[str, Any],
    selected_dimensions: set[str],
    config: dict[str, Any],
) -> dict[str, float]:
    scores: dict[str, float] = {}

    if _is_selected(selected_dimensions, "modularization"):
        section = cfg_get(config, "scoring.modularization", {})
        main_entry_signal = int(evidence.get("main_entry_signal", 0) or 0)
        core_module_count = int(evidence.get("core_module_count", 0) or 0)
        orchestrated_module_count = int(evidence.get("orchestrated_module_count", 0) or 0)
        stub_penalty_flag = int(evidence.get("stub_penalty_flag", 0) or 0)

        score = 0.0
        score += safe_float(cfg_get(section, "main_entry_signal", 0.0)) if main_entry_signal else 0.0

        core_points = 0.0
        for rule in cfg_get(section, "core_module_thresholds", []):
            if core_module_count >= int(rule.get("min", 0)):
                core_points = max(core_points, safe_float(rule.get("points", 0.0)))
        score += core_points

        orchestration_points = 0.0
        for rule in cfg_get(section, "orchestrated_module_thresholds", []):
            if orchestrated_module_count >= int(rule.get("min", 0)):
                orchestration_points = max(orchestration_points, safe_float(rule.get("points", 0.0)))
        score += orchestration_points

        if stub_penalty_flag:
            score -= safe_float(cfg_get(section, "stub_penalty", 0.0))

        scores[DIMENSION_TO_SCORE_COLUMN["modularization"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "code_quality"):
        section = cfg_get(config, "scoring.code_quality", {})
        ruff_value = evidence.get("ruff_issue_count", None)
        ruff_issues = 999.0 if ruff_value is None else float(ruff_value)

        pylint_score = float(evidence.get("pylint_score", 0.0) or 0.0)
        radon_avg = float(evidence.get("radon_average_cc", 20.0) or 20.0)

        score = safe_float(cfg_get(section, "base_score", 10.0))
        score -= min(safe_float(cfg_get(section, "ruff_cap", 0.0)), ruff_issues * safe_float(cfg_get(section, "ruff_per_issue", 0.0)))
        score -= max(0.0, (safe_float(cfg_get(section, "pylint_target", 0.0)) - pylint_score) * safe_float(cfg_get(section, "pylint_gap_weight", 0.0)))
        score -= max(0.0, (radon_avg - safe_float(cfg_get(section, "radon_baseline", 0.0))) * safe_float(cfg_get(section, "radon_gap_weight", 0.0)))
        scores[DIMENSION_TO_SCORE_COLUMN["code_quality"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "documentation"):
        qualitative_only = bool(cfg_get(config, "scoring.documentation.qualitative_only", True))
        if not qualitative_only:
            section = cfg_get(config, "scoring.documentation", {})
            readme_present = float(evidence.get("readme_present", 0) or 0)

            score = 0.0
            score += safe_float(cfg_get(section, "readme_present", 0.0)) if readme_present else 0.0
            scores[DIMENSION_TO_SCORE_COLUMN["documentation"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "testing"):
        section = cfg_get(config, "scoring.testing", {})
        pytest_passed = int(evidence.get("pytest_passed", 0) or 0)
        pytest_total_count = int(evidence.get("pytest_total_count", 0) or 0)
        pytest_pass_rate = evidence.get("pytest_pass_rate")
        coverage_pct = evidence.get("coverage_pct")
        import_error = int(evidence.get("pytest_import_error", 0) or 0)
        timeout_error = int(evidence.get("pytest_timeout", 0) or 0)

        pytest_weight = safe_float(
            cfg_get(section, "pytest_pass_rate_weight", cfg_get(section, "pytest_passed", 0.0))
        )

        score = 0.0

        if pytest_total_count > 0 and isinstance(pytest_pass_rate, (int, float)):
            score += pytest_weight * float(pytest_pass_rate)
        elif pytest_passed:
            score += pytest_weight

        if isinstance(coverage_pct, (int, float)):
            score += min(
                safe_float(cfg_get(section, "coverage_cap", 0.0)),
                float(coverage_pct) / safe_float(cfg_get(section, "coverage_divisor", 1.0)),
            )

        if import_error or timeout_error:
            score = min(score, safe_float(cfg_get(section, "environment_cap", 10.0)))

        scores[DIMENSION_TO_SCORE_COLUMN["testing"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "dependencies"):
        section = cfg_get(config, "scoring.dependencies", {})
        env_present = int(evidence.get("environment_yml_present", 0) or 0)
        conda_present = int(evidence.get("conda_yml_present", 0) or 0)

        score = 0.0
        score += (
            safe_float(cfg_get(section, "env_or_conda_present", 0.0))
            if (env_present or conda_present)
            else 0.0
        )
        scores[DIMENSION_TO_SCORE_COLUMN["dependencies"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "config_reproducibility"):
        section = cfg_get(config, "scoring.config_reproducibility", {})

        signal_names = [
            "config_yaml_present",
            "config_yaml_has_keys",
            "environment_yml_present",
            "conda_lock_yml_present",
            "main_reads_config_signal",
            "env_contract_signal",
            "gitignore_excludes_env",
            "dotenv_usage_signal",
        ]

        score = 0.0
        for signal_name in signal_names:
            if int(evidence.get(signal_name, 0) or 0):
                score += safe_float(cfg_get(section, signal_name, 0.0))

        runtime_config_keys_found = int(evidence.get("runtime_config_keys_found", 0) or 0)
        if runtime_config_keys_found >= int(cfg_get(section, "runtime_config_keys_min", 0)):
            score += safe_float(cfg_get(section, "runtime_config_keys_points", 0.0))

        path_hits = int(evidence.get("code_hardcoded_path_hits", 0) or 0)
        hyperparam_hits = int(evidence.get("code_hardcoded_hyperparam_hits", 0) or 0)

        score -= min(
            safe_float(cfg_get(section, "hardcoded_path_penalty_cap", 0.0)),
            path_hits * safe_float(cfg_get(section, "hardcoded_path_penalty", 0.0)),
        )
        score -= min(
            safe_float(cfg_get(section, "hardcoded_hyperparam_penalty_cap", 0.0)),
            hyperparam_hits * safe_float(cfg_get(section, "hardcoded_hyperparam_penalty", 0.0)),
        )

        cap_reasons: list[str] = []
        if not int(evidence.get("config_yaml_present", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_config_yaml_cap", 10.0)))
            cap_reasons.append("missing_config_yaml")

        if not int(evidence.get("conda_lock_yml_present", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_conda_lock_cap", 10.0)))
            cap_reasons.append("missing_conda_lock")

        if int(evidence.get("secret_like_literal_hits", 0) or 0) > 0:
            score = min(score, safe_float(cfg_get(section, "secret_literal_cap", 10.0)))
            cap_reasons.append("secret_like_literal")

        evidence["config_reproducibility_cap_reason"] = "|".join(cap_reasons)
        scores[DIMENSION_TO_SCORE_COLUMN["config_reproducibility"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "security_secrets"):
        section = cfg_get(config, "scoring.security_secrets", {})

        score = 0.0

        if int(evidence.get("sec_gitignore_excludes_env", 0) or 0):
            score += safe_float(cfg_get(section, "sec_gitignore_excludes_env", 0.0))

        if int(evidence.get("sec_dockerignore_excludes_env", 0) or 0):
            score += safe_float(cfg_get(section, "sec_dockerignore_excludes_env", 0.0))

        if int(evidence.get("sec_secret_literal_hits", 0) or 0) == 0:
            score += safe_float(cfg_get(section, "sec_no_secret_literals", 0.0))

        cap_reasons: list[str] = []

        if int(evidence.get("sec_env_file_tracked_by_git", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "sec_tracked_env_cap", 2.0)))
            cap_reasons.append("env_tracked_in_git")

        if int(evidence.get("sec_secret_literal_hits", 0) or 0) > 0:
            score = min(score, safe_float(cfg_get(section, "sec_secret_literal_cap", 4.0)))
            cap_reasons.append("secret_literals_found")

        if evidence.get("sec_tracked_env_like_files", ""):
            score = min(score, safe_float(cfg_get(section, "sec_tracked_env_like_files_cap", 5.0)))
            cap_reasons.append("env_like_files_tracked")

        evidence["security_secrets_cap_reason"] = "|".join(cap_reasons)
        scores[DIMENSION_TO_SCORE_COLUMN["security_secrets"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "logging_observability"):
        section = cfg_get(config, "scoring.logging_observability", {})

        score = 0.0

        if int(evidence.get("log_print_free", 0) or 0):
            score += safe_float(cfg_get(section, "log_print_free", 0.0))

        if int(evidence.get("log_logger_module_present", 0) or 0):
            score += safe_float(cfg_get(section, "log_logger_module_present", 0.0))

        if int(evidence.get("log_file_handler_present", 0) or 0):
            score += safe_float(cfg_get(section, "log_file_handler_present", 0.0))

        if int(evidence.get("log_stream_handler_present", 0) or 0):
            score += safe_float(cfg_get(section, "log_stream_handler_present", 0.0))

        if int(evidence.get("log_dual_output_signal", 0) or 0):
            score += safe_float(cfg_get(section, "log_dual_output_signal", 0.0))

        if int(evidence.get("log_logfile_path_present", 0) or 0):
            score += safe_float(cfg_get(section, "log_logfile_path_present", 0.0))

        if int(evidence.get("log_logger_usage_signal", 0) or 0):
            score += safe_float(cfg_get(section, "log_logger_usage_signal", 0.0))

        cap_reasons: list[str] = []
        allowed_print_calls = int(cfg_get(section, "print_calls_allowed", 3))
        print_hits = int(evidence.get("log_print_statement_hits", 0) or 0)

        if print_hits > allowed_print_calls:
            score = min(score, safe_float(cfg_get(section, "too_many_print_calls_cap", 5.0)))
            cap_reasons.append("too_many_print_calls")

        if int(evidence.get("log_logger_module_fallback_used", 0) or 0):
            score -= safe_float(cfg_get(section, "fallback_logger_module_penalty", 1.0))

        if not int(evidence.get("log_logger_module_present", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_logger_module_cap", 4.0)))
            cap_reasons.append("missing_logger_module")

        if not int(evidence.get("log_dual_output_signal", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_dual_output_cap", 6.0)))
            cap_reasons.append("missing_dual_output")

        if not int(evidence.get("log_logger_usage_signal", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_usage_cap", 8.0)))
            cap_reasons.append("missing_usage")

        evidence["logging_observability_cap_reason"] = "|".join(cap_reasons)
        scores[DIMENSION_TO_SCORE_COLUMN["logging_observability"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "experiment_tracking"):
        section = cfg_get(config, "scoring.experiment_tracking", {})

        score = 0.0

        if int(evidence.get("wandb_import_present", 0) or 0):
            score += safe_float(cfg_get(section, "wandb_import_present", 0.0))
        if int(evidence.get("wandb_init_in_main", 0) or 0):
            score += safe_float(cfg_get(section, "wandb_init_in_main", 0.0))
        if int(evidence.get("wandb_config_logged", 0) or 0):
            score += safe_float(cfg_get(section, "wandb_config_logged", 0.0))
        if int(evidence.get("wandb_run_metadata_logged", 0) or 0):
            score += safe_float(cfg_get(section, "wandb_run_metadata_logged", 0.0))
        if int(evidence.get("wandb_eval_metrics_logged", 0) or 0):
            score += safe_float(cfg_get(section, "wandb_eval_metrics_logged", 0.0))
        if int(evidence.get("wandb_rich_eval_tracking_logged", 0) or 0):
            score += safe_float(cfg_get(section, "wandb_rich_eval_tracking_logged", 0.0))
        if int(evidence.get("wandb_model_artifact_logged", 0) or 0):
            score += safe_float(cfg_get(section, "wandb_model_artifact_logged", 0.0))

        cap_reasons: list[str] = []

        if not int(evidence.get("wandb_init_in_main", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_wandb_init_in_main_cap", 4.0)))
            cap_reasons.append("missing_wandb_init_in_main")

        if not int(evidence.get("wandb_eval_metrics_logged", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_eval_metrics_cap", 7.0)))
            cap_reasons.append("missing_eval_metrics")

        if not int(evidence.get("wandb_model_artifact_logged", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_model_artifact_cap", 8.0)))
            cap_reasons.append("missing_model_artifact")

        evidence["wandb_cap_reason"] = "|".join(cap_reasons)
        scores[DIMENSION_TO_SCORE_COLUMN["experiment_tracking"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "model_registry"):
        section = cfg_get(config, "scoring.model_registry", {})

        score = 0.0
        production_override_enabled = bool(cfg_get(section, "production_override_enabled", True))
        registry_backed = int(evidence.get("reg_serving_path_registry_backed", 0) or 0)
        prod_alias_used = int(evidence.get("reg_serving_prod_alias_used", 0) or 0)
        production_selected = int(evidence.get("reg_production_registry_selected", 0) or 0)
        local_fallback = int(evidence.get("reg_serving_local_fallback_present", 0) or 0)
        local_only = int(evidence.get("reg_serving_local_only", 0) or 0)

        if registry_backed:
            score += safe_float(cfg_get(section, "reg_serving_path_registry_backed", 0.0))
        if prod_alias_used:
            score += safe_float(cfg_get(section, "reg_serving_prod_alias_used", 0.0))

        caps: list[float] = []
        reasons: list[str] = []
        if local_only:
            caps.append(safe_float(cfg_get(section, "local_only_cap", 0.0)))
            reasons.append("local_only")
        if not registry_backed:
            caps.append(safe_float(cfg_get(section, "missing_registry_serving_cap", 2.0)))
            reasons.append("no_registry_serving_path")
        if not prod_alias_used:
            caps.append(safe_float(cfg_get(section, "missing_prod_alias_cap", 7.0)))
            reasons.append("missing_prod_alias")
        if local_fallback and not (production_override_enabled and production_selected):
            caps.append(safe_float(cfg_get(section, "local_fallback_cap", 7.0)))
            reasons.append("default_or_fallback_local")

        if caps:
            score = min(score, min(caps))

        evidence["reg_model_registry_cap_reason"] = "|".join(reasons)
        scores[DIMENSION_TO_SCORE_COLUMN["model_registry"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "api_serving"):
        section = cfg_get(config, "scoring.api_serving", {})

        score = 0.0
        signal_names = [
            "api_fastapi_app_present",
            "api_pydantic_contract_present",
            "api_health_endpoint_present",
            "api_predict_endpoint_present",
            "api_uvicorn_serving_present",
        ]
        for signal_name in signal_names:
            if int(evidence.get(signal_name, 0) or 0):
                score += safe_float(cfg_get(section, signal_name, 0.0))

        if not int(evidence.get("api_fastapi_app_present", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "no_fastapi_app_cap", 0.0)))
            evidence["api_serving_cap_reason"] = "no_fastapi_app"
        elif not int(evidence.get("api_predict_endpoint_present", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_predict_endpoint_cap", 5.0)))
            evidence["api_serving_cap_reason"] = "missing_predict_endpoint"
        elif not int(evidence.get("api_predict_calls_inference_logic", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "predict_without_inference_logic_cap", 7.0)))
            evidence["api_serving_cap_reason"] = "predict_without_inference_logic"
        else:
            evidence["api_serving_cap_reason"] = ""

        scores[DIMENSION_TO_SCORE_COLUMN["api_serving"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "error_handling"):
        section = cfg_get(config, "scoring.error_handling", {})

        signal_names = [
            "validation_function_present",
            "validation_raise_present",
            "validation_none_or_type_guard",
            "validation_empty_guard",
            "validation_required_columns_guard",
            "validation_missing_values_guard",
            "validation_target_guard",
            "validation_dtype_guard",
            "validation_range_or_domain_guard",
        ]

        score = 0.0
        for signal_name in signal_names:
            if int(evidence.get(signal_name, 0) or 0):
                score += safe_float(cfg_get(section, signal_name, 0.0))

        advanced_guard_count = sum(
            int(evidence.get(name, 0) or 0)
            for name in [
                "validation_target_guard",
                "validation_dtype_guard",
                "validation_range_or_domain_guard",
            ]
        )

        foundation_ok = all(
            int(evidence.get(name, 0) or 0)
            for name in [
                "validation_function_present",
                "validation_raise_present",
            ]
        )

        if advanced_guard_count == 0:
            score = min(score, safe_float(cfg_get(section, "no_advanced_validation_cap", 10.0)))

        if not foundation_ok:
            score = min(score, safe_float(cfg_get(section, "missing_foundation_cap", 10.0)))

        scores[DIMENSION_TO_SCORE_COLUMN["error_handling"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "artifacting"):
        section = cfg_get(config, "scoring.artifacting", {})

        signal_names = [
            "processed_data_artifact_present",
            "model_artifact_present",
            "prediction_artifact_present",
            "structured_asset_paths_present",
        ]

        score = 0.0
        for signal_name in signal_names:
            if int(evidence.get(signal_name, 0) or 0):
                score += safe_float(cfg_get(section, signal_name, 0.0))

        if not int(evidence.get("model_artifact_present", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_model_artifact_cap", 10.0)))

        if not int(evidence.get("prediction_artifact_present", 0) or 0):
            score = min(score, safe_float(cfg_get(section, "missing_prediction_artifact_cap", 10.0)))

        scores[DIMENSION_TO_SCORE_COLUMN["artifacting"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "pipeline"):
        section = cfg_get(config, "scoring.pipeline", {})
        valid_target_notebooks_with_src_import = int(
            evidence.get("valid_target_notebooks_with_src_import", 0) or 0
        )

        score = 0.0
        if valid_target_notebooks_with_src_import >= 1:
            score += safe_float(cfg_get(section, "valid_target_notebook_present", 0.0))

        scores[DIMENSION_TO_SCORE_COLUMN["pipeline"]] = round2(clamp(score))

    if _is_selected(selected_dimensions, "version_control"):
        section = cfg_get(config, "scoring.version_control", {})
        github_prs_found = int(evidence.get("github_prs_found", 0) or 0)
        github_unique_pr_authors = int(evidence.get("github_unique_pr_authors", 0) or 0)
        github_unique_reviewers = int(evidence.get("github_unique_reviewers", 0) or 0)

        score = 0.0

        if (
            github_unique_pr_authors >= int(cfg_get(section, "strong_min_pr_authors", 5))
            and github_prs_found >= int(cfg_get(section, "strong_min_prs", 5))
            and github_unique_reviewers >= int(cfg_get(section, "strong_min_reviewers", 3))
        ):
            score = 10.0
        elif (
            github_unique_pr_authors >= int(cfg_get(section, "good_min_pr_authors", 4))
            and github_prs_found >= int(cfg_get(section, "good_min_prs", 4))
        ):
            score = 8.0
        elif (
            github_unique_pr_authors >= int(cfg_get(section, "basic_min_pr_authors", 3))
            and github_prs_found >= int(cfg_get(section, "basic_min_prs", 3))
        ):
            score = 6.0
        elif (
            github_unique_pr_authors >= int(cfg_get(section, "limited_min_pr_authors", 2))
            and github_prs_found >= int(cfg_get(section, "limited_min_prs", 2))
        ):
            score = 4.0
        elif (
            github_unique_pr_authors >= int(cfg_get(section, "minimal_min_pr_authors", 1))
            and github_prs_found >= int(cfg_get(section, "minimal_min_prs", 1))
        ):
            score = 2.0
        else:
            score = 0.0

        scores[DIMENSION_TO_SCORE_COLUMN["version_control"]] = round2(score)

    return scores


def make_dimension_comments(
    evidence: dict[str, Any],
    scores: dict[str, float],
    selected_dimensions: set[str],
    repo_dir: Path | None = None,
) -> dict[str, str]:
    comments: dict[str, str] = {}

    def skipped() -> str:
        return "SKIPPED"

    if _is_selected(selected_dimensions, "modularization"):
        entry_signal = int(evidence.get("main_entry_signal", 0) or 0)
        core_count = int(evidence.get("core_module_count", 0) or 0)
        orchestrated_count = int(evidence.get("orchestrated_module_count", 0) or 0)
        stub_penalty_flag = int(evidence.get("stub_penalty_flag", 0) or 0)

        if entry_signal and orchestrated_count >= 7 and not stub_penalty_flag:
            comments["single_entry_modularization_comment"] = "Main entry point exists and the pipeline is modular and fully orchestrated end to end"
        elif entry_signal and orchestrated_count >= 6 and not stub_penalty_flag:
            comments["single_entry_modularization_comment"] = "Main entry point is clear and the pipeline is modular, but one core stage is not fully orchestrated from main.py"
        elif entry_signal and core_count >= 3 and orchestrated_count >= 2:
            comments["single_entry_modularization_comment"] = "The repo has a clear entry point and some modular orchestration, but integration is still partial"
        else:
            comments["single_entry_modularization_comment"] = "Single entry point or modular orchestration evidence is limited"
    else:
        comments["single_entry_modularization_comment"] = skipped()

    if _is_selected(selected_dimensions, "code_quality"):
        ruff_issues = int(evidence.get("ruff_issue_count", 999) or 999)
        pylint_score = safe_float(evidence.get("pylint_score"), 0.0)
        radon_avg = safe_float(evidence.get("radon_average_cc"), 20.0)

        if ruff_issues == 0 and pylint_score >= 9.0 and radon_avg <= 5.0:
            comments["code_quality_efficiency_comment"] = (
                "Linting is clean, maintainability is strong, and code complexity stays low"
            )
        elif ruff_issues <= 10 and pylint_score >= 8.0 and radon_avg <= 7.0:
            comments["code_quality_efficiency_comment"] = (
                "Code quality is generally strong, with only minor style or complexity issues"
            )
        else:
            comments["code_quality_efficiency_comment"] = (
                "Code quality evidence shows notable style, maintainability, or complexity issues"
            )
    else:
        comments["code_quality_efficiency_comment"] = skipped()

    if _is_selected(selected_dimensions, "documentation"):
        readme_present = int(evidence.get("readme_present", 0) or 0)

        if readme_present:
            comments["documentation_clarity_comment"] = "README.md exists and is ready for qualitative review"
        else:
            comments["documentation_clarity_comment"] = "README.md is missing"
    else:
        comments["documentation_clarity_comment"] = skipped()

    if _is_selected(selected_dimensions, "testing"):
        pass_count = int(evidence.get("pytest_pass_count", 0) or 0)
        total_count = int(evidence.get("pytest_total_count", 0) or 0)
        coverage_pct = evidence.get("coverage_pct")

        if evidence.get("pytest_import_error", 0):
            comments["testing_coverage_comment"] = "Pytest evidence is unreliable because dependencies are missing in the grading environment"
        elif evidence.get("pytest_timeout", 0):
            comments["testing_coverage_comment"] = "Pytest timed out, so testing evidence is incomplete"
        elif evidence.get("pytest_passed", 0):
            if isinstance(coverage_pct, (int, float)):
                comments["testing_coverage_comment"] = (
                    f"Pytest runs successfully with {pass_count}/{total_count} tests passing and {round2(coverage_pct)}% coverage"
                )
            else:
                comments["testing_coverage_comment"] = (
                    f"Pytest runs successfully with {pass_count}/{total_count} tests passing"
                )
        elif total_count > 0 and isinstance(coverage_pct, (int, float)):
            comments["testing_coverage_comment"] = (
                f"Pytest ran with {pass_count}/{total_count} tests passing and {round2(coverage_pct)}% coverage"
            )
        elif total_count > 0:
            comments["testing_coverage_comment"] = (
                f"Pytest ran with {pass_count}/{total_count} tests passing, but coverage could not be read cleanly"
            )
        else:
            comments["testing_coverage_comment"] = "Tests were found, but the suite did not run cleanly enough to produce usable scoring evidence"
    else:
        comments["testing_coverage_comment"] = skipped()

    if _is_selected(selected_dimensions, "dependencies"):
        if evidence.get("environment_yml_present", 0) or evidence.get("conda_yml_present", 0):
            comments["dependency_management_comment"] = "Environment definition file is present"
        else:
            comments["dependency_management_comment"] = "No environment.yml or conda.yml file was found"
    else:
        comments["dependency_management_comment"] = skipped()

    if _is_selected(selected_dimensions, "config_reproducibility"):
        config_yaml_present = int(evidence.get("config_yaml_present", 0) or 0)
        conda_lock_present = int(evidence.get("conda_lock_yml_present", 0) or 0)
        main_reads_config = int(evidence.get("main_reads_config_signal", 0) or 0)
        path_hits = int(evidence.get("code_hardcoded_path_hits", 0) or 0)
        hyperparam_hits = int(evidence.get("code_hardcoded_hyperparam_hits", 0) or 0)
        secret_hits = int(evidence.get("secret_like_literal_hits", 0) or 0)

        missing_items: list[str] = []
        if not config_yaml_present:
            missing_items.append("missing config.yaml")
        if not conda_lock_present:
            missing_items.append("missing conda-lock.yml")
        if not main_reads_config:
            missing_items.append("main.py does not clearly read config")
        if path_hits > 0:
            missing_items.append("hardcoded paths remain")
        if hyperparam_hits > 0:
            missing_items.append("hardcoded hyperparameters remain")
        if secret_hits > 0:
            missing_items.append("secret-like literals remain")

        if not missing_items:
            comments["config_reproducibility_comment"] = "Runtime settings are centralized and the environment setup is reproducibility-friendly"
        else:
            comments["config_reproducibility_comment"] = (
                "Config reproducibility gaps: " + "; ".join(missing_items)
            )
    else:
        comments["config_reproducibility_comment"] = skipped()

    if _is_selected(selected_dimensions, "security_secrets"):
        gitignore_ok = int(evidence.get("sec_gitignore_excludes_env", 0) or 0)
        dockerignore_ok = int(evidence.get("sec_dockerignore_excludes_env", 0) or 0)
        dockerignore_present = int(evidence.get("sec_dockerignore_present", 0) or 0)
        env_tracked = int(evidence.get("sec_env_file_tracked_by_git", 0) or 0)
        secret_hits = int(evidence.get("sec_secret_literal_hits", 0) or 0)
        tracked_env_like = evidence.get("sec_tracked_env_like_files", "")

        missing_items: list[str] = []
        if env_tracked:
            missing_items.append(".env tracked in git")
        if tracked_env_like:
            missing_items.append("env-like files tracked")
        if secret_hits > 0:
            missing_items.append("secret literals found")
        if not gitignore_ok:
            missing_items.append(".gitignore does not clearly exclude .env")
        if not dockerignore_present:
            missing_items.append(".dockerignore missing")
        elif not dockerignore_ok:
            missing_items.append(".dockerignore does not exclude .env")

        if gitignore_ok and dockerignore_ok and secret_hits == 0 and not env_tracked and not tracked_env_like:
            comments["security_secrets_comment"] = ".env is excluded from both git and Docker and no secret literals were found"
        else:
            comments["security_secrets_comment"] = "Security/secrets gaps: " + "; ".join(missing_items)
    else:
        comments["security_secrets_comment"] = skipped()

    if _is_selected(selected_dimensions, "logging_observability"):
        allowed_print_calls = 3
        print_hits = int(evidence.get("log_print_statement_hits", 0) or 0)
        print_files = evidence.get("log_print_hit_files", "")
        logger_present = int(evidence.get("log_logger_module_present", 0) or 0)
        dual_output = int(evidence.get("log_dual_output_signal", 0) or 0)
        file_handler = int(evidence.get("log_file_handler_present", 0) or 0)
        stream_handler = int(evidence.get("log_stream_handler_present", 0) or 0)
        usage_signal = int(evidence.get("log_logger_usage_signal", 0) or 0)
        suffix = f" in: {print_files}" if print_files else ""

        missing_items: list[str] = []
        if not logger_present:
            missing_items.append("src/logger.py missing")
        if not file_handler:
            missing_items.append("file handler missing")
        if not stream_handler:
            missing_items.append("stream/console handler missing")
        if not usage_signal:
            missing_items.append("logger not clearly used in production modules")
        if print_hits > allowed_print_calls:
            missing_items.append(f"print calls exceed allowed cleanup slack{suffix}")
        elif print_hits > 0:
            missing_items.append(f"print calls remain within allowed cleanup slack{suffix}")
        if int(evidence.get("log_logger_module_fallback_used", 0) or 0):
            missing_items.append("logging.py used instead of expected logger.py")

        if not missing_items:
            comments["logging_observability_comment"] = (
                "Logging is production-ready: dual-output, used across production modules, and within the allowed print cleanup slack"
            )
        else:
            comments["logging_observability_comment"] = "Logging gaps: " + "; ".join(missing_items)
    else:
        comments["logging_observability_comment"] = skipped()

    if _is_selected(selected_dimensions, "experiment_tracking"):
        init_present = int(evidence.get("wandb_init_in_main", 0) or 0)
        config_logged = int(evidence.get("wandb_config_logged", 0) or 0)
        metadata_present = int(evidence.get("wandb_run_metadata_logged", 0) or 0)
        metrics_present = int(evidence.get("wandb_eval_metrics_logged", 0) or 0)
        rich_tracking_present = int(evidence.get("wandb_rich_eval_tracking_logged", 0) or 0)
        model_artifact_present = int(evidence.get("wandb_model_artifact_logged", 0) or 0)
        wandb_log_in_helper = False

        if not metrics_present and repo_dir is not None:
            for path in production_python_files(repo_dir):
                if path == _find_main_path(repo_dir):
                    continue
                cleaned = _clean_python_for_detection(read_text(path))
                if "wandb.log(" in cleaned:
                    wandb_log_in_helper = True
                    break

        missing_items: list[str] = []
        if not metadata_present:
            missing_items.append(
                "run-level metadata is not clearly logged to W&B (e.g. dataset size, split info, selected model, entrypoint, or model artifact path)"
            )
        if not metrics_present:
            if wandb_log_in_helper:
                missing_items.append(
                    "Evaluation metrics are logged in a helper module rather than orchestrated from main.py. The rubric requires W&B tracking to be centrally owned by main.py."
                )
            else:
                missing_items.append(
                    "evaluation metrics are not clearly logged to W&B (e.g. metrics/, validation/test metrics, rmse, mae, accuracy, f1, precision, recall, or auc)"
                )
        if not rich_tracking_present:
            missing_items.append(
                "richer W&B evaluation tracking is missing (e.g. tables, plots, confusion matrix, ROC/PR curves, or comparison tables)"
            )
        if not model_artifact_present:
            missing_items.append("model artifact logging not clearly evidenced")

        if not init_present:
            missing_items.insert(0, "W&B not clearly initialized from main.py")
        if not config_logged:
            missing_items.append("full config/hyperparameters not passed to wandb.init (config= missing)")

        if (
            init_present
            and config_logged
            and metadata_present
            and metrics_present
            and rich_tracking_present
            and model_artifact_present
        ):
            comments["experiment_tracking_comment"] = (
                "W&B tracking is centrally initialized and captures run metadata, evaluation evidence, and model artifacts"
            )
        else:
            comments["experiment_tracking_comment"] = (
                "Experiment tracking gaps: " + "; ".join(missing_items)
            )
    else:
        comments["experiment_tracking_comment"] = skipped()

    if _is_selected(selected_dimensions, "model_registry"):
        registry_backed = int(evidence.get("reg_serving_path_registry_backed", 0) or 0)
        prod_alias_used = int(evidence.get("reg_serving_prod_alias_used", 0) or 0)
        production_selected = int(evidence.get("reg_production_registry_selected", 0) or 0)
        local_fallback = int(evidence.get("reg_serving_local_fallback_present", 0) or 0)
        local_only = int(evidence.get("reg_serving_local_only", 0) or 0)

        if local_only:
            comments["model_registry_comment"] = (
                "Serving loads a local unmanaged model file and no registry-backed serving path is clearly active"
            )
        elif registry_backed and not prod_alias_used and local_fallback:
            comments["model_registry_comment"] = (
                "Serving has a W&B registry branch, but defaults to a local model and does not clearly use the prod alias"
            )
        elif registry_backed and prod_alias_used and production_selected and local_fallback:
            comments["model_registry_comment"] = (
                "Production serving loads W&B prod first, but local fallback remains in the API"
            )
        elif registry_backed and prod_alias_used and not local_fallback:
            comments["model_registry_comment"] = (
                "API supports W&B prod registry serving, and deployment wiring is sufficient to treat it as the production path"
            )
        elif registry_backed and not prod_alias_used:
            comments["model_registry_comment"] = (
                "API has a W&B registry load path, but the serving alias is not clearly set to prod"
            )
        elif registry_backed and prod_alias_used and local_fallback:
            comments["model_registry_comment"] = (
                "API can load W&B prod, but serving still defaults to a local unmanaged model"
            )
        else:
            comments["model_registry_comment"] = (
                "Registry use is not clearly evidenced in the serving path"
            )
    else:
        comments["model_registry_comment"] = skipped()

    if _is_selected(selected_dimensions, "api_serving"):
        fastapi_app_present = int(evidence.get("api_fastapi_app_present", 0) or 0)
        pydantic_contract_present = int(evidence.get("api_pydantic_contract_present", 0) or 0)
        health_endpoint_present = int(evidence.get("api_health_endpoint_present", 0) or 0)
        predict_endpoint_present = int(evidence.get("api_predict_endpoint_present", 0) or 0)
        uvicorn_serving_present = int(evidence.get("api_uvicorn_serving_present", 0) or 0)
        predict_calls_inference_logic = int(evidence.get("api_predict_calls_inference_logic", 0) or 0)

        if not fastapi_app_present:
            comments["api_serving_comment"] = "No real FastAPI serving layer is clearly evidenced"
        elif not predict_endpoint_present:
            comments["api_serving_comment"] = "FastAPI app exists, but /predict is missing"
        elif not predict_calls_inference_logic:
            comments["api_serving_comment"] = "Predict endpoint exists, but inference logic is not clearly wired"
        elif not pydantic_contract_present:
            comments["api_serving_comment"] = "API endpoints exist, but the Pydantic contract is not clearly enforced"
        elif not uvicorn_serving_present:
            comments["api_serving_comment"] = "Uvicorn serving is not clearly evidenced"
        elif health_endpoint_present and predict_endpoint_present:
            comments["api_serving_comment"] = (
                "FastAPI serves /health and /predict with a clear Pydantic request contract"
            )
        else:
            comments["api_serving_comment"] = "FastAPI app exists, but /health is missing"
    else:
        comments["api_serving_comment"] = skipped()

    if _is_selected(selected_dimensions, "error_handling"):
        function_present = int(evidence.get("validation_function_present", 0) or 0)
        raise_present = int(evidence.get("validation_raise_present", 0) or 0)

        core_guard_count = sum(
            int(evidence.get(name, 0) or 0)
            for name in [
                "validation_none_or_type_guard",
                "validation_empty_guard",
                "validation_required_columns_guard",
                "validation_missing_values_guard",
            ]
        )

        advanced_guard_count = sum(
            int(evidence.get(name, 0) or 0)
            for name in [
                "validation_target_guard",
                "validation_dtype_guard",
                "validation_range_or_domain_guard",
            ]
        )

        if function_present and raise_present and core_guard_count >= 3 and advanced_guard_count >= 2:
            comments["error_handling_validation_comment"] = (
                "Validation breadth is strong and covers core schema checks plus deeper dataset-specific rules"
            )
        elif function_present and raise_present and core_guard_count >= 2:
            comments["error_handling_validation_comment"] = (
                "Validation covers basic fail-fast and core schema checks, but deeper target, dtype, or domain rules are limited"
            )
        elif function_present or raise_present:
            comments["error_handling_validation_comment"] = (
                "Validation evidence is present, but it is too narrow to support comprehensive error handling"
            )
        else:
            comments["error_handling_validation_comment"] = (
                "Validation and robust error handling signals are limited"
            )
    else:
        comments["error_handling_validation_comment"] = skipped()

    if _is_selected(selected_dimensions, "artifacting"):
        processed_ok = int(evidence.get("processed_data_artifact_present", 0) or 0)
        model_ok = int(evidence.get("model_artifact_present", 0) or 0)
        prediction_ok = int(evidence.get("prediction_artifact_present", 0) or 0)

        if processed_ok and model_ok and prediction_ok:
            comments["artifacting_reproducibility_comment"] = (
                "main.py clearly produces processed data, a serialized model, and prediction artifacts"
            )
        elif model_ok and (processed_ok or prediction_ok):
            comments["artifacting_reproducibility_comment"] = (
                "Artifact storage is partly structured, but one core pipeline output is still missing"
            )
        else:
            comments["artifacting_reproducibility_comment"] = (
                "Artifact generation from main.py is limited or not clearly evidenced"
            )
    else:
        comments["artifacting_reproducibility_comment"] = skipped()

    if _is_selected(selected_dimensions, "pipeline"):
        target_notebooks_found = int(evidence.get("target_notebooks_found", 0) or 0)
        valid_target_notebooks = int(evidence.get("valid_target_notebooks", 0) or 0)
        valid_target_notebooks_with_src_import = int(
            evidence.get("valid_target_notebooks_with_src_import", 0) or 0
        )

        if valid_target_notebooks_with_src_import >= 1:
            comments["pipeline_completeness_comment"] = "At least one valid experimentation or modular notebook was found and it imports from src"
        elif valid_target_notebooks >= 1:
            comments["pipeline_completeness_comment"] = "A valid target notebook was found, but it does not import from src"
        elif target_notebooks_found >= 1:
            comments["pipeline_completeness_comment"] = "A notebook matched the naming rule, but it could not be parsed as a valid notebook file"
        else:
            comments["pipeline_completeness_comment"] = "No valid experimentation or modular notebook matched the naming rule"
    else:
        comments["pipeline_completeness_comment"] = skipped()

    if _is_selected(selected_dimensions, "version_control"):
        api_authenticated = int(evidence.get("github_api_authenticated", 0) or 0)
        pr_authors = int(evidence.get("github_unique_pr_authors", 0) or 0)
        reviewers = int(evidence.get("github_unique_reviewers", 0) or 0)
        prs = int(evidence.get("github_prs_found", 0) or 0)

        if not api_authenticated:
            comments["version_control_workflow_comment"] = "GitHub token was not detected, so Pull Request evidence may be incomplete"
        elif pr_authors >= 5 and reviewers >= 3 and prs >= 5:
            comments["version_control_workflow_comment"] = "Strong GitHub workflow evidence shows broad team participation through Pull Requests and reviews"
        elif pr_authors >= 4 and prs >= 4:
            comments["version_control_workflow_comment"] = "Good GitHub workflow evidence shows several contributors using Pull Requests, but participation is not yet near full-team"
        elif pr_authors >= 3 and prs >= 3:
            comments["version_control_workflow_comment"] = "Basic GitHub workflow evidence is present, but contribution coverage across the team is still partial"
        elif pr_authors >= 1 and prs >= 1:
            comments["version_control_workflow_comment"] = "Only limited GitHub workflow participation is evidenced"
        else:
            comments["version_control_workflow_comment"] = "Little or no usable Pull Request workflow evidence was found"
    else:
        comments["version_control_workflow_comment"] = skipped()

    ran_comments = [value for key, value in comments.items() if key.endswith("_comment") and value != "SKIPPED"]
    if ran_comments:
        comments["overall_comment"] = "First-pass qualitative feedback generated from the selected rubric dimensions"
    else:
        comments["overall_comment"] = "No rubric dimensions were selected for this run"

    return comments


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
