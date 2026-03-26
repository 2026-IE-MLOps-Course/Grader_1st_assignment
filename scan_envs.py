from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLONES_DIR = "grading_workspace/clones"
DEFAULT_OUTPUT_DIR = "outputs/dependency_scan"
DEFAULT_THRESHOLD = 2


ALWAYS_INCLUDE_CONDA = [
    "python>=3.9",
    "pandas",
    "requests",
    "pytest",
    "pytest-cov",
    "pip",
]

ALWAYS_INCLUDE_PIP = []

SUSPICIOUS_PATTERNS = [
    "cuda",
    "cudnn",
    "pytorch-cuda",
    "tensorflow-gpu",
    "gpu",
    "win",
    "windows",
    "mkl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan cloned student repos for environment.yml files and build a consolidated grading environment"
    )
    parser.add_argument(
        "--clones-dir",
        default=DEFAULT_CLONES_DIR,
        help="Directory containing already-cloned repos",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV and environment outputs will be written",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Include dependency in consolidated environment if it appears in at least this many repos",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def find_environment_file(repo_dir: Path) -> Path | None:
    candidates = [
        repo_dir / "environment.yml",
        repo_dir / "environment.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def strip_inline_comment(line: str) -> str:
    if "#" not in line:
        return line.rstrip()
    out = []
    in_single = False
    in_double = False
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse_conda_environment_yaml(text: str) -> dict[str, list[str]]:
    """
    Lightweight parser for the subset of conda environment.yml used in these repos

    Supports:
    - channels
    - dependencies
    - pip subsection under dependencies
    """
    channels: list[str] = []
    conda_deps: list[str] = []
    pip_deps: list[str] = []

    lines = text.splitlines()

    in_channels = False
    in_dependencies = False
    in_pip_block = False
    pip_block_indent = None

    for raw_line in lines:
        line_no_comment = strip_inline_comment(raw_line)
        if not line_no_comment.strip():
            continue

        indent = len(line_no_comment) - len(line_no_comment.lstrip(" "))
        stripped = line_no_comment.strip()

        if indent == 0 and stripped == "channels:":
            in_channels = True
            in_dependencies = False
            in_pip_block = False
            pip_block_indent = None
            continue

        if indent == 0 and stripped == "dependencies:":
            in_channels = False
            in_dependencies = True
            in_pip_block = False
            pip_block_indent = None
            continue

        if indent == 0 and stripped.endswith(":") and stripped not in {"channels:", "dependencies:"}:
            in_channels = False
            in_dependencies = False
            in_pip_block = False
            pip_block_indent = None
            continue

        if in_channels:
            if stripped.startswith("- "):
                value = stripped[2:].strip()
                if value:
                    channels.append(value)
            continue

        if in_dependencies:
            if stripped.startswith("- pip:"):
                in_pip_block = True
                pip_block_indent = indent
                continue

            if in_pip_block:
                if indent <= (pip_block_indent or 0):
                    in_pip_block = False
                elif stripped.startswith("- "):
                    value = stripped[2:].strip()
                    if value:
                        pip_deps.append(value)
                    continue

            if not in_pip_block and stripped.startswith("- "):
                value = stripped[2:].strip()
                if value and value != "pip":
                    conda_deps.append(value)
                elif value == "pip":
                    conda_deps.append(value)

    return {
        "channels": channels,
        "conda": conda_deps,
        "pip": pip_deps,
    }


def normalize_package_name(dep: str) -> str:
    value = dep.strip()

    if not value:
        return ""

    if value.startswith("- "):
        value = value[2:].strip()

    if ";" in value:
        value = value.split(";", 1)[0].strip()

    if "[" in value:
        value = value.split("[", 1)[0].strip()

    value = re.split(r"\s+", value)[0]
    value = re.split(r"(==|>=|<=|~=|!=|=|>|<)", value, maxsplit=1)[0].strip()
    value = value.lower().replace("_", "-")

    return value


def detect_version_style(dep: str) -> str:
    dep = dep.strip()
    if "==" in dep:
        return "exact"
    if any(op in dep for op in [">=", "<=", "~=", "!=", ">", "<"]):
        return "range"
    if re.search(r"[A-Za-z0-9_-]+=+[A-Za-z0-9]", dep):
        return "conda-pin"
    return "unpinned"


def repo_id_from_folder(repo_dir: Path) -> str:
    return repo_dir.name


def build_repo_dependency_records(clones_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for repo_dir in sorted(clones_dir.iterdir()):
        if not repo_dir.is_dir():
            continue

        env_file = find_environment_file(repo_dir)
        if env_file is None:
            skipped.append(repo_id_from_folder(repo_dir))
            continue

        parsed = parse_conda_environment_yaml(read_text(env_file))

        for raw_dep in parsed["conda"]:
            norm = normalize_package_name(raw_dep)
            if not norm:
                continue
            records.append(
                {
                    "repo_id": repo_id_from_folder(repo_dir),
                    "source_file": str(env_file),
                    "manager": "conda",
                    "raw_dependency": raw_dep,
                    "normalized_name": norm,
                    "version_style": detect_version_style(raw_dep),
                }
            )

        for raw_dep in parsed["pip"]:
            norm = normalize_package_name(raw_dep)
            if not norm:
                continue
            records.append(
                {
                    "repo_id": repo_id_from_folder(repo_dir),
                    "source_file": str(env_file),
                    "manager": "pip",
                    "raw_dependency": raw_dep,
                    "normalized_name": norm,
                    "version_style": detect_version_style(raw_dep),
                }
            )

    return records, skipped


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_frequency_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repo_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_examples: dict[tuple[str, str], Counter] = defaultdict(Counter)
    version_styles: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for row in records:
        key = (row["manager"], row["normalized_name"])
        repo_sets[key].add(row["repo_id"])
        raw_examples[key][row["raw_dependency"]] += 1
        version_styles[key][row["version_style"]] += 1

    output: list[dict[str, Any]] = []
    for (manager, name), repos in sorted(repo_sets.items()):
        output.append(
            {
                "manager": manager,
                "normalized_name": name,
                "repo_count": len(repos),
                "repos": " | ".join(sorted(repos)),
                "most_common_raw_form": raw_examples[(manager, name)].most_common(1)[0][0],
                "version_styles_seen": " | ".join(
                    f"{style}:{count}" for style, count in version_styles[(manager, name)].most_common()
                ),
            }
        )
    return output


def build_repo_matrix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repo_ids = sorted({row["repo_id"] for row in records})
    dep_keys = sorted({(row["manager"], row["normalized_name"]) for row in records})

    dep_to_repos: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_examples: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for row in records:
        key = (row["manager"], row["normalized_name"])
        dep_to_repos[key].add(row["repo_id"])
        raw_examples[key][row["raw_dependency"]] += 1

    output: list[dict[str, Any]] = []
    for manager, dep_name in dep_keys:
        row: dict[str, Any] = {
            "manager": manager,
            "normalized_name": dep_name,
            "most_common_raw_form": raw_examples[(manager, dep_name)].most_common(1)[0][0],
            "repo_count": len(dep_to_repos[(manager, dep_name)]),
        }
        for repo_id in repo_ids:
            row[repo_id] = 1 if repo_id in dep_to_repos[(manager, dep_name)] else 0
        output.append(row)

    return output


def choose_consolidated_dependencies(
    records: list[dict[str, Any]],
    threshold: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    repo_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_examples: dict[tuple[str, str], Counter] = defaultdict(Counter)
    managers_for_name: dict[str, Counter] = defaultdict(Counter)

    for row in records:
        key = (row["manager"], row["normalized_name"])
        repo_sets[key].add(row["repo_id"])
        raw_examples[key][row["raw_dependency"]] += 1
        managers_for_name[row["normalized_name"]][row["manager"]] += 1

    selected_conda: set[str] = set()
    selected_pip: set[str] = set()
    notes: list[dict[str, Any]] = []

    always_conda_normalized = {normalize_package_name(dep) for dep in ALWAYS_INCLUDE_CONDA}
    always_pip_normalized = {normalize_package_name(dep) for dep in ALWAYS_INCLUDE_PIP}

    normalized_to_best_raw: dict[tuple[str, str], str] = {}
    for key, counter in raw_examples.items():
        normalized_to_best_raw[key] = counter.most_common(1)[0][0]

    all_names = sorted({row["normalized_name"] for row in records})
    for name in all_names:
        conda_count = len(repo_sets.get(("conda", name), set()))
        pip_count = len(repo_sets.get(("pip", name), set()))
        total_count = len(set().union(repo_sets.get(("conda", name), set()), repo_sets.get(("pip", name), set())))
        preferred_manager = "conda" if conda_count >= pip_count else "pip"

        rare = total_count < threshold
        suspicious = any(token in name for token in SUSPICIOUS_PATTERNS)
        dual_manager = conda_count > 0 and pip_count > 0

        action = "skip"
        reason_parts = []

        if name in always_conda_normalized:
            selected_conda.add(name)
            action = "include_conda_always"
            reason_parts.append("core grader stack")
        elif name in always_pip_normalized:
            selected_pip.add(name)
            action = "include_pip_always"
            reason_parts.append("core grader stack")
        elif total_count >= threshold:
            if preferred_manager == "conda":
                selected_conda.add(name)
                action = "include_conda_threshold"
            else:
                selected_pip.add(name)
                action = "include_pip_threshold"
            reason_parts.append(f"appears_in_{total_count}_repos")
        else:
            reason_parts.append("rare_package")

        if suspicious:
            reason_parts.append("suspicious_or_platform_specific")

        if dual_manager:
            reason_parts.append("appears_in_both_conda_and_pip")

        notes.append(
            {
                "normalized_name": name,
                "conda_repo_count": conda_count,
                "pip_repo_count": pip_count,
                "total_repo_count": total_count,
                "preferred_manager": preferred_manager,
                "rare": int(rare),
                "suspicious": int(suspicious),
                "dual_manager": int(dual_manager),
                "action": action,
                "reason": " | ".join(reason_parts),
                "example_conda_form": normalized_to_best_raw.get(("conda", name), ""),
                "example_pip_form": normalized_to_best_raw.get(("pip", name), ""),
            }
        )

    selected_conda_raw = []
    for dep in ALWAYS_INCLUDE_CONDA:
        if normalize_package_name(dep) in selected_conda and dep not in selected_conda_raw:
            selected_conda_raw.append(dep)

    selected_pip_raw = []
    for dep in ALWAYS_INCLUDE_PIP:
        if normalize_package_name(dep) in selected_pip and dep not in selected_pip_raw:
            selected_pip_raw.append(dep)

    for name in sorted(selected_conda):
        if name in always_conda_normalized:
            continue
        best_raw = normalized_to_best_raw.get(("conda", name), name)
        selected_conda_raw.append(simplify_dependency_spec(best_raw, manager="conda"))

    for name in sorted(selected_pip):
        if name in always_pip_normalized:
            continue
        best_raw = normalized_to_best_raw.get(("pip", name), name)
        selected_pip_raw.append(simplify_dependency_spec(best_raw, manager="pip"))

    selected_conda_raw = dedupe_preserve_order(selected_conda_raw)
    selected_pip_raw = dedupe_preserve_order(selected_pip_raw)

    if "pip" not in [normalize_package_name(dep) for dep in selected_conda_raw]:
        selected_conda_raw.append("pip")

    return selected_conda_raw, selected_pip_raw, notes


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        key = item.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def simplify_dependency_spec(dep: str, manager: str) -> str:
    dep = dep.strip()
    name = normalize_package_name(dep)

    if not name:
        return dep

    if name == "python":
        return "python>=3.9"

    if manager == "conda":
        return name
    return name


def write_consolidated_environment(
    output_path: Path,
    conda_deps: list[str],
    pip_deps: list[str],
) -> None:
    lines = [
        "name: mlops-grader-light",
        "channels:",
        "  - conda-forge",
        "  - defaults",
        "dependencies:",
    ]

    for dep in conda_deps:
        lines.append(f"  - {dep}")

    if pip_deps:
        lines.append("  - pip:")
        for dep in pip_deps:
            lines.append(f"      - {dep}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    clones_dir = Path(args.clones_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not clones_dir.exists():
        raise SystemExit(f"Clones directory not found: {clones_dir}")

    records, skipped_repos = build_repo_dependency_records(clones_dir)

    if not records:
        raise SystemExit(
            "No environment.yml files were found or no dependencies could be parsed from them"
        )

    frequency_rows = build_frequency_table(records)
    matrix_rows = build_repo_matrix(records)
    conda_deps, pip_deps, notes_rows = choose_consolidated_dependencies(
        records=records,
        threshold=args.threshold,
    )

    dependency_frequency_path = output_dir / "dependency_frequency.csv"
    dependency_matrix_path = output_dir / "dependency_repo_matrix.csv"
    dependency_notes_path = output_dir / "dependency_notes.csv"
    consolidated_env_path = output_dir / "consolidated_environment.yml"
    skipped_repos_path = output_dir / "repos_without_environment.csv"

    write_csv(frequency_rows, dependency_frequency_path)
    write_csv(matrix_rows, dependency_matrix_path)
    write_csv(notes_rows, dependency_notes_path)
    write_csv([{"repo_id": repo_id} for repo_id in skipped_repos], skipped_repos_path)
    write_consolidated_environment(consolidated_env_path, conda_deps, pip_deps)

    scanned_repo_count = len(sorted({row["repo_id"] for row in records}))
    total_repo_dirs = len([p for p in clones_dir.iterdir() if p.is_dir()])

    print("Dependency scan complete")
    print(f"Clones dir: {clones_dir}")
    print(f"Total repo folders found: {total_repo_dirs}")
    print(f"Repos with environment.yml parsed: {scanned_repo_count}")
    print(f"Repos skipped because no environment.yml was found: {len(skipped_repos)}")
    print("")
    print("Outputs")
    print(f"- {dependency_frequency_path}")
    print(f"- {dependency_matrix_path}")
    print(f"- {dependency_notes_path}")
    print(f"- {skipped_repos_path}")
    print(f"- {consolidated_env_path}")
    print("")
    print("Consolidated conda dependencies")
    for dep in conda_deps:
        print(f"  - {dep}")
    if pip_deps:
        print("")
        print("Consolidated pip dependencies")
        for dep in pip_deps:
            print(f"  - {dep}")


if __name__ == "__main__":
    main()