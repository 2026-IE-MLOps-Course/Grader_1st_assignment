"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single command entry point
  (python -m src.main) that runs the pipeline end-to-end reliably.
- Responsibility (separation of concerns): Orchestrates steps; does not contain
  the detailed logic of cleaning/features/training/evaluation.
- Pipeline contract (inputs and outputs): Produces three artifacts:
  - data/processed/clean.csv
  - models/model.joblib
  - reports/predictions.csv

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe
from sklearn.metrics import r2_score


def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """
    Load YAML configuration from disk.

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    ValueError
        If config.yaml is empty or invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path.resolve()}")

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("config.yaml is empty or invalid (expected a YAML mapping).")

    return cfg


def main() -> None:
    """
    Inputs:
    - None (reads config.yaml)
    Outputs:
    - None (creates artifacts on disk)
    Why this contract matters for reliable ML delivery:
    - A single script entry point makes runs reproducible and CI-friendly.
    """
    print("[main.main] Starting end-to-end pipeline...")  # TODO: replace with logging later

    cfg = load_config()

    # Ensure required directories exist (idempotent)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Config sections
    data_cfg = cfg.get("data", {})
    art_cfg = cfg.get("artifacts", {})
    task_cfg = cfg.get("task", {})
    feat_cfg = cfg.get("features", {})

    # Required config keys (fail fast)
    raw_path = Path(data_cfg.get("raw_path", ""))
    processed_path = Path(data_cfg.get("processed_path", ""))
    model_path = Path(art_cfg.get("model_path", ""))
    predictions_path = Path(art_cfg.get("predictions_path", ""))

    problem_type = task_cfg.get("problem_type")
    target_column = task_cfg.get("target_column")
    year_column = task_cfg.get("year_column")
    test_year = task_cfg.get("test_year")

    if not raw_path:
        raise ValueError("config.yaml missing: data.raw_path")
    if not processed_path:
        raise ValueError("config.yaml missing: data.processed_path")
    if not model_path:
        raise ValueError("config.yaml missing: artifacts.model_path")
    if not predictions_path:
        raise ValueError("config.yaml missing: artifacts.predictions_path")

    if problem_type not in {"regression", "classification"}:
        raise ValueError("config.yaml task.problem_type must be 'regression' or 'classification'")
    if not target_column:
        raise ValueError("config.yaml missing: task.target_column")
    if not year_column:
        raise ValueError("config.yaml missing: task.year_column (year split is required)")
    if test_year is None:
        raise ValueError("config.yaml missing: task.test_year (year split is required)")

    numeric_cols = feat_cfg.get("numeric_passthrough", [])
    categorical_cols = feat_cfg.get("categorical_onehot", [])
    quantile_bin_cols = feat_cfg.get("quantile_bin", [])
    n_bins = feat_cfg.get("n_bins", 3)

    # 1) Load raw data
    df_raw = load_raw_data(raw_path)

    # 2) Clean data
    df_clean = clean_dataframe(df_raw, target_column=target_column)

    # 3) Save processed CSV artifact
    save_csv(df_clean, processed_path)

    # 4) Validate schema
    required_columns = [target_column] + list(numeric_cols) + list(categorical_cols)
    validate_dataframe(df_clean, required_columns=required_columns)

    # 5) Year-based split (REQUIRED)
    if year_column not in df_clean.columns:
        raise ValueError(
            f"Year split required, but year column '{year_column}' not found in cleaned dataframe."
        )

    print(f"[main.main] Using year split: train < {test_year}, test == {test_year}")  # TODO: replace with logging later
    df_train = df_clean[df_clean[year_column] < test_year].copy()
    df_test = df_clean[df_clean[year_column] == test_year].copy()

    if df_train.empty:
        raise ValueError(
            f"Year split produced empty TRAIN set. No rows where {year_column} < {test_year}."
        )
    if df_test.empty:
        raise ValueError(
            f"Year split produced empty TEST set. No rows where {year_column} == {test_year}."
        )

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    # 6) Fail-fast feature presence checks
    configured_cols = list(numeric_cols) + list(categorical_cols)
    missing_feats = [c for c in configured_cols if c not in X_train.columns]
    if missing_feats:
        raise ValueError(f"Configured feature columns missing from training data: {missing_feats}")

    # 7) Build preprocessing recipe (unfitted)
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=quantile_bin_cols,
        categorical_onehot_cols=categorical_cols,
        numeric_passthrough_cols=numeric_cols,
        n_bins=n_bins,
    )

    # 8) Train model (Pipeline fits preprocess ONLY on train -> leakage-safe)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=problem_type,
    )

    # 9) Save model artifact
    save_model(model, model_path)

    # 10) Evaluate
    score = evaluate_model(model, X_test=X_test, y_test=y_test, problem_type=problem_type)
    print(f"[main.main] Final score returned (single float): {score:.4f}")  # TODO: replace with logging later

    # 11) Inference (example: run on X_test)
    df_pred = run_inference(model, X_infer=X_test)

    # 12) Save predictions artifact
    save_csv(df_pred, predictions_path)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add richer reporting (feature importances, experiment tracking)
    # Why: Production systems need traceability and monitoring signals.
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print("[main.main] Pipeline completed successfully.")  # TODO: replace with logging later

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("Train R2:", r2_score(y_train, train_preds))
    print("Test  R2:", r2_score(y_test, test_preds))

if __name__ == "__main__":
    main()
