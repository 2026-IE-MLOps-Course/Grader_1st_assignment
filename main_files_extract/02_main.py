"""
Educational Goal:
- Why this module exists in an MLOps system: The orchestrator is the single
  entry point that drives the entire pipeline from configuration. It delegates
  execution to specialised, single-purpose modules, creating a highly readable
  flow for auditing and traceability.
- Responsibility (separation of concerns): Coordinate Load → Clean → Validate
  → Split → Build Recipe → Train → Evaluate → Infer. Never implement business
  logic directly — always delegate to the appropriate module.
- Pipeline contract (inputs and outputs):
  Input  — SETTINGS dictionary (bridge to future config.yaml).
  Output — Artifacts: data/processed/clean.csv, models/model.joblib,
           reports/predictions.csv.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_csv, save_csv, save_model
from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference


# ================================================================== #
#  CONFIGURATION — SETTINGS DICTIONARY (bridge to future config.yaml) #
# ================================================================== #
# !!! STUDENTS: You MUST update this block to match YOUR dataset !!!  #
# This SETTINGS dictionary is pre-configured for the VoyageIQ travel  #
# dataset. If you use the dummy scaffolding CSV, set                  #
# is_example_config = True and update the feature lists.              #
# ================================================================== #
SETTINGS = {
    "is_example_config": False,  # Set True if using dummy scaffolding data
    "target_column": "total_cost",
    "problem_type": "regression",  # "regression" or "classification"
    "random_seed": 42,
    "test_size": 0.15,
    "val_size": 0.15,
    "paths": {
        "raw_data": "data/raw/travel_raw.csv",
        "processed_data": "data/processed/clean.csv",
        "model_artifact": "models/model.joblib",
        "predictions_output": "reports/predictions.csv",
    },
    "features": {
        "quantile_bin": [],  # No binning for travel cost regression
        "categorical_onehot": [
            "destination_country",
            "traveler_gender",
            "traveler_nationality",
            "accommodation_type",
            "transportation_type",
        ],
        "numeric_passthrough": [
            "duration_days",
            "traveler_age",
            "travel_month",
            "day_of_week",
        ],
        "n_bins": 3,
    },
}


def main():
    """Execute the end-to-end ML pipeline.

    Steps:
    1. Create directories for artifacts.
    2. Load raw data.
    3. Clean data (deterministic, idempotent).
    4. Save processed CSV.
    5. Validate data against schema.
    6. Train/val/test split (BEFORE any feature fitting).
    7. Fail-fast feature checks.
    8. Build feature recipe (unfitted ColumnTransformer).
    9. Train model (fit preprocessor + model on train only).
    10. Save model artifact.
    11. Evaluate on held-out validation and test splits.
    12. Run inference and save predictions.
    """
    print("=" * 60)  # TODO: replace with logging later
    print("Pipeline started")  # TODO: replace with logging later
    print("=" * 60)  # TODO: replace with logging later

    # ── 1. Create output directories ───────────────────────
    print("[main] Creating output directories …")  # TODO: replace with logging later
    for key in ["processed_data", "model_artifact", "predictions_output"]:
        Path(SETTINGS["paths"][key]).parent.mkdir(parents=True, exist_ok=True)

    # ── Check if running example config ────────────────────
    if SETTINGS["is_example_config"]:
        print(
            "\n⚠️  WARNING: Running with EXAMPLE configuration. "
            "Update SETTINGS to match your real dataset!\n"
        )  # TODO: replace with logging later

    # ── 2. Load raw data ───────────────────────────────────
    print("[main] STEP 1 — Loading raw data")  # TODO: replace with logging later
    raw_path = Path(SETTINGS["paths"]["raw_data"])
    df_raw = load_raw_data(raw_path)

    # ── 3. Clean data ──────────────────────────────────────
    print("[main] STEP 2 — Cleaning data")  # TODO: replace with logging later
    target_col = SETTINGS["target_column"]
    df_clean = clean_dataframe(df_raw, target_col)

    # ── 4. Save processed CSV ──────────────────────────────
    print("[main] Saving processed data …")  # TODO: replace with logging later
    save_csv(df_clean, Path(SETTINGS["paths"]["processed_data"]))

    # ── 5. Validate data ───────────────────────────────────
    print("[main] STEP 3 — Validating data")  # TODO: replace with logging later
    feat_cfg = SETTINGS["features"]
    required_columns = (
        feat_cfg["numeric_passthrough"]
        + feat_cfg["categorical_onehot"]
        + feat_cfg["quantile_bin"]
        + [target_col]
    )
    validate_dataframe(df_clean, required_columns)

    # ── 6. Train / Val / Test split ────────────────────────
    print("[main] STEP 4 — Splitting data (train/val/test)")  # TODO: replace with logging later
    seed = SETTINGS["random_seed"]
    test_size = SETTINGS["test_size"]
    val_size = SETTINGS["val_size"]
    problem_type = SETTINGS["problem_type"]

    # Determine feature columns
    keep_cols = (
        feat_cfg["numeric_passthrough"]
        + feat_cfg["categorical_onehot"]
        + feat_cfg["quantile_bin"]
    )
    keep_cols = [c for c in keep_cols if c in df_clean.columns]

    X = df_clean[keep_cols]
    y = df_clean[target_col]

    # Stratify only for classification
    try:
        stratify_arg = y if problem_type == "classification" else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_arg,
        )
    except ValueError:
        # Fallback: stratification may fail with very small classes
        print(  # TODO: replace with logging later
            "[main] WARNING: Stratification failed, falling back to "
            "unstratified split."
        )
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )

    # Second split: train vs val
    relative_val = val_size / (1 - test_size)
    try:
        stratify_arg = y_temp if problem_type == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val,
            random_state=seed,
            stratify=stratify_arg,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val,
            random_state=seed,
        )

    print(  # TODO: replace with logging later
        f"[main] Split sizes — train: {len(X_train)}, "
        f"val: {len(X_val)}, test: {len(X_test)}"
    )

    # ── 7. Fail-fast feature checks ───────────────────────
    print("[main] STEP 5 — Checking feature columns …")  # TODO: replace with logging later
    all_configured_cols = (
        feat_cfg["quantile_bin"]
        + feat_cfg["categorical_onehot"]
        + feat_cfg["numeric_passthrough"]
    )
    missing_cols = [c for c in all_configured_cols if c not in X_train.columns]
    if missing_cols:
        raise ValueError(
            f"Configured feature columns missing from training data: {missing_cols}"
        )

    # Verify numeric dtypes for quantile_bin and numeric_passthrough columns
    for col in feat_cfg["quantile_bin"] + feat_cfg["numeric_passthrough"]:
        if col in X_train.columns and not pd.api.types.is_numeric_dtype(X_train[col]):
            raise TypeError(
                f"Column '{col}' is configured as numeric but has dtype "
                f"'{X_train[col].dtype}'. Fix in clean_data or SETTINGS."
            )

    # ── 8. Build feature recipe ────────────────────────────
    print("[main] STEP 6 — Building feature preprocessor recipe")  # TODO: replace with logging later
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=feat_cfg["quantile_bin"],
        categorical_onehot_cols=feat_cfg["categorical_onehot"],
        numeric_passthrough_cols=feat_cfg["numeric_passthrough"],
        n_bins=feat_cfg["n_bins"],
    )

    # ── 9. Train model ─────────────────────────────────────
    print("[main] STEP 7 — Training model")  # TODO: replace with logging later
    pipeline = train_model(X_train, y_train, preprocessor, problem_type)

    # ── 10. Save model artifact ────────────────────────────
    print("[main] Saving model artifact …")  # TODO: replace with logging later
    save_model(pipeline, Path(SETTINGS["paths"]["model_artifact"]))

    # ── 11. Evaluate on validation split ───────────────────
    print("[main] STEP 8 — Evaluating on validation split")  # TODO: replace with logging later
    val_metric = evaluate_model(pipeline, X_val, y_val, problem_type)
    print(f"[main] Validation metric: {val_metric:.4f}")  # TODO: replace with logging later

    # ── 12. Evaluate on test split ─────────────────────────
    print("[main] STEP 9 — Evaluating on test split")  # TODO: replace with logging later
    test_metric = evaluate_model(pipeline, X_test, y_test, problem_type)
    print(f"[main] Test metric: {test_metric:.4f}")  # TODO: replace with logging later

    # ── 13. Inference on test split (demonstration) ────────
    print("[main] Running inference on test split …")  # TODO: replace with logging later
    preds = run_inference(pipeline, X_test)
    save_csv(preds, Path(SETTINGS["paths"]["predictions_output"]))

    print("=" * 60)  # TODO: replace with logging later
    print("Pipeline completed successfully.")  # TODO: replace with logging later
    print("=" * 60)  # TODO: replace with logging later


if __name__ == "__main__":
    main()

