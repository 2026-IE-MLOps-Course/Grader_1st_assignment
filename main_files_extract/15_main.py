"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single, repeatable
  entrypoint that stitches together data -> cleaning -> validation ->
  features -> training -> evaluation -> inference -> artifact saving.
- Responsibility (separation of concerns): Orchestrates the workflow only;
  all real logic lives in the other src/ modules.
- Pipeline contract: Reads raw CSV(s), writes clean CSV + model artifact
  + predictions CSV, prints metrics to console.
TODO: Replace print statements with standard library logging in a later session
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_dataset, load_csv
from src.clean_data import clean_housing_data
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_regression
from src.infer import run_inference
from src.utils import (
    load_config,
    make_dummy_ames_like_csv,
    fail_fast_feature_checks,
    ensure_parent_dir,
)


# ------------------------------
# CONFIGURATION — loaded from config.yaml
# ------------------------------
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def main() -> None:
    """
    Inputs:
    - None (reads all settings from config.yaml).
    Outputs:
    - Writes: data/processed/clean.csv, models/model.joblib,
      reports/predictions.csv
    Why this contract matters for reliable ML delivery:
    - A single entrypoint is easy to run in CI, easy to reproduce across
      machines, and easy to schedule/deploy.
    """
    print("[main] Starting pipeline")  # TODO: replace with logging later

    # ------------------------------
    # 0) Load config + resolve paths
    # ------------------------------
    print("[main] Step 0 - Load config and ensure output directories exist")
    cfg = load_config(_CONFIG_PATH)

    train_path = Path(cfg["data"]["raw"]["train_path"])
    test_path = Path(cfg["data"]["raw"]["test_path"])
    clean_out_path = Path(cfg["data"]["processed"]["clean_path"])
    model_out_path = Path(cfg["output"]["model_path"])
    preds_out_path = Path(cfg["output"]["predictions_path"])

    problem_type = cfg["pipeline"]["problem_type"]
    target_column = cfg["pipeline"]["target_column"]
    id_column = cfg["pipeline"]["id_column"]

    ensure_parent_dir(clean_out_path)
    ensure_parent_dir(model_out_path)
    ensure_parent_dir(preds_out_path)

    # ------------------------------
    # 1) Load (or create dummy)
    # ------------------------------
    print("[main] Step 1 - Load raw data")  # TODO: replace with logging later
    if not train_path.exists():
        make_dummy_ames_like_csv(train_path)

    data = load_dataset(train_path, test_path if test_path.exists() else None)
    df_train_raw = data["train"].df

    # No custom ingestion logic required for this dataset.
    # If your dataset requires special parsing (e.g., multiple CSVs,
    # nested JSON, SQL), implement that logic in src/load_data.py.

    # ------------------------------
    # 2) Clean (train)
    # ------------------------------
    print("[main] Step 2 - Clean training data")
    clean_train = clean_housing_data(
        df_train_raw,
        target_col=target_column,
        drop_cols=cfg["cleaning"]["drop_cols"],
        require_target=True,
    )
    X_all = clean_train.X
    y_all = clean_train.y
    if y_all is None:
        raise ValueError(
            "Cleaning returned y=None for training data. "
            "Check target_column in config.yaml."
        )

    # Save processed clean.csv (include target for traceability)
    print("[main] Saving processed clean CSV")
    df_clean_materialized = X_all.copy()
    df_clean_materialized[target_column] = y_all.values
    df_clean_materialized.to_csv(clean_out_path, index=False)

    # ------------------------------
    # 3) Validate (fail fast)
    # ------------------------------
    print("[main] Step 3 - Validate features dataframe")
    required_cols = list(
        dict.fromkeys(
            cfg["features"]["quantile_bin"]
            + cfg["features"]["categorical_onehot"]
            + cfg["features"]["numeric_passthrough"]
        )
    )
    validate_dataframe(X_all, required_cols)

    # ------------------------------
    # 4) Train/test split
    # ------------------------------
    print("[main] Step 4 - Train/test split")
    stratify = y_all if problem_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=cfg["train"]["test_size"],
            random_state=cfg["train"]["random_state"],
            stratify=stratify,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=cfg["train"]["test_size"],
            random_state=cfg["train"]["random_state"],
            stratify=None,
        )

    # ------------------------------
    # 5) Fail-fast feature config checks (post-split)
    # ------------------------------
    fail_fast_feature_checks(
        X_train,
        quantile_bin=cfg["features"]["quantile_bin"],
        categorical_onehot=cfg["features"]["categorical_onehot"],
        numeric_passthrough=cfg["features"]["numeric_passthrough"],
    )

    # ------------------------------
    # 6) Build preprocessor
    # ------------------------------
    print("[main] Step 6 - Build feature preprocessor")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=cfg["features"]["quantile_bin"],
        categorical_onehot_cols=cfg["features"]["categorical_onehot"],
        numeric_passthrough_cols=cfg["features"]["numeric_passthrough"],
        n_bins=int(cfg["features"]["n_bins"]),
    )

    # ------------------------------
    # 7) Train model
    # ------------------------------
    print("[main] Step 7 - Train model")
    model = train_model(X_train, y_train, preprocessor, problem_type)

    # Save artifact for infer.py (dict with pipeline + metadata)
    print("[main] Saving model artifact")
    artifact = {
        "pipeline": model,
        "metadata": {
            "problem_type": problem_type,
            "target_transform": (
                "log1p" if problem_type == "regression" else "none"
            ),
        },
    }
    joblib.dump(artifact, model_out_path)

    # ------------------------------
    # 8) Evaluate (console only)
    # ------------------------------
    print("[main] Step 8 - Evaluate on held-out test")
    if problem_type == "regression":
        # train.py trains y in log-space -> predictions are log-space
        y_pred_log = model.predict(X_test)
        y_pred_price = np.expm1(y_pred_log)
        y_true_price = y_test.astype(float).values

        metrics = evaluate_regression(
            y_true_price, y_pred_price, compute_rmsle=True
        )
        print(f"[main] Regression metrics (price-scale): {metrics}")
    else:
        print("[main] Classification: evaluation not wired yet.")

    # ------------------------------
    # 9) Inference on unseen data + save predictions
    # ------------------------------
    print("[main] Step 9 - Inference + save predictions")
    infer_input_path = Path(cfg["data"]["inference"]["input_path"])

    if infer_input_path.exists():
        print(f"[main] Loading inference data from: {infer_input_path}")
        infer_result = load_csv(infer_input_path)
        clean_infer = clean_housing_data(
            infer_result.df,
            target_col=target_column,
            drop_cols=[],  # keep Id so run_inference can include it in output
            require_target=False,
        )
        X_infer = clean_infer.X
        print(f"[main] Inference rows: {len(X_infer)}")
    else:
        print(
            f"[main] No inference file at {infer_input_path}; "
            "falling back to X_test split"
        )
        X_infer = X_test.copy()

    preds_df = run_inference(
        input_df=X_infer,
        artifact=artifact,
        id_col=id_column,
        pred_col=target_column,
    )

    ensure_parent_dir(str(preds_out_path))
    preds_df.to_csv(preds_out_path, index=False)
    print(f"[main] Saved predictions to: {preds_out_path}")


if __name__ == "__main__":
    main()
