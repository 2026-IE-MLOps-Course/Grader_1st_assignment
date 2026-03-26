"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single, readable “pipeline story” that runs end-to-end from raw data to saved artifacts.
- Responsibility (separation of concerns): Orchestrate steps and materialize artifacts (processed data, trained model, predictions).
- Pipeline contract (inputs and outputs): Input is repository paths + SETTINGS; outputs are saved artifacts and printed metrics.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

# 1) Imports
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.validate import validate_dataframe
from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import load_csv, save_csv, save_model

# 2) CONFIGURATION (SETTINGS dictionary bridge)
# LOUD REMINDER:
# This SETTINGS dict is a temporary bridge until config.yml is introduced.
# You MUST map these fields to your real dataset columns and business requirements.
SETTINGS = {
    "is_example_config": False,
    "problem_type": "regression",
    "random_state": 42,
    "test_size": 0.25,
    "target_column": "rent",
    "paths": {
        "raw_data": "data/raw/Houses_for_rent.csv",
        "processed_data": "data/processed/clean.csv",
        "model": "models/model.joblib",
        "predictions": "reports/predictions.csv",
        "inference": "data/inference/Houses_for_rent_inference.csv"
    },
    "features": {
        # Pre-configured to match the dummy CSV created by src/load_data.py
        "quantile_bin": [],  # keep empty by default; dummy numeric feature passes through
        "categorical_onehot": ["district"],
        "numeric_passthrough": [
            "sq.mt",
            "floor",
            "bedrooms",
            "outer",
            "duplex",
            "cottage",
            "elevator",
            "penthouse",
            "semidetached"
            ],
        "n_bins": 3,
    },
}


def main():
    """
    Inputs:
    - None (reads SETTINGS and files from repo-relative paths)
    Outputs:
    - None (side effects: writes clean.csv, model.joblib, predictions.csv; prints metric)
    Why this contract matters for reliable ML delivery:
    - A single entrypoint makes execution consistent across laptops, CI, and future schedulers (Airflow, Prefect, etc.).
    """
    print("[main.main] Starting end-to-end pipeline")  # TODO: replace with logging later

    # --------------------------------------------------------
    # Step 0: Ensure output directories exist (manual materialization only)
    # --------------------------------------------------------
    print("[main.main] Ensuring required directories exist")  # TODO: replace with logging later
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("data/inference").mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Step 1: Example-config check
    # --------------------------------------------------------
    if SETTINGS.get("is_example_config", False):
        print("LOUD NOTE: SETTINGS is using the example dummy configuration.")  # TODO: replace with logging later
        print("Update SETTINGS to match your real dataset before production use.")  # TODO: replace with logging later

    # Resolve paths
    raw_path = Path(SETTINGS["paths"]["raw_data"])
    processed_path = Path(SETTINGS["paths"]["processed_data"])
    model_path = Path(SETTINGS["paths"]["model"])
    preds_path = Path(SETTINGS["paths"]["predictions"])
    inference_path = Path(SETTINGS["paths"]["inference"])

    target_column = SETTINGS["target_column"]
    problem_type = SETTINGS["problem_type"]

    # --------------------------------------------------------
    # Step 2: Load
    # --------------------------------------------------------
    print("[main.main] Loading raw data")  # TODO: replace with logging later
    df_raw = load_raw_data(raw_path)

    # --------------------------------------------------------
    # Step 3: Clean
    # --------------------------------------------------------
    print("[main.main] Cleaning data")  # TODO: replace with logging later

    feature_cfg = SETTINGS["features"]
    configured_feature_cols = (
        feature_cfg.get("quantile_bin", [])
        + feature_cfg.get("categorical_onehot", [])
        + feature_cfg.get("numeric_passthrough", [])
    )
    required_columns = list(
        dict.fromkeys(configured_feature_cols + [target_column])
    )
    
    df_clean = clean_dataframe(
        df_raw,
        target_column=target_column,
        required_columns=required_columns
    )

    # --------------------------------------------------------
    # Step 4: Save processed CSV (artifact requirement)
    # --------------------------------------------------------
    print(f"[main.main] Saving processed data to {processed_path}")  # TODO: replace with logging later
    save_csv(df_clean, processed_path)

    # --------------------------------------------------------
    # Step 5: Validate
    # --------------------------------------------------------
    print("[main.main] Validating cleaned data")  # TODO: replace with logging later

    validate_dataframe(df_clean, required_columns=required_columns)

    # --------------------------------------------------------
    # Step 6: Train/test split (BEFORE any feature fitting to prevent leakage)
    # --------------------------------------------------------
    print("[main.main] Splitting train/test")  # TODO: replace with logging later
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    stratify = y if problem_type == "classification" else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=stratify,
        )
    except ValueError as e:
        print(f"[main.main] Stratified split failed ({e}); falling back to non-stratified split.")  # TODO: replace with logging later
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=None,
        )

    # --------------------------------------------------------
    # Step 7: Fail-fast feature checks (columns exist + quantile_bin cols are numeric)
    # --------------------------------------------------------
    print("[main.main] Running fail-fast feature configuration checks")  # TODO: replace with logging later

    missing_cols = [
        c for c in configured_feature_cols if c not in X_train.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Feature config error: these configured feature columns are missing from X_train: {missing_cols}. "
            "Update SETTINGS['features'] to match your dataset."
        )

    # Explicitly check that quantile_bin columns are numeric
    quantile_cols = feature_cfg.get("quantile_bin", [])
    for c in quantile_cols:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            raise ValueError(
                f"Feature config error: column '{c}' is in SETTINGS['features']['quantile_bin'] but is not numeric. "
                "Move it to categorical_onehot or fix the dtype in cleaning."
            )

    # --------------------------------------------------------
    # Step 8: Build feature recipe (unfitted ColumnTransformer)
    # --------------------------------------------------------
    print("[main.main] Building feature preprocessor recipe")  # TODO: replace with logging later
    preprocessor = get_feature_preprocessor(
        bin_cols=feature_cfg.get("quantile_bin", []),
        categorical_cols=feature_cfg.get("categorical_onehot", []),
        numeric_cols=feature_cfg.get("numeric_passthrough", []),
        n_bins=int(feature_cfg.get("n_bins", 3)),
    )

    # --------------------------------------------------------
    # Step 9: Train model (Pipeline fits preprocess+model on TRAIN only)
    # --------------------------------------------------------
    print("[main.main] Training model")  # TODO: replace with logging later
     
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=problem_type
    )

    # --------------------------------------------------------
    # Step 10: Save model (artifact requirement)
    # --------------------------------------------------------
    print(f"[main.main] Saving model to {model_path}")  # TODO: replace with logging later
    save_model(model, model_path)

    # --------------------------------------------------------
    # Step 11: Evaluate
    # --------------------------------------------------------
    print("[main.main] Evaluating model")  # TODO: replace with logging later
    metric_value = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type
    )

    if problem_type == "regression":
        print(f"[main.main] Held-out RMSE: {metric_value}")  # TODO: replace with logging later
    else:
        print(f"[main.main] Held-out weighted F1: {metric_value}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # Step 12: Inference on example data + save predictions (artifact requirement)
    # --------------------------------------------------------
    print("[main.main] Running inference on held-out test features and saving predictions")  # TODO: replace with logging later
    df_infer = load_csv(inference_path)
    df_infer_clean = clean_dataframe(
        df_raw=df_infer,
        target_column=SETTINGS['target_column'],
        required_columns=required_columns
    )
    X_infer = df_infer_clean[required_columns]
    df_pred = run_inference(model=model, X_infer=X_infer)
    save_csv(df_pred, preds_path)

    print("[main.main] Pipeline complete. Artifacts created:")  # TODO: replace with logging later
    print(f" - {processed_path}")  # TODO: replace with logging later
    print(f" - {model_path}")  # TODO: replace with logging later
    print(f" - {preds_path}")  # TODO: replace with logging later


if __name__ == "__main__":
    main()
