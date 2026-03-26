# src/main.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Orchestrate the pipeline in a readable entry point
- Responsibility (separation of concerns): Coordinate steps, handle splits, inject configuration, and delegate work to modules
- Pipeline contract (inputs and outputs): Produces a cleaned dataset, a trained pipeline artifact, and an inference predictions artifact

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

# --------------------------------------------------------
# PATHS & CONFIGURATION
# --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "opiod_raw_data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
INFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "inference" / "opioid_infer_01.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "predictions.csv"

BINARY_SUM_COLS = [
    "A", "B", "C", "D", "E", "F",
    "H", "I", "J", "K", "L", "M", "N",
    "R", "S", "T",
    "Low_inc", "SURG",
]

SETTINGS = {
    "is_example_config": False,
    "target_column": "OD",
    "problem_type": "classification",
    "split": {"test_size": 0.05, "val_size": 0.15, "random_state": 42},
    "features": {
        "quantile_bin": ["rx_ds"],
        "categorical_onehot": [],
        "numeric_passthrough": [],
        "binary_sum_cols": BINARY_SUM_COLS,
        "n_bins": 4,
    },
    "validation": {
        "numeric_non_negative_cols": ["rx_ds"],
    },
}


def _three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
):
    """
    Why this contract matters for reliable ML delivery
    - Train learns, validation guides decisions, test audits the final result
    """
    if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
        raise ValueError(
            "Fatal: split sizes must satisfy 0 < test_size, 0 < val_size, and test_size + val_size < 1"
        )

    stratify_y = y if stratify else None

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )

        relative_val_size = val_size / (1.0 - test_size)
        stratify_temp = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=stratify_temp,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except ValueError as e:
        print(
            f"[main] Warning: Stratified split failed: {e}. Falling back to random split.")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


def _get_feature_columns_from_settings() -> list[str]:
    cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["binary_sum_cols"]
    )
    return list(dict.fromkeys(cols))


def main():
    print("[main.main] Starting pipeline")

    if SETTINGS.get("is_example_config", False):
        raise ValueError(
            "Fatal: SETTINGS is an example. Update target_column and feature lists for your dataset, then set 'is_example_config': False"
        )

    # 1) LOAD
    print("[main.main] 1) LOAD")
    df_raw = load_raw_data(RAW_DATA_PATH)

    # 2) CLEAN (training mode, target required)
    print("[main.main] 2) CLEAN (TRAINING DATA)")
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])

    # 3) SAVE PROCESSED CSV
    print("[main.main] 3) SAVE PROCESSED CSV")
    save_csv(df_clean, CLEAN_DATA_PATH)

    # 4) VALIDATE (training mode)
    print("[main.main] 4) VALIDATE (TRAINING DATA)")
    required_columns = [SETTINGS["target_column"]] + \
        _get_feature_columns_from_settings()

    validate_dataframe(
        df=df_clean,
        required_columns=required_columns,
        check_missing_values=False,
        target_column=SETTINGS["target_column"],
        target_allowed_values=[
            0, 1] if SETTINGS["problem_type"] == "classification" else None,
        numeric_non_negative_cols=SETTINGS["validation"]["numeric_non_negative_cols"],
    )

    # 5) SPLIT
    print("[main.main] 5) SPLIT INTO TRAIN, VALIDATION, TEST")
    X_full = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    identifier_col = "ID" if "ID" in X_full.columns else None
    if identifier_col:
        X_full_no_id = X_full.drop(columns=[identifier_col])
    else:
        X_full_no_id = X_full

    X_train, X_val, X_test, y_train, y_val, y_test = _three_way_split(
        X_full_no_id,
        y,
        test_size=SETTINGS["split"]["test_size"],
        val_size=SETTINGS["split"]["val_size"],
        random_state=SETTINGS["split"]["random_state"],
        stratify=(SETTINGS["problem_type"] == "classification"),
    )

    print("[main.main] Split sizes")
    print("Train:", X_train.shape, "Validation:",
          X_val.shape, "Test:", X_test.shape)

    if len(X_test) == 0:
        raise ValueError(
            "Fatal: test split is empty. Check split ratios and dataset size.")

    # 6) FAIL FAST FEATURE CHECKS
    configured_cols = _get_feature_columns_from_settings()
    if not configured_cols:
        raise ValueError(
            "Fatal: No feature columns configured in SETTINGS['features']")

    missing = set(configured_cols) - set(X_train.columns)
    if missing:
        raise ValueError(
            f"Fatal: Configured columns not found in dataset: {sorted(missing)}")

    for col in SETTINGS["features"]["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise ValueError(
                f"Fatal: Column '{col}' must be numeric for quantile binning. Found dtype={X_train[col].dtype}"
            )

    # 7) BUILD FEATURE RECIPE
    print("[main.main] 7) BUILD FEATURE RECIPE")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        binary_sum_cols=SETTINGS["features"]["binary_sum_cols"],
        n_bins=SETTINGS["features"]["n_bins"],
    )

    # 8) TRAIN
    print("[main.main] 8) TRAIN")
    model_pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=SETTINGS["problem_type"],
    )

    # 8.5) EVALUATE
    print("[main.main] 8.5) EVALUATE (VALIDATION)")
    val_metrics = evaluate_model(
        model=model_pipeline,
        X_eval=X_val,
        y_eval=y_val,
        problem_type=SETTINGS["problem_type"],
    )
    print(f"[main.main] Validation metrics={val_metrics}")

    # 9) SAVE MODEL
    print("[main.main] 9) SAVE MODEL")
    save_model(model_pipeline, MODEL_PATH)

    # 10) INFERENCE (NEW DATA FILE)
    print("[main.main] 10) INFERENCE (NEW DATA FILE)")

    if not INFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Inference file not found at {INFERENCE_DATA_PATH}. "
            "Place a file like opioid_infer_01.csv under data/inference"
        )

    # Load new unseen data
    df_infer_raw = load_raw_data(INFERENCE_DATA_PATH)

    # Clean in inference mode (no target required)
    df_infer_clean = clean_dataframe(df_infer_raw, target_column=None)

    # Validate feature contract only
    infer_required_columns = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["binary_sum_cols"]
    )
    infer_required_columns = list(dict.fromkeys(infer_required_columns))

    validate_dataframe(
        df=df_infer_clean,
        required_columns=infer_required_columns,
        check_missing_values=False,
        target_column=None,
        target_allowed_values=None,
        numeric_non_negative_cols=SETTINGS["validation"]["numeric_non_negative_cols"],
    )

    # Keep ID for traceability but never send it into the model
    identifier_col = "ID" if "ID" in df_infer_clean.columns else None
    X_infer = df_infer_clean.drop(
        columns=[identifier_col]) if identifier_col else df_infer_clean

    # Run inference through the trained pipeline artifact
    df_predictions = run_inference(
        model=model_pipeline,
        X_infer=X_infer,
        include_proba=(SETTINGS["problem_type"] == "classification"),
    )

    # Re-attach ID for audit joins
    if identifier_col:
        df_predictions.insert(
            0, identifier_col, df_infer_clean[identifier_col].values)

    print("[main.main] Inference results")
    print(df_predictions.head(10))

    # Persist predictions as artifact
    save_csv(df_predictions, PREDICTIONS_PATH)

    print(f"[main.main] Wrote predictions artifact to {PREDICTIONS_PATH}")

    print("[main.main] Done")
    print(f"[main.main] Wrote {CLEAN_DATA_PATH}")
    print(f"[main.main] Wrote {MODEL_PATH}")
    print(f"[main.main] Wrote {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
