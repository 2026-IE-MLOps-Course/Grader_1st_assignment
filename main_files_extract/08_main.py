"""
Educational Goal:
- Orchestrate the full ML pipeline (load -> clean -> validate -> split -> train -> eval -> infer -> save).
- Enforce split-first boundaries to prevent leakage.
- Produce consistent artifacts:
  - data/processed/clean.csv
  - models/model.joblib (pickle content is acceptable per guidelines)
  - reports/predictions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import get_logger, save_csv, save_model
from src.validate import validate_dataframe


# ========================================================
# CONFIGURATION (SETTINGS dictionary bridge)
# ========================================================
SETTINGS = {
    "is_example_config": True,
    "problem_type": "classification",  # "regression" or "classification"
    "target_column": "target",
    "raw_data_path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "processed_data_path": "data/processed/clean.csv",
    "model_path": "models/model.joblib",  # pickle content acceptable
    "predictions_path": "reports/predictions.csv",
    "random_state": 42,
    # 3-way split (enforced early)
    "test_size": 0.2,
    "val_size": 0.2,  # % of the *remaining train* after test split
    "features": {
        "quantile_bin": ["num_feature"],
        "categorical_onehot": ["cat_feature"],
        "numeric_passthrough": [],
        "n_bins": 3,
    },

}


def _ensure_dirs(logger) -> None:
    for p in ["data/raw", "data/processed", "models", "reports", "logs"]:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", p)


def _maybe_switch_to_telco(logger) -> None:
    telco_repo_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    telco_alt_path = Path("/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if (not telco_repo_path.exists()) and telco_alt_path.exists():
        logger.info("Found Telco CSV at /mnt/data; copying into data/raw")
        df_alt = pd.read_csv(telco_alt_path)
        save_csv(df_alt, telco_repo_path)

    if telco_repo_path.exists():
        logger.info("Telco dataset detected. Switching SETTINGS to Telco schema.")
        SETTINGS["is_example_config"] = False
        SETTINGS["problem_type"] = "classification"
        SETTINGS["target_column"] = "Churn"
        SETTINGS["raw_data_path"] = str(telco_repo_path)
        SETTINGS["features"] = {
            "quantile_bin": ["tenure", "MonthlyCharges", "TotalCharges"],
            "numeric_passthrough": ["SeniorCitizen"],
            "categorical_onehot": [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
            ],
            "n_bins": 5,
        }
        SETTINGS["schema"] = {
            "gender": {'type': 'categorical', 'accept_nan': False},
            "SeniorCitizen": {'type': 'numeric', 'accept_nan': False},
            "Partner": {'type': 'categorical', 'accept_nan': False},
            "Dependents": {'type': 'categorical', 'accept_nan': False},
            "tenure": {'type': 'numeric', 'accept_nan': False},
            "PhoneService": {'type': 'categorical', 'accept_nan': False},
            "MultipleLines": {'type': 'categorical', 'accept_nan': False},
            "InternetService": {'type': 'categorical', 'accept_nan': False},
            "OnlineSecurity": {'type': 'categorical', 'accept_nan': False},
            "OnlineBackup": {'type': 'categorical', 'accept_nan': False},
            "DeviceProtection": {'type': 'categorical', 'accept_nan': False},
            "TechSupport": {'type': 'categorical', 'accept_nan': False},
            "StreamingTV": {'type': 'categorical', 'accept_nan': False},
            "StreamingMovies": {'type': 'categorical', 'accept_nan': False},
            "Contract": {'type': 'categorical', 'accept_nan': False},
            "PaperlessBilling": {'type': 'categorical', 'accept_nan': False},
            "PaymentMethod": {'type': 'categorical', 'accept_nan': False},
            "MonthlyCharges": {'type': 'numeric', 'accept_nan': False},
            "TotalCharges": {'type': 'numeric', 'accept_nan': False},
        }
        SETTINGS["target_config"] = {
            'column': 'Churn', 'type': 'classification', 'allowed_classes': [1,0]
        }


def _fail_fast_feature_checks(
    X: pd.DataFrame,
    target_col: str,
    feature_cfg: dict,
) -> list[str]:
    configured_feature_cols = (
        feature_cfg.get("quantile_bin", [])
        + feature_cfg.get("categorical_onehot", [])
        + feature_cfg.get("numeric_passthrough", [])
    )

    if target_col not in X.columns and target_col in X.columns:
        # defensive (normally unreachable)
        raise ValueError(f"Target column '{target_col}' should not be in X.")

    missing_in_X = [c for c in configured_feature_cols if c not in X.columns]
    if missing_in_X:
        raise ValueError(
            "Configured feature columns missing in X. "
            f"Missing: {missing_in_X}. "
            "Update SETTINGS['features'] to match your dataset."
        )

    for c in feature_cfg.get("quantile_bin", []):
        if not pd.api.types.is_numeric_dtype(X[c]):
            raise ValueError(
                f"Column '{c}' is configured for quantile binning but is not numeric. "
                "Fix cleaning or change SETTINGS['features']['quantile_bin']."
            )

    return configured_feature_cols


def main() -> None:
    logger = get_logger("main")

    logger.info("Starting pipeline")
    try:
        _ensure_dirs(logger)
        _maybe_switch_to_telco(logger)

        if SETTINGS.get("is_example_config", False):
            logger.warning(
                "Running EXAMPLE config (dummy dataset schema). "
                "To use real data, update SETTINGS."
            )

        # Step 1) Load
        raw_path = Path(SETTINGS["raw_data_path"])
        logger.info("Loading raw data from: %s", raw_path)
        df_raw = load_raw_data(raw_path)
        logger.info("Raw shape: %s", df_raw.shape)
        
        # Step 2) Clean
        logger.info("Cleaning data (target=%s)", SETTINGS["target_column"])
        df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])
        logger.info("Clean shape: %s", df_clean.shape)
        
        # Step 3) Save processed CSV
        processed_path = Path(SETTINGS["processed_data_path"])
        logger.info("Saving processed data to: %s", processed_path)
        save_csv(df_clean, processed_path)
        
        # Step 4) Validate (security gate immediately after cleaning)
        target_col = SETTINGS["target_column"]
        feature_cfg = SETTINGS["features"]
        configured_feature_cols = (
            feature_cfg.get("quantile_bin", [])
            + feature_cfg.get("categorical_onehot", [])
            + feature_cfg.get("numeric_passthrough", [])
        )
        schema=SETTINGS['schema']
        target_config=SETTINGS["target_config"]
        required_cols = list(set(configured_feature_cols + [target_col]))

        logger.info("Validating required columns: %s", required_cols)
        validate_dataframe(df=df_clean, schema=schema, target_config=target_config)
        
        # Step 5) 3-way split EARLY (Train / Val / Test)
        logger.info("Splitting data into Train/Val/Test (leakage-safe)")
        X_all = df_clean.drop(columns=[target_col])
        y_all = df_clean[target_col]
        
        stratify = y_all if SETTINGS["problem_type"] == "classification" else None

        # First split off test
        try:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X_all,
                y_all,
                test_size=SETTINGS["test_size"],
                random_state=SETTINGS["random_state"],
                stratify=stratify,
            )
        except ValueError:
            logger.warning("Stratified split failed; falling back to non-stratified")
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X_all,
                y_all,
                test_size=SETTINGS["test_size"],
                random_state=SETTINGS["random_state"],
                stratify=None,
            )
        
        # Then split trainval into train and val
        stratify_tv = y_trainval if SETTINGS["problem_type"] == "classification" else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=SETTINGS["val_size"],
                random_state=SETTINGS["random_state"],
                stratify=stratify_tv,
            )
        except ValueError:
            logger.warning("Stratified val split failed; falling back to non-stratified")
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=SETTINGS["val_size"],
                random_state=SETTINGS["random_state"],
                stratify=None,
            )

        logger.info(
            "Split sizes -> train=%d, val=%d, test=%d",
            len(X_train),
            len(X_val),
            len(X_test),
        )
        
        # Step 6) Fail-fast feature checks
        logger.info("Running fail-fast feature checks")
        _fail_fast_feature_checks(X_all, target_col, feature_cfg)
        
        # Step 7) Build feature recipe (blueprint only; training will fit)
        logger.info("Building feature preprocessor (unfitted)")
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=feature_cfg.get("quantile_bin", []),
            categorical_onehot_cols=feature_cfg.get("categorical_onehot", []),
            numeric_passthrough_cols=feature_cfg.get("numeric_passthrough", [])#,
            # n_bins=feature_cfg.get("n_bins", 3),
        )
        
        # Step 8) Train (fit ONLY on train)
        # param_grid=param_grid
        logger.info("Training model (fit only on TRAIN split)")
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            problem_type=SETTINGS["problem_type"],
            # param_grid=param_grid
        )
        
        # Step 9) Save model artifact
        model_path = Path(SETTINGS["model_path"])
        logger.info("Saving model artifact to: %s", model_path)
        save_model(model, model_path)
        
        # Step 10) Evaluate on VAL (test stays vaulted)
        logger.info("Evaluating on VAL split")
        metric_value = evaluate_model(
            model=model,
            X_test=X_val,
            y_test=y_val,
            problem_type=SETTINGS["problem_type"],
        )
        logger.info("Validation metric = %s", metric_value)
        
        # Step 11) Inference on TEST sample (demo output)
        logger.info("Running inference on TEST sample and saving predictions")
        X_infer = X_test.head(20)
        df_pred = run_inference(model, X_infer)
        save_csv(df_pred, Path(SETTINGS["predictions_path"]))

        logger.info("Pipeline complete ✅")
        
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        # Make failures obvious in CI
        sys.exit(1)


if __name__ == "__main__":
    main()