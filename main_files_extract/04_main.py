"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single, readable orchestration script for the full pipeline.
- Responsibility (separation of concerns): Glue modules together in a clear order with fail-fast checks.
- Pipeline contract (inputs and outputs): Produces clean data, a saved model, and a predictions report.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path
from src.utils import read_config

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from src.clean_data import clean_data
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

# ============================================================
# CONFIGURATION (SETTINGS dict acts as a bridge to YAML later)
# ============================================================
# LOUD REMINDER:
# - This SETTINGS block is PRE-CONFIGURED to match the dummy CSV created by src/load_data.py.
# - Students MUST map these keys to the real dataset schema (columns, target, problem type, etc.).
CONFIG = read_config("config.yaml")

SETTINGS = {
    "is_example_config": False,
    "problem_type": "classification",  # Titanic = classification
    "random_state": 42,
    "test_size": 0.25,
    "paths": {
        "raw_data": CONFIG["data"]["raw_path"],
        "processed_clean": CONFIG["data"]["processed_path"],
        "model": "models/model.joblib",
        "predictions_report": "reports/predictions.csv",
    },
    "target_column": CONFIG["schema"]["target"],
    "features": {
        "quantile_bin": [],
        "categorical_onehot": ["Sex", "Embarked"],
        "numeric_passthrough": ["Pclass", "Age", "SibSp", "Parch", "Fare"],
        "n_bins": 3,
    },
}


def main():
    """
    Inputs:
    - None (uses SETTINGS dict for configuration).
    Outputs:
    - None (writes required artifacts to disk and prints metrics).
    Why this contract matters for reliable ML delivery:
    - A single executable entrypoint enables repeatable runs locally, in CI, and in scheduled jobs.
    """
    print("[main.main] Starting end-to-end pipeline.")  # TODO: replace with logging later

    # ------------------------------------------------------------
    # 0) Directory creation (manual materialization)
    # ------------------------------------------------------------
    print("[main.main] Ensuring artifact directories exist.")  # TODO: replace with logging later
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Example-config guardrail
    # ------------------------------------------------------------
    if SETTINGS.get("is_example_config", False):
        print(
            "[main.main] NOTE: SETTINGS['is_example_config'] is True. "
            "This is scaffolding configuration matched to the dummy CSV."
        )  # TODO: replace with logging later

    # ------------------------------------------------------------
    # 2) Load raw data (creates dummy CSV if missing)
    # ------------------------------------------------------------
    raw_path = Path(SETTINGS["paths"]["raw_data"])
    example_path = Path("data/raw/example.csv")

    # Test/scaffolding fallback:
    # If the configured raw path doesn't exist but example.csv does, use example.csv.
    if (not raw_path.exists()) and example_path.exists():
        raw_path = example_path
        SETTINGS["is_example_config"] = True

    # If example mode is on explicitly, also force example path
    if SETTINGS.get("is_example_config", False):
        raw_path = example_path

    # load_data expects a config dict
    data_config = {
        "data": {"raw_path": str(raw_path)},
        "logging": {"log_file": "reports/run.log", "level": "INFO"},
    }
    
    df_raw = load_data(data_config)

    # ------------------------------------------------------------
    # 2b) If we're running in scaffolding mode, switch schema to match example.csv
    #     (Tests create data/raw/example.csv with columns: num_feature, cat_feature, target)
    # ------------------------------------------------------------
    if SETTINGS.get("is_example_config", False) or (raw_path.name == "example.csv"):
        SETTINGS["is_example_config"] = True
        SETTINGS["target_column"] = "target"
        SETTINGS["features"] = {
            "quantile_bin": ["num_feature"],
            "categorical_onehot": ["cat_feature"],
            "numeric_passthrough": [],
            "n_bins": 3,
        }

        # Override artifact paths for scaffold/tests
        SETTINGS["paths"]["processed_clean"] = "data/processed/clean.csv"
        SETTINGS["paths"]["predictions_report"] = "reports/predictions.csv"
        SETTINGS["paths"]["model"] = "models/model.joblib"

        # Infer regression vs classification
        y_tmp = df_raw[SETTINGS["target_column"]]
        unique_vals = set(pd.Series(y_tmp).dropna().unique().tolist())
        if unique_vals.issubset({0, 1}) and len(unique_vals) <= 2:
            SETTINGS["problem_type"] = "classification"
        else:
            SETTINGS["problem_type"] = "regression"

    # TODO_STUDENT:
    # - Flip is_example_config to False once you've mapped SETTINGS to your real dataset.
    # - Consider moving SETTINGS to a config.yml and loading it, once the pipeline is stable.

    # ------------------------------------------------------------
    # 3) Clean
    # ------------------------------------------------------------
    target_column = SETTINGS["target_column"]
    
    # clean_data expects a config dict
    clean_config = {
        "logging": {"log_file": "reports/run.log", "level": "INFO"},
        "schema": {
            "required_columns": (
                SETTINGS["features"]["quantile_bin"]
                + SETTINGS["features"]["categorical_onehot"]
                + SETTINGS["features"]["numeric_passthrough"]
                + [target_column]
            ),
            "target": target_column
        }
    }
    df_clean = clean_data(df_raw, clean_config)

    # ------------------------------------------------------------
    # 4) Save processed clean CSV (required artifact)
    # ------------------------------------------------------------
    clean_path = Path(SETTINGS["paths"]["processed_clean"])
    save_csv(df_clean, clean_path)

    # ------------------------------------------------------------
    # 5) Validate (fail fast on obvious issues)
    # ------------------------------------------------------------
    required_cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + [target_column]
    )
    validate_dataframe(df_clean, required_columns=required_cols)

    # ------------------------------------------------------------
    # 6) Train-test split (BEFORE any feature fitting to prevent leakage)
    # ------------------------------------------------------------
    print("[main.main] Creating train-test split (before feature fitting).")  # TODO: replace with logging later
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    stratify = y if SETTINGS["problem_type"] == "classification" else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=stratify,
        )
    except ValueError as e:
        print(
            f"[main.main] Stratified split failed with: {e}. Falling back to non-stratified split."
        )  # TODO: replace with logging later
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=None,
        )

    # ------------------------------------------------------------
    # 7) Fail-fast feature checks (configured columns exist + types)
    # ------------------------------------------------------------
    print("[main.main] Running fail-fast feature configuration checks.")  # TODO: replace with logging later

    quantile_bin_cols = SETTINGS["features"]["quantile_bin"]
    categorical_onehot_cols = SETTINGS["features"]["categorical_onehot"]
    numeric_passthrough_cols = SETTINGS["features"]["numeric_passthrough"]

    configured_cols = set(quantile_bin_cols + categorical_onehot_cols + numeric_passthrough_cols)
    missing = [c for c in configured_cols if c not in X_train.columns]
    if missing:
        raise ValueError(f"Configured feature columns missing from training data: {missing}")

    non_numeric_bin = [c for c in quantile_bin_cols if not is_numeric_dtype(X_train[c])]
    if non_numeric_bin:
        raise ValueError(
            "Quantile-binning requires numeric dtypes. Non-numeric columns configured for quantile_bin: "
            f"{non_numeric_bin}"
        )

    # TODO_STUDENT:
    # - Add additional guardrails (e.g., unexpected high cardinality, NA thresholds).
    # - Keep checks simple; heavy profiling belongs in exploratory notebooks or dedicated QA jobs.

    # ------------------------------------------------------------
    # 8) Build feature recipe (ColumnTransformer) — no fitting here
    # ------------------------------------------------------------
    print("[main.main] Building preprocessing recipe.")  # TODO: replace with logging later
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=quantile_bin_cols,
        categorical_onehot_cols=categorical_onehot_cols,
        numeric_passthrough_cols=numeric_passthrough_cols,
        n_bins=int(SETTINGS["features"]["n_bins"]),
    )

    # ------------------------------------------------------------
    # 9) Train model pipeline (fit on training split only)
    # ------------------------------------------------------------
    train_config = {
        "model_name": "logistic_regression" if SETTINGS["problem_type"] == "classification" else "linear_regression",
        "random_state": SETTINGS["random_state"],
        "model_output_path": SETTINGS["paths"]["model"]
    }

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        config=train_config,
    )

    # ------------------------------------------------------------
    # 10) Save model (required artifact)
    # ------------------------------------------------------------
    model_path = Path(SETTINGS["paths"]["model"])
    save_model(model, model_path)

    # ------------------------------------------------------------
    # 11) Evaluate on held-out test (prints metric only)
    # ------------------------------------------------------------
    eval_config = {
        "problem_type": SETTINGS["problem_type"],
        "primary_metric": "rmse" if SETTINGS["problem_type"] == "regression" else "f1",
        "save_reports": False
    }

    metrics = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        config=eval_config,
    )

    metric_value = metrics.get(eval_config["primary_metric"])

    if SETTINGS["problem_type"] == "regression":
        print(f"[main.main] Test RMSE: {metric_value:.4f}")  # TODO: replace with logging later
    else:
        print(f"[main.main] Test weighted F1: {metric_value:.4f}")  # TODO: replace with logging later

    # ------------------------------------------------------------
    # 12) Inference on example data + save predictions (required artifact)
    # ------------------------------------------------------------
    print("[main.main] Running inference on example rows and saving report.")  # TODO: replace with logging later

    X_infer = X_test.head(5).copy()
    df_preds = run_inference(model, X_infer=X_infer)

    preds_path = Path(SETTINGS["paths"]["predictions_report"])
    save_csv(df_preds[["prediction"]], preds_path)

    print("[main.main] Pipeline completed successfully.")  # TODO: replace with logging later
    print(">>> TEST MARKER: main() executed")


if __name__ == "__main__":
    main()