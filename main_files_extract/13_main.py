"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow
        (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
    config.yml in a later session
"""

# ===========================================================================
# 1. IMPORTS
# ===========================================================================
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.utils import save_csv, save_model

# Get kmeans functions
from src.kmeans.kmeans_evaluate import evaluate_kmeans_model
from src.kmeans.kmeans_infer import run_kmeans_inference
from src.kmeans.kmeans_train import train_kmeans_model

# Get logitic regression functions
from src.logit_regression.logit_train import train_logit_model
from src.logit_regression.logit_evaluate import evaluate_logit_model
from src.logit_regression.logit_infer import run_logit_inference

# Get decision tree functions
from src.dtrees.dtrees_train import train_dtrees_model
from src.dtrees.dtrees_eval import evaluate_dtrees_model
from src.dtrees.dtrees_infer import run_dtrees_inference

# ===========================================================================
# 2. CONFIGURATION  —  SETTINGS BRIDGE
# ===========================================================================
SETTINGS = {
    "raw_data_path":       Path("data/raw/train.csv"),
    "processed_data_path": Path("data/processed/clean.csv"),
    "kmeans_model_path":   Path("models/kmeans_model.joblib"),
    "logit_model_path":    Path("models/logit_model.joblib"),
    "dtrees_model_path":    Path("models/dtrees_model.joblib"),
    "kmeans_predictions_path":    Path("reports/kmeans_predictions.csv"),
    "logit_predictions_path":    Path("reports/logit_predictions.csv"),
    "dtrees_predictions_path":    Path("reports/dtrees_predictions.csv"),
    "target_column": "Heart Disease",
    "problem_type": "classification",
    "test_size":    0.2,
    "random_state": 42,
    "k_means_bins": 3,
    "max_iterations": 2000,
    "features": {
        "quantile_bin":        [],
        "categorical_onehot":  [],
        "numeric_passthrough": ["id", "Sex", "FBS over 120",
                                "EKG results", "Exercise angina",
                                "ST depression"],
        "ordinal_encode": ["Chest pain type", "Number of vessels fluro",
                           "Thallium", "Slope of ST"],
        "min_max_scaler": ["Age", "BP", "Cholesterol", "Max HR"],
        "n_bins": 3
    }
}


# ===========================================================================
# 3. MAIN PIPELINE
# ===========================================================================
def main():
    """
    Inputs:
    - SETTINGS dictionary defined at module level (acts as config bridge)
    Outputs:
    - data/processed/clean.csv   — cleaned dataset written to disk
    - models/model.joblib        — trained sklearn Pipeline written to disk
    - reports/predictions.csv   — inference output written to disk
    """
    # -------------------------------------------------------------------
    # STEP 0: Ensure output directories exist
    # -------------------------------------------------------------------
    print("\n[main] Creating output directories if they do not exist...")
    # TODO: replace with logging later
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # STEP 1: Load raw data
    # -------------------------------------------------------------------
    print("[main] Step 1 — Loading raw data...")
    # TODO: replace with logging later
    df_raw = load_raw_data(SETTINGS["raw_data_path"])

    # -------------------------------------------------------------------
    # STEP 2: Clean data
    # -------------------------------------------------------------------
    print("[main] Step 2 — Cleaning data...")
    # TODO: replace with logging later
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])

    # -------------------------------------------------------------------
    # STEP 3: Save processed CSV
    # -------------------------------------------------------------------
    print("[main] Step 3 — Saving processed CSV...")
    # TODO: replace with logging later
    save_csv(df_clean, SETTINGS["processed_data_path"])

    # -------------------------------------------------------------------
    # STEP 4: Validate cleaned data
    # -------------------------------------------------------------------
    print("[main] Step 4 — Validating cleaned data...")
    # TODO: replace with logging later
    required_cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["ordinal_encode"]
        + SETTINGS["features"]["min_max_scaler"]
        + [SETTINGS["target_column"]]
    )
    validate_dataframe(df_clean, required_columns=required_cols)

    # -------------------------------------------------------------------
    # STEP 5: Train / test split  (MUST happen BEFORE feature fitting)
    # -------------------------------------------------------------------
    print("[main] Step 5 — Splitting into train and test sets...")
    # TODO: replace with logging later
    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    stratify_arg = None
    if SETTINGS["problem_type"] == "classification":
        try:
            stratify_arg = y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=SETTINGS["test_size"],
                random_state=SETTINGS["random_state"],
                stratify=stratify_arg,
            )
        except ValueError as e:
            print(
                f"[main] Stratified split failed ({e}). "
                "Falling back to random split."
            )
            # TODO: replace with logging later
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=SETTINGS["test_size"],
                random_state=SETTINGS["random_state"],
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
        )

    print(f"[main]Train size:{len(X_train)} rows|Test size:{len(X_test)} rows")
    # TODO: replace with logging later

    # -------------------------------------------------------------------
    # STEP 6: Fail-fast feature checks
    # -------------------------------------------------------------------
    print("[main] Step 6 — Verifying feature columns exist in training data")
    # TODO: replace with logging later
    all_feature_cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
        + SETTINGS["features"]["ordinal_encode"]
        + SETTINGS["features"]["min_max_scaler"]
    )
    missing_cols = [c for c in all_feature_cols if c not in X_train.columns]
    if missing_cols:
        raise ValueError(
                    f"[main] The following configured feature columns are "
                    f"missing from the data: {missing_cols}\n"
                    "Update the 'features' section in SETTINGS to match "
                    "your actual column names."
                )

    # Check that quantile_bin columns are numeric
    for col in SETTINGS["features"]["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise TypeError(
                    f"[main] Column '{col}' is listed under "
                    "'quantile_bin' but is not numeric. "
                    "Only numeric columns can be binned."
                )

    # -------------------------------------------------------------------
    # STEP 7: Build feature preprocessor (ColumnTransformer recipe only —
    #          no fitting here, fitting happens inside train_model)
    # -------------------------------------------------------------------
    print("[main] Step 7 — Building feature preprocessor recipe...")
    # TODO: replace with logging later
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        min_max_cols=SETTINGS["features"]["min_max_scaler"],
        ordinal_encode_cols=SETTINGS["features"]["ordinal_encode"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],

        n_bins=SETTINGS["features"]["n_bins"],
    )

    # -------------------------------------------------------------------
    # STEP 8: Train model (Pipeline: preprocess + estimator, fit on train)
    # -------------------------------------------------------------------
    print("[main] Step 8 — Training model...")
    # TODO: replace with logging later
    kmeans_model = train_kmeans_model(X_train=X_train,
                                      preprocessor=preprocessor,
                                      n_clusters=SETTINGS["k_means_bins"])

    # dtrees train model
    dtrees_model = train_dtrees_model(X_train=X_train, y_train=y_train,
                                      preprocessor=preprocessor,
                                      problem_type=SETTINGS["problem_type"],)

    # logit train model
    logit_model = train_logit_model(X_train, y_train, preprocessor,
                                    SETTINGS["problem_type"],
                                    SETTINGS["random_state"],
                                    SETTINGS["max_iterations"])

    # -------------------------------------------------------------------
    # STEP 9: Save trained model artifact
    # -------------------------------------------------------------------
    print("[main] Step 9 — Saving model artifact...")
    # TODO: replace with logging later
    save_model(kmeans_model, SETTINGS["kmeans_model_path"])
    save_model(dtrees_model, SETTINGS["dtrees_model_path"])
    save_model(logit_model, SETTINGS["logit_model_path"])

    # -------------------------------------------------------------------
    # STEP 10: Evaluate on held-out test set
    # -------------------------------------------------------------------
    print("[main] Step 10 — Evaluating model on test set...")
    # TODO: replace with logging later

    # Evaluate kmeans model
    kmeans_metric_value = evaluate_kmeans_model(model=kmeans_model,
                                                X_test=X_test)
    metric_label = ("RMSE" if SETTINGS["problem_type"] == "regression"
                    else "F1 (weighted)")
    print(f"[main] Kmeans Metrics {metric_label}: {kmeans_metric_value:.4f}")
    # TODO: replace with logging later

    # Evaluate dtrees model
    dtrees_metrics = evaluate_dtrees_model(model=dtrees_model, X_test=X_test,
                                           y_test=y_test,
                                           prob_type=SETTINGS['problem_type'])
    print(f"[main] Decision Tree Metrics {metric_label}: {dtrees_metrics}")
    # TODO: replace with logging later

    # Evaluate logit model
    logit_metrics = evaluate_logit_model(model=logit_model, X_test=X_test,
                                         y_test=y_test,
                                         prob_type=SETTINGS['problem_type'])
    print(f"[main] Logistic Regression Metrics{metric_label}: {logit_metrics}")

    # -------------------------------------------------------------------
    # STEP 11: Run inference on example data (first 5 rows of test set)
    # -------------------------------------------------------------------
    print("[main] Step 11 — Running inference on example rows...")
    # TODO: replace with logging later
    # Kmeans inference example
    X_example = X_test.head(5)
    kmeans_predictions_df = run_kmeans_inference(model=kmeans_model,
                                                 X_infer=X_example)
    # Dtrees inference example
    dtrees_predictions_df = run_dtrees_inference(model=dtrees_model,
                                                 X_infer=X_example)
    # Logit inference example
    logit_predictions_df = run_logit_inference(model=logit_model,
                                               X_infer=X_example)

    # -------------------------------------------------------------------
    # STEP 12: Save predictions artifact
    # -------------------------------------------------------------------
    print("[main] Step 12 — Saving predictions CSV...")
    # TODO: replace with logging later
    save_csv(kmeans_predictions_df, SETTINGS["kmeans_predictions_path"])
    save_csv(dtrees_predictions_df, SETTINGS["dtrees_predictions_path"])
    save_csv(logit_predictions_df, SETTINGS["logit_predictions_path"])

    print("\n[main] Pipeline complete!")  # TODO: replace with logging later
    print(f"  Processed data : {SETTINGS['processed_data_path']}")
    print(f"  Model artifact : {SETTINGS['logit_model_path']}")
    print(f"  Predictions    : {SETTINGS['kmeans_predictions_path']}")


if __name__ == "__main__":
    main()
