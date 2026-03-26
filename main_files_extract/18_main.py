"""
Module: Main Pipeline
---------------------
Orchestrates:
Load -> Clean -> Validate -> Split -> Features -> Train -> Evaluate -> Infer

Run from repo root:
  python -m src.main
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

SETTINGS = {
    "paths": {
        "raw_data": "data/raw/insurance.csv",
        "processed_clean": "data/processed/clean.csv",
        "model": "models/linreg_insurance.joblib",
        "predictions": "reports/predictions.csv",
        "reports_dir": "reports",
    },
    # Notebook target is log(charges)
    "target_column": "log_charges",
    "test_size": 0.2,
    "random_state": 42,
    "features": {
        "numeric": ["age", "bmi", "children"],
        "categorical": ["sex", "smoker", "region"],
    },
}


def main() -> None:
    print("[main.main] Starting end-to-end insurance prediction pipeline")

    # Ensure standard dirs
    for d in ["data/raw", "data/processed", "models", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    raw_path = Path(SETTINGS["paths"]["raw_data"])
    df_raw = load_raw_data(raw_path)

    # Step 2: Clean (adds log_charges and drops charges)
    target_column = SETTINGS["target_column"]
    df_clean = clean_dataframe(df_raw, target_column=target_column)

    # Step 3: Save clean
    clean_path = Path(SETTINGS["paths"]["processed_clean"])
    save_csv(df_clean, clean_path)

    # Step 4: Validate
    required_cols = [target_column] + SETTINGS["features"]["numeric"] + SETTINGS["features"]["categorical"]
    validate_dataframe(df_clean, required_columns=required_cols, target_column=target_column)

    # Step 5: Split
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Simple split (not stratified; regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SETTINGS["test_size"],
        random_state=SETTINGS["random_state"],
    )

    # Step 6: Feature recipe
    preprocessor = get_feature_preprocessor(
        numeric_cols=SETTINGS["features"]["numeric"],
        categorical_cols=SETTINGS["features"]["categorical"],
    )

    # Step 7: Train
    from src.train import train_model

    model = train_model(X_train=X_train, y_train=y_train, preprocessor=preprocessor)

    # Step 8: Save model
    model_path = Path(SETTINGS["paths"]["model"])
    save_model(model, model_path)

    # Step 9: Evaluate + save reports
    reports_dir = Path(SETTINGS["paths"]["reports_dir"])
    _ = evaluate_model(model=model, X_test=X_test, y_test=y_test, reports_dir=reports_dir)

    # Step 10: Inference on X_test (example batch) + save predictions
    df_pred = run_inference(model=model, X_infer=X_test)
    pred_path = Path(SETTINGS["paths"]["predictions"])
    save_csv(df_pred, pred_path, index=True)  # keep index to align with test rows

    print("[main.main] Pipeline complete")
    print(f"[main.main] Saved model: {model_path}")
    print(f"[main.main] Saved predictions: {pred_path}")
    print(f"[main.main] Reports directory: {reports_dir.resolve()}")


if __name__ == "__main__":
    main()