"""
Module: main.py
---------------
Role: Orchestrates the entire ML pipeline.

Pipeline steps:
1. Load raw data
2. Clean dataset
3. Validate schema
4. Feature engineering
5. Train/Test split
6. Model training
7. Model evaluation
8. Run inference and save predictions
"""

import os
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_data
from src.validate import validate_dataframe
from src.features import engineer_features, get_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference


# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Main Pipeline
# -----------------------------------------------------

def main():

    logger.info("Starting ML pipeline...")

    # Paths
    raw_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    clean_path = Path("data/processed/clean.csv")
    model_path = Path("models/model.joblib")
    preds_path = Path("reports/predictions.csv")

    # Ensure folders exist
    for p in [clean_path, model_path, preds_path]:
        os.makedirs(p.parent, exist_ok=True)

    # -------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------

    logger.info("STEP 1: Loading raw data")
    df_raw = load_raw_data(raw_path)

    # -------------------------------------------------
    # STEP 2: Clean Data
    # -------------------------------------------------

    logger.info("STEP 2: Cleaning data")
    df_clean = clean_data(df_raw, drop_na=True)

    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Clean dataset saved to {clean_path}")

    # -------------------------------------------------
    # STEP 3: Validate Data
    # -------------------------------------------------

    logger.info("STEP 3: Validating dataset")

    required_columns = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    validate_dataframe(df_clean, required_columns)

    # -------------------------------------------------
    # STEP 4: Feature Engineering
    # -------------------------------------------------

    logger.info("STEP 4: Feature engineering")

    df_eng = engineer_features(df_clean)

    X = df_eng.drop(columns=["Churn"])
    y = df_eng["Churn"].map({"No": 0, "Yes": 1})

    # -------------------------------------------------
    # STEP 5: Train/Test Split
    # -------------------------------------------------

    logger.info("STEP 5: Train/Test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    # -------------------------------------------------
    # STEP 6: Preprocessor
    # -------------------------------------------------

    logger.info("STEP 6: Creating preprocessing pipeline")

    preprocessor = get_preprocessor(X_train)

    # -------------------------------------------------
    # STEP 7: Train Model
    # -------------------------------------------------

    logger.info("STEP 7: Training model")

    pipeline = train_model(
        X_train,
        y_train,
        preprocessor,
        model_path=str(model_path)
    )

    # -------------------------------------------------
    # STEP 8: Evaluate Model
    # -------------------------------------------------

    logger.info("STEP 8: Evaluating model")

    metric_value = evaluate_model(
        pipeline,
        X_test,
        y_test,
        problem_type="classification"
    )

    logger.info(f"Evaluation metric value: {metric_value:.4f}")

    # -------------------------------------------------
    # STEP 9: Inference
    # -------------------------------------------------

    logger.info("STEP 9: Running inference")

    predictions_df = run_inference(pipeline, X_test)

    predictions_df.to_csv(preds_path, index=False)

    logger.info(f"Predictions saved to {preds_path}")

    logger.info("Pipeline completed successfully!")


# -----------------------------------------------------
# Script Entry Point
# -----------------------------------------------------

if __name__ == "__main__":
    main()