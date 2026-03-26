"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Validate -> Clean -> Train -> Evaluate -> Infer).
Usage: python src/main.py
"""

import yaml
from pathlib import Path
from src.load_data import load_data
from src.validate import validate_dataframe
from src.clean_data import clean_data
from src.feature_engineering import build_features, FeatureConfig
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from sklearn.preprocessing import FunctionTransformer

CONFIG_PATH = Path("config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

REQUIRED_COLUMNS = config["validation"]["required_columns"]
TARGET_COL = config["target"]["column"]
NUMERIC_COLS = config["features"]["numeric"]
CATEGORICAL_COLS = config["features"]["categorical"]
PROBLEM_TYPE = config["train"]["problem_type"]
MODEL_PATH = config["train"]["model_path"]


def run_pipeline(config: dict) -> float:
    """Run the full ML pipeline. Returns the evaluation metric."""
    # Step 1: Load data
    df = load_data(Path(config["data"]["raw"]))

    # Step 2: Validate (before cleaning — columns still have original casing)
    validate_dataframe(df, config["validation"]["required_columns"])

    # Step 3: Clean
    df = clean_data(df)

    # Step 4: Feature engineering (columns are now lowercase)
    cfg = FeatureConfig(
        target_col=config["target"]["column"],
        numeric_cols=tuple(config["features"]["numeric"]),
        categorical_cols=tuple(config["features"]["categorical"]),
    )
    df = build_features(df, cfg)

    # Step 5: Split features and target
    target_col = config["target"]["column"]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Step 6: Train (FunctionTransformer passes features through — already engineered above)
    preprocessor = FunctionTransformer()
    fitted_pipeline, X_val, y_val, X_test, y_test = train_model(
        X, y, preprocessor,
        config["train"]["problem_type"],
        config["train"]["model_path"],
        test_size=config["train"]["test_size"],
        random_state=config["train"]["seed"],
    )

    # Step 7: Evaluate
    metric = evaluate_model(fitted_pipeline, X_test, y_test, config["train"]["problem_type"])
    print(f"Pipeline complete. {config['train']['problem_type']} metric: {metric:.4f}")

    # Step 8: Infer
    predictions = run_inference(fitted_pipeline, X_test)
    print(predictions.head())

    return metric


if __name__ == "__main__":
    run_pipeline(config)
