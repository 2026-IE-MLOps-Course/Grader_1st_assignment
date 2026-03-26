"""
Educational Goal:
- Why this module exists in an MLOps system: One readable entrypoint makes runs repeatable (local, CI, schedulers).
- Responsibility (separation of concerns): Orchestrate steps without hiding logic in abstractions.
- Pipeline contract (inputs and outputs): raw -> clean.csv, model.joblib, predictions.csv

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_data
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_data
from src.train import train_model
from src.utils import save_csv, save_model
# validate_dataframe removed since validate module currently has no implementation



SETTINGS = {
    "is_example_config": True,
    "problem_type": "classification",  # iris is classification: species
    "random_seed": 42,
    "test_size": 0.2,
    "target_column": "species",
    "data": {
        # load_data accepts either a csv path or a seaborn dataset name
        # using the iris dataset from seaborn for tutorial reproducibility
        "source": "seaborn",
        "dataset_name": "iris",
        # path where cleaned data should be stored (required by clean_data)
        "processed": "data/processed/clean.csv",
    },
    "cleaning": {
        # optional cleaning settings may be added here later
    },
    "paths": {
        "raw_data": "data/raw/iris.csv",
        "clean_data": "data/processed/clean.csv",
        "model": "models/model.joblib",
        "predictions": "reports/predictions.csv",
    },
    "features": {
        # Iris has 4 numeric columns; we’ll demonstrate quantile binning on all 4.
        "quantile_bin": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "categorical_onehot": [],
        "numeric_passthrough": [],
        "n_bins": 3,
    },
}


def main():
    """
    Inputs:
    - None (uses SETTINGS and filesystem)
    Outputs:
    - None (writes artifacts + prints metric)
    Why this contract matters for reliable ML delivery:
    - One entrypoint enables repeatable execution in CI and future orchestration tools.
    """
    print("[main.main] Starting iris pipeline")  # TODO: replace with logging later

    # 1) Ensure directories
    print("[main.main] Ensuring directories exist")  # TODO: replace with logging later
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # 2) Loud config reminder
    if SETTINGS.get("is_example_config", False):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("LOUD WARNING: Example SETTINGS are active (iris tutorial).")
        print("You must update SETTINGS for your real dataset later.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Load SETTINGS from config.yaml in production:
    # import yaml
    # with open('config.yaml') as f:
    #     SETTINGS = yaml.safe_load(f)['pipeline']
    # For now, hardcoded SETTINGS are used for the tutorial.
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # 3) Load data using config-driven loader
    df_raw = load_data(SETTINGS)

    # 4) Clean using config (clean_data expects the same SETTINGS dict)
    target_col = SETTINGS["target_column"]
    df_clean = clean_data(df_raw, SETTINGS)

    # 5) Save processed
    save_csv(df_clean, Path(SETTINGS["paths"]["clean_data"]))

    # 6) Basic validation (module stubbed out)
    # simple check because src.validate currently has no implementation
    if df_clean is None or len(df_clean) == 0:
        raise ValueError("Validation failed: cleaned DataFrame is empty.")
    feat = SETTINGS["features"]

    # 7) Split BEFORE fitting features (leakage prevention)
    print("[main.main] Train/test split (before fitting preprocessors)")  # TODO: replace with logging later
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    stratify = y if SETTINGS["problem_type"] == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_seed"],
            stratify=stratify,
        )
    except ValueError as e:
        print(f"[main.main] Stratify failed ({e}) -> fallback without stratify")  # TODO: replace with logging later
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_seed"],
            stratify=None,
        )

    # 8) Fail-fast feature checks
    print("[main.main] Fail-fast feature checks")  # TODO: replace with logging later
    configured = feat["quantile_bin"] + feat["categorical_onehot"] + feat["numeric_passthrough"]
    missing = [c for c in configured if c not in X_train.columns]
    if missing:
        raise ValueError(f"Configured feature columns missing in X_train: {missing}")

    for c in feat["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            raise ValueError(f"Column '{c}' is in quantile_bin but is not numeric.")

    # 9) Build feature recipe (unfitted)
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=feat["quantile_bin"],
        categorical_onehot_cols=feat["categorical_onehot"],
        numeric_passthrough_cols=feat["numeric_passthrough"],
        n_bins=feat["n_bins"],
    )

    # 10) Train
    model = train_model(X_train=X_train, y_train=y_train, preprocessor=preprocessor, problem_type=SETTINGS["problem_type"])

    # 11) Save model
    save_model(model, Path(SETTINGS["paths"]["model"]))

    # 12) Evaluate
    score = evaluate_model(model, X_test=X_test, y_test=y_test, problem_type=SETTINGS["problem_type"])
    print(f"[main.main] Test weighted F1: {score:.4f}")  # TODO: replace with logging later

    # 13) Inference
    preds = run_inference(model, X_test)  # run_inference signature expects (model, X)
    # 14) Save predictions
    save_csv(preds, Path(SETTINGS["paths"]["predictions"]))

    print("[main.main] Done. Wrote clean.csv, model.joblib, predictions.csv")  # TODO: replace with logging later


if __name__ == "__main__":
    main()