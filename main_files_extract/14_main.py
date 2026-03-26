"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python src/main.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from src.utils import save_csv, save_model

# ── Project-wide settings ────────────────────────────────────────────
SETTINGS = {
    "is_example_config": True,
    "target_column": "Points",
    "problem_type": "regression",

    # Paths
    "raw_data_path": "data/raw/nhl_player_stats.csv",
    "processed_data_path": "data/processed/clean.csv",
    "model_path": "models/model.joblib",
    "predictions_path": "data/inference/predictions.csv",

    # Feature groups
    "features": {
        "quantile_bin_cols": ["Icetime_Minutes", "Shot_Attempts"],
        "categorical_onehot_cols": ["Pos"],
        "numeric_passthrough_cols": [
            "Faceoff_Win_Pct",
            "Takeaways",
            "Giveaways",
            "Shooting_Pct_On_Unblocked",
            "PIM_Drawn",
            "Pct_Shift_Starts_Offensive_Zone",
            "On_Ice_Corsi_Pct",
        ],
        "n_bins": 3,
    },

    # Training
    "test_size": 0.2,
    "random_state": 42,
}


def main():
    """Run the full ML pipeline end-to-end."""

    # ── 0. Create output directories ─────────────────────────────────
    for key in ("processed_data_path", "model_path", "predictions_path"):
        Path(SETTINGS[key]).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Example-config guard ──────────────────────────────────────
    if SETTINGS.get("is_example_config"):
        print("[main] WARNING: Running with example config. "
              "Set is_example_config to False for production.")

    # ── 2. Load raw data ─────────────────────────────────────────────
    print("\n=== Step 1: Load Data ===")
    df_raw = load_raw_data(SETTINGS["raw_data_path"])
    print(f"[main] Raw data shape: {df_raw.shape}")

    # ── 3. Clean ─────────────────────────────────────────────────────
    print("\n=== Step 2: Clean Data ===")
    df_clean = clean_dataframe(df_raw, SETTINGS["target_column"])

    # ── 4. Save processed CSV ────────────────────────────────────────
    save_csv(df_clean, Path(SETTINGS["processed_data_path"]))
    print(f"[main] Saved cleaned data to {SETTINGS['processed_data_path']}")

    # ── 5. Validate ──────────────────────────────────────────────────
    print("\n=== Step 3: Validate ===")
    feature_cfg = SETTINGS["features"]
    all_feature_cols = (
        feature_cfg.get("quantile_bin_cols", [])
        + feature_cfg.get("categorical_onehot_cols", [])
        + feature_cfg.get("numeric_passthrough_cols", [])
    )
    required = [SETTINGS["target_column"]] + all_feature_cols
    validate_dataframe(df_clean, required)

    # ── 6. Train / test split ────────────────────────────────────────
    print("\n=== Step 4: Train/Test Split ===")
    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
        )
    print(f"[main] Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 7. Build feature preprocessor ────────────────────────────────
    print("\n=== Step 5: Feature Engineering ===")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=feature_cfg.get("quantile_bin_cols"),
        categorical_onehot_cols=feature_cfg.get("categorical_onehot_cols"),
        numeric_passthrough_cols=feature_cfg.get("numeric_passthrough_cols"),
        n_bins=feature_cfg.get("n_bins", 3),
    )

    # ── 8. Train pipeline ────────────────────────────────────────────
    print("\n=== Step 6: Train Model ===")
    pipeline = train_model(
        X_train, y_train, preprocessor, SETTINGS["problem_type"],
    )

    # ── 9. Save model ────────────────────────────────────────────────
    save_model(pipeline, Path(SETTINGS["model_path"]))
    print(f"[main] Model saved to {SETTINGS['model_path']}")

    # ── 10. Evaluate ─────────────────────────────────────────────────
    print("\n=== Step 7: Evaluate ===")
    score = evaluate_model(
        pipeline, X_test, y_test, SETTINGS["problem_type"],
    )
    print(f"[main] Final score: {score:.4f}")

    # ── 11. Inference ────────────────────────────────────────────────
    print("\n=== Step 8: Inference ===")
    preds_df = run_inference(pipeline, X_test)

    # ── 12. Save predictions ─────────────────────────────────────────
    save_csv(preds_df, Path(SETTINGS["predictions_path"]))
    print(f"[main] Predictions saved to {SETTINGS['predictions_path']}")

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
