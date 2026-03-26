"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow
(Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python src/main.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import yaml
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model, make_plots, save_metrics, save_plots
from src.infer import run_inference


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text())


def main() -> None:
    cfg = load_config("config.yaml")

    # 1. Load raw data
    df_raw = load_raw_data(config_path="config.yaml")

    # 2. Clean
    target = cfg["target_column"]
    df_clean = clean_dataframe(df_raw, target_column=target)

    # 3. Validate
    feat_cfg = cfg["features"]
    val_cfg = cfg["validation"]
    required_columns = list(dict.fromkeys(
        [target]
        + feat_cfg["quantile_bin"]
        + feat_cfg["numeric_passthrough"]
        + feat_cfg["binary_sum_cols"]
    ))

    validate_dataframe(
        df=df_clean,
        required_columns=required_columns,
        check_missing_values=val_cfg["check_missing_values"],
        target_column=target,
        target_allowed_values=[0, 1],
        numeric_non_negative_cols=val_cfg["numeric_non_negative_cols"],
    )

    # 4. Feature preprocessor
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=feat_cfg["quantile_bin"],
        categorical_onehot_cols=feat_cfg["categorical_onehot"],
        numeric_passthrough_cols=feat_cfg["numeric_passthrough"],
        binary_sum_cols=feat_cfg["binary_sum_cols"],
        n_bins=feat_cfg["n_bins"],
    )

    # 5. Three-way split (train / val / test)
    split_cfg = cfg["split"]
    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y,
    )
    rel_val = split_cfg["val_size"] / (1.0 - split_cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=rel_val,
        random_state=split_cfg["random_state"],
        stratify=y_temp,
    )

    # 6. Train
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=cfg["problem_type"],
    )

    # 7. Evaluate on validation split
    metrics = evaluate_model(
        model=model,
        X_eval=X_val,
        y_eval=y_val,
        problem_type=cfg["problem_type"],
    )

    plots = make_plots(model, X_test, y_test)

    # 8. Save artifacts
    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]

    processed_path = Path(data_cfg["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)

    Path(paths_cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths_cfg["model_path"])

    save_metrics(metrics, paths_cfg["metrics_path"])
    save_plots(plots, paths_cfg["plots_path"])

    df_preds = run_inference(
        model=model, X_infer=X_test, include_proba=True
    )
    pred_path = Path(paths_cfg["predictions_path"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(pred_path, index=True)

    print("Done")
    print(f"Model saved:       {paths_cfg['model_path']}")
    print(f"Metrics saved:     {paths_cfg['metrics_path']}")
    print(f"Plots saved:       {paths_cfg['plots_path']}")
    print(f"Predictions saved: {paths_cfg['predictions_path']}")
    print(f"Metrics:           {metrics}")


if __name__ == "__main__":
    main()
