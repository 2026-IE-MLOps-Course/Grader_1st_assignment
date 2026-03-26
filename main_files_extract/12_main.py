"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow
(Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main
"""
import logging
from pathlib import Path

import yaml  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

import src.clean_data as clean_data
import src.evaluate as evaluate
import src.features as features
import src.infer as infer
import src.load_data as load_data
import src.train as train
import src.validate as validate
from src.utils import save_csv, save_json, save_model, setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load the external configuration
SETTINGS = load_config("config.yaml")

setup_logging(
    level=SETTINGS.get("logging", {}).get("level", "INFO"),
    log_file=SETTINGS.get("logging", {}).get("file"),
)


def main():
    try:
        logger.info(
            "--- Starting MLOps Pipeline: %s ---",
            SETTINGS["project"]["name"],
        )

        # 1. Infrastructure Setup
        for folder in ["data/raw", "data/processed", "models", "reports"]:
            Path(folder).mkdir(parents=True, exist_ok=True)

        # 2. Load
        df_raw = load_data.load_raw_data(Path(SETTINGS["paths"]["raw_data"]))

        # 3. Clean
        df_clean = clean_data.clean_dataframe(
            df_raw, SETTINGS["project"]["target_column"]
        )
        save_csv(df_clean, Path(SETTINGS["paths"]["clean_data"]))

        # 4. Validate
        required_cols = (
            SETTINGS["features"]["quantile_bin"]
            + SETTINGS["features"]["categorical_onehot"]
            + SETTINGS["features"]["numeric_passthrough"]
            + [SETTINGS["project"]["target_column"]]
        )
        validate.validate_dataframe(
            df_clean,
            required_cols,
            target_column=SETTINGS["project"]["target_column"],
            allow_feature_nulls=True,
        )

        # 5. Split (Train / Val / Test)
        target = SETTINGS["project"]["target_column"]
        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        test_size = SETTINGS["train"]["test_size"]
        val_size = SETTINGS["train"].get("val_size", 0.2)
        seed = SETTINGS["train"]["seed"]

        strat = y if SETTINGS["project"][
            "problem_type"] == "classification" else None

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=strat,
        )

        strat_trainval = (
            y_trainval
            if SETTINGS["project"]["problem_type"] == "classification"
            else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size,
            random_state=seed,
            stratify=strat_trainval,
        )

        logger.info(
            "Split sizes:"
            "train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
            len(X_train), 100 * len(X_train) / len(X),
            len(X_val), 100 * len(X_val) / len(X),
            len(X_test), 100 * len(X_test) / len(X),
        )

        # 6. Feature Engineering
        preprocessor = features.get_feature_preprocessor(
            quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
            categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
            numeric_passthrough_cols=SETTINGS["features"][
                "numeric_passthrough"],
            n_bins=SETTINGS["features"]["n_bins"],
        )

        # 7. Train
        model_pipeline = train.train_model(
            X_train,
            y_train,
            preprocessor,
            SETTINGS["project"]["problem_type"],
            train_config=SETTINGS.get("train", {}),
        )

        # 8. Evaluate (Val and Test)
        val_metrics = evaluate.evaluate_model(
            model_pipeline,
            X_val,
            y_val,
            SETTINGS["project"]["problem_type"],
        )

        final_preprocessor = features.get_feature_preprocessor(
            quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
            categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
            numeric_passthrough_cols=SETTINGS[
                "features"][
                "numeric_passthrough"],
            n_bins=SETTINGS["features"]["n_bins"],
        )

        final_model_pipeline = train.train_model(
            X_trainval,
            y_trainval,
            final_preprocessor,
            SETTINGS["project"]["problem_type"],
            train_config=SETTINGS.get("train", {}),
        )

        save_model(final_model_pipeline, Path(SETTINGS["paths"]["model_path"]))

        test_metrics = evaluate.evaluate_model(
            final_model_pipeline,
            X_test,
            y_test,
            SETTINGS["project"]["problem_type"],
        )

        metrics = {"val": val_metrics, "test": test_metrics}
        save_json(metrics, Path("reports/metrics.json"))
        save_json(SETTINGS, Path("reports/run_config.json"))

        # 9. Inference
        df_preds = infer.run_inference(final_model_pipeline, X_test)
        save_csv(df_preds, Path(SETTINGS["paths"]["report_path"]))

        logger.info("--- Pipeline Completed Successfully ---")
        return 0

    except Exception:
        logger.exception(
            "Pipeline failed due to an unhandled exception."
        )
        raise


if __name__ == "__main__":
    main()
