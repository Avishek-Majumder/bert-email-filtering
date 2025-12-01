"""
Training and evaluation pipeline for classical ML models.

This module reproduces the classical baseline experiments from
"Harnessing BERT for Advanced Email Filtering in Cybersecurity" by:

- loading the SMS Spam Collection dataset
- performing a stratified 70/30 train–test split
- extracting TF–IDF features from preprocessed text
- training multiple ML models:
    * Random Forest
    * Logistic Regression
    * SVM
    * XGBoost (optional, if xgboost is installed)
    * Gradient Boosting
    * Naive Bayes
    * KNN
- evaluating each model via accuracy, precision, recall, F1-score
- saving metrics to CSV/JSON under experiments/results/

This module is designed to be callable both as a library function and
as a standalone script (via `python -m src.training.train_ml`).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.datasets import load_sms_dataset, DEFAULT_DATA_CONFIG_PATH
from src.data.split import train_test_split_df
from src.features.tfidf_vectorizer import (
    fit_tfidf_from_series,
    transform_texts_to_tfidf,
)
from src.models.ml_models import build_all_ml_models, load_ml_config
from src.utils.training_utils import (
    load_train_config,
    ensure_dir_exists,
    seed_everything,
    get_logger,
)
from src.evaluation.metrics import compute_classification_metrics


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _prepare_tfidf_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a TF–IDF vectorizer on training texts and transform both train
    and test texts.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set with columns ["text", "label_id", ...].
    test_df : pd.DataFrame
        Test set with columns ["text", "label_id", ...].
    data_config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, X_test, y_train, y_test)
    """
    train_texts = train_df["text"]
    test_texts = test_df["text"]

    # Fit TF–IDF on training texts and persist the vectorizer.
    _ = fit_tfidf_from_series(
        train_texts=train_texts,
        data_config_path=data_config_path,
        artifacts_dir=None,  # resolved via train.yaml
        save=True,
    )

    # Reload the fitted vectorizer to ensure consistency.
    from src.features.tfidf_vectorizer import load_tfidf_vectorizer

    vectorizer = load_tfidf_vectorizer()

    X_train = transform_texts_to_tfidf(
        texts=train_texts.values,
        vectorizer=vectorizer,
        data_config_path=data_config_path,
    )
    X_test = transform_texts_to_tfidf(
        texts=test_texts.values,
        vectorizer=vectorizer,
        data_config_path=data_config_path,
    )

    y_train = train_df["label_id"].values
    y_test = test_df["label_id"].values

    return X_train, X_test, y_train, y_test


def _maybe_wrap_with_scaler(
    model_name: str,
    model: object,
    ml_cfg: Dict[str, Any],
) -> object:
    """
    Optionally wrap a model with a StandardScaler for TF–IDF features.

    For some linear models (Logistic Regression, SVM, KNN), it is often
    beneficial to standardize features. We enable this behavior based on
    the "use_feature_scaling" flag in config/ml.yaml.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g., "logistic_regression".
    model : object
        Underlying sklearn estimator.
    ml_cfg : Dict[str, Any]
        Full ML configuration dictionary.

    Returns
    -------
    object
        Either the original model, or a sklearn Pipeline with a scaler
        followed by the model.
    """
    general_cfg = ml_cfg.get("general", {}) or {}
    use_scaling = bool(general_cfg.get("use_feature_scaling", True))

    if not use_scaling:
        return model

    # Only scale models that benefit from it and support sparse input
    # with StandardScaler(with_mean=False).
    models_needing_scaling = {"logistic_regression", "svm", "knn"}
    if model_name not in models_needing_scaling:
        return model

    scaler = StandardScaler(with_mean=False)
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("clf", model),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate_ml_models(
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    ml_config_path: str = "config/ml.yaml",
    train_config_path: str = "config/train.yaml",
) -> pd.DataFrame:
    """
    End-to-end pipeline to train and evaluate all classical ML models.

    Parameters
    ----------
    data_config_path : str
        Path to config/data.yaml.
    ml_config_path : str
        Path to config/ml.yaml.
    train_config_path : str
        Path to config/train.yaml.

    Returns
    -------
    pd.DataFrame
        DataFrame of metrics for each model, with one row per model and
        columns: ["model", "accuracy", "precision", "recall", "f1"].
    """
    # Load configs
    data_df, label_mapping = load_sms_dataset(config_path=data_config_path)
    train_df, test_df = train_test_split_df(
        data_df,
        label_column="label_id",
        config_path=data_config_path,
    )

    train_cfg = load_train_config(train_config_path)
    ml_cfg = load_ml_config(ml_config_path)

    seed = int(train_cfg["general"].get("random_state", 42))
    deterministic = bool(train_cfg["general"].get("deterministic", True))
    cudnn_benchmark = bool(train_cfg["general"].get("cudnn_benchmark", False))
    seed_everything(seed=seed, deterministic=deterministic, cudnn_benchmark=cudnn_benchmark)

    logger = get_logger(
        name="train_ml",
        config=train_cfg,
        log_file_suffix="ml",
    )

    logger.info("Loaded dataset with %d samples.", len(data_df))
    logger.info("Label mapping: %s", label_mapping)
    logger.info("Train size: %d, Test size: %d", len(train_df), len(test_df))

    # Prepare TF–IDF features
    logger.info("Fitting TF–IDF vectorizer and transforming texts...")
    X_train, X_test, y_train, y_test = _prepare_tfidf_features(
        train_df=train_df,
        test_df=test_df,
        data_config_path=data_config_path,
    )

    logger.info("TF–IDF feature shapes: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

    # Build models
    models = build_all_ml_models(config_path=ml_config_path)
    logger.info("Built %d ML models: %s", len(models), list(models.keys()))

    # Prepare output directories
    results_dir = train_cfg["paths"]["results_dir"]
    models_dir = train_cfg["paths"]["models_dir"]
    ensure_dir_exists(results_dir)
    ensure_dir_exists(models_dir)

    # Train, evaluate, and collect metrics
    metrics_records = []

    for model_name, base_model in models.items():
        logger.info("=" * 80)
        logger.info("Training model: %s", model_name)

        model = _maybe_wrap_with_scaler(model_name, base_model, ml_cfg)

        # Fit model
        model.fit(X_train, y_train)
        logger.info("Model '%s' trained.", model_name)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            average="binary",
        )
        logger.info(
            "Metrics for %s - acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        # Save model-specific metrics to JSON
        metrics_with_name = {
            "model": model_name,
            **metrics,
        }
        metrics_records.append(metrics_with_name)

        metrics_json_path = os.path.join(
            results_dir,
            f"metrics_{model_name}.json",
        )
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_with_name, f, indent=2)

        logger.info("Saved metrics JSON for %s to %s", model_name, metrics_json_path)

        # Optionally save the trained model
        save_models = bool(train_cfg["save"].get("save_models", True))
        if save_models:
            # Avoid attempting to pickle large pipelines with non-serializable attributes?
            # sklearn Pipelines and most estimators are pickleable by default.
            import joblib

            model_path = os.path.join(models_dir, f"model_{model_name}.joblib")
            overwrite = bool(train_cfg["save"].get("overwrite_existing", False))
            if not os.path.exists(model_path) or overwrite:
                joblib.dump(model, model_path)
                logger.info("Saved trained model '%s' to %s", model_name, model_path)
            else:
                logger.info(
                    "Model file already exists and overwrite_existing is False: %s",
                    model_path,
                )

    # Aggregate metrics into a DataFrame and save CSV
    metrics_df = pd.DataFrame(metrics_records)
    csv_path = os.path.join(results_dir, "ml_results.csv")
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Saved aggregated ML metrics to %s", csv_path)

    return metrics_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point when running this module as a script.
    """
    _ = train_and_evaluate_ml_models()


if __name__ == "__main__":
    main()
