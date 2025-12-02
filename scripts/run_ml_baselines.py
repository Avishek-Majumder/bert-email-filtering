"""
Run classical ML baselines for spam detection.

This script is a convenience wrapper around
`src.training.train_ml.train_and_evaluate_ml_models`, which:

- loads the configured dataset
- fits TF–IDF features
- trains all ML models specified in config/ml.yaml
- evaluates them on the test set
- writes metrics under experiments/results/
- saves trained models under experiments/models/

Usage (from project root):

    python -m scripts.run_ml_baselines
    # or
    python scripts/run_ml_baselines.py
"""

from __future__ import annotations

import argparse

from src.training.train_ml import train_and_evaluate_ml_models
from src.utils.training_utils import load_train_config, get_logger


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Currently minimal, but can be extended later for
    per-model selection, custom config paths, etc.
    """
    parser = argparse.ArgumentParser(
        description="Run classical ML baselines for spam detection."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="config/data.yaml",
        help="Path to data config YAML (default: config/data.yaml).",
    )
    parser.add_argument(
        "--ml-config",
        type=str,
        default="config/ml.yaml",
        help="Path to ML config YAML (default: config/ml.yaml).",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="config/train.yaml",
        help="Path to global train config YAML (default: config/train.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load training config for logging / paths
    train_cfg = load_train_config(args.train_config)
    logger = get_logger(
        name="run_ml_baselines",
        config=train_cfg,
        log_file_suffix="ml_baselines",
    )

    logger.info("=" * 80)
    logger.info("Starting classical ML baselines.")
    logger.info(
        "Configs: data=%s, ml=%s, train=%s",
        args.data_config,
        args.ml_config,
        args.train_config,
    )

    metrics_df = train_and_evaluate_ml_models(
        data_config_path=args.data_config,
        ml_config_path=args.ml_config,
        train_config_path=args.train_config,
    )

    if not metrics_df.empty:
        logger.info("Completed ML baselines. Metrics:")
        logger.info("\n%s", metrics_df.sort_values("f1", ascending=False))
    else:
        logger.warning("ML baselines finished, but metrics DataFrame is empty.")

    logger.info("Classical ML baselines run completed.")


if __name__ == "__main__":
    main()
