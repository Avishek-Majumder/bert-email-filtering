"""
Run deep learning (DL) baselines for spam detection.

This script is a convenience wrapper around
`src.training.train_dl.train_and_evaluate_dl_models`, which:

- loads the configured dataset
- builds a vocabulary from training texts
- converts texts to padded integer sequences
- trains CNN, LSTM, BiLSTM, and RNN models
- evaluates them on the test set
- writes metrics under experiments/results/
- saves model weights under experiments/models/

Usage (from project root):

    python -m scripts.run_dl_baselines
    # or
    python scripts/run_dl_baselines.py
"""

from __future__ import annotations

import argparse

from src.training.train_dl import train_and_evaluate_dl_models
from src.utils.training_utils import load_train_config, get_logger


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Currently minimal, but can be extended later for
    selective architectures, custom config paths, etc.
    """
    parser = argparse.ArgumentParser(
        description="Run deep learning baselines (CNN, LSTM, BiLSTM, RNN) for spam detection."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="config/data.yaml",
        help="Path to data config YAML (default: config/data.yaml).",
    )
    parser.add_argument(
        "--dl-config",
        type=str,
        default="config/dl.yaml",
        help="Path to DL config YAML (default: config/dl.yaml).",
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
        name="run_dl_baselines",
        config=train_cfg,
        log_file_suffix="dl_baselines",
    )

    logger.info("=" * 80)
    logger.info("Starting deep learning baselines (CNN, LSTM, BiLSTM, RNN).")
    logger.info(
        "Configs: data=%s, dl=%s, train=%s",
        args.data_config,
        args.dl_config,
        args.train_config,
    )

    train_and_evaluate_dl_models(
        data_config_path=args.data_config,
        dl_config_path=args.dl_config,
        train_config_path=args.train_config,
    )

    logger.info("Deep learning baselines run completed.")


if __name__ == "__main__":
    main()
