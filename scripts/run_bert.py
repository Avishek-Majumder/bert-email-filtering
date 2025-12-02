"""
Run BERT fine-tuning for spam detection.

This script is a convenience wrapper around
`src.training.train_bert.train_and_evaluate_bert`, which:

- loads the configured dataset
- builds BERT-ready datasets (input_ids, attention_mask, labels)
- fine-tunes a pretrained BERT model for binary classification
- evaluates it on the test set
- writes metrics under experiments/results/
- saves the model under experiments/models/ and experiments/bert/

Usage (from project root):

    python -m scripts.run_bert
    # or
    python scripts/run_bert.py
"""

from __future__ import annotations

import argparse

from src.training.train_bert import train_and_evaluate_bert
from src.utils.training_utils import load_train_config, get_logger


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Currently minimal, but can be extended later for
    custom BERT configs, epochs, etc.
    """
    parser = argparse.ArgumentParser(
        description="Run BERT fine-tuning for spam detection."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="config/data.yaml",
        help="Path to data config YAML (default: config/data.yaml).",
    )
    parser.add_argument(
        "--bert-config",
        type=str,
        default="config/bert.yaml",
        help="Path to BERT config YAML (default: config/bert.yaml).",
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
        name="run_bert",
        config=train_cfg,
        log_file_suffix="bert",
    )

    logger.info("=" * 80)
    logger.info("Starting BERT fine-tuning.")
    logger.info(
        "Configs: data=%s, bert=%s, train=%s",
        args.data_config,
        args.bert_config,
        args.train_config,
    )

    metrics = train_and_evaluate_bert(
        data_config_path=args.data_config,
        bert_config_path=args.bert_config,
        train_config_path=args.train_config,
    )

    logger.info("BERT fine-tuning completed. Final test metrics:")
    logger.info(
        "acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f",
        metrics.get("accuracy", float("nan")),
        metrics.get("precision", float("nan")),
        metrics.get("recall", float("nan")),
        metrics.get("f1", float("nan")),
    )
    logger.info("Full metrics dict: %s", metrics)


if __name__ == "__main__":
    main()
