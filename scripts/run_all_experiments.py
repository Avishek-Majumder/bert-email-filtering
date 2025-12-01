"""
End-to-end runner for all spam detection experiments.

This script orchestrates the full pipeline described in
"Harnessing BERT for Advanced Email Filtering in Cybersecurity":

1) Classical ML baselines with TF–IDF features
2) Deep learning sequence models (CNN, LSTM, BiLSTM, RNN)
3) BERT fine-tuning
4) Aggregation of all metrics into a single comparison table

Usage (from the project root):

    python -m scripts.run_all_experiments

or:

    python scripts/run_all_experiments.py
"""

from __future__ import annotations

import os

from src.training.train_ml import train_and_evaluate_ml_models
from src.training.train_dl import train_and_evaluate_dl_models
from src.training.train_bert import train_and_evaluate_bert
from src.evaluation.analysis import aggregate_all_results
from src.utils.training_utils import load_train_config, get_logger, ensure_dir_exists


def main() -> None:
    # Load global training configuration for paths/logging
    train_cfg = load_train_config()
    logger = get_logger(
        name="run_all_experiments",
        config=train_cfg,
        log_file_suffix="all",
    )

    paths_cfg = train_cfg.get("paths", {}) or {}
    results_dir = paths_cfg.get("results_dir", "experiments/results")
    ensure_dir_exists(results_dir)

    logger.info("=" * 80)
    logger.info("Starting full experimental pipeline (ML + DL + BERT).")

    # ------------------------------------------------------------------
    # 1) Classical ML baselines
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Running classical ML baselines (TF–IDF + RF/LR/SVM/... ).")
    ml_df = train_and_evaluate_ml_models()
    logger.info("Finished classical ML baselines.")
    if not ml_df.empty:
        logger.info("ML models summary (top by F1):\n%s", ml_df.sort_values("f1", ascending=False).head())

    # ------------------------------------------------------------------
    # 2) Deep learning sequence models
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Running deep learning models (CNN, LSTM, BiLSTM, RNN).")
    train_and_evaluate_dl_models()
    logger.info("Finished deep learning models.")

    # ------------------------------------------------------------------
    # 3) BERT fine-tuning
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Running BERT fine-tuning experiments.")
    bert_metrics = train_and_evaluate_bert()
    logger.info("Finished BERT fine-tuning. BERT metrics: %s", bert_metrics)

    # ------------------------------------------------------------------
    # 4) Aggregate all results
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Aggregating ML, DL, and BERT results into a single table.")
    combined_df = aggregate_all_results()
    logger.info("Aggregated results shape: %s", combined_df.shape)

    # Show a concise ranking by F1-score
    if not combined_df.empty and "f1" in combined_df.columns:
        ranking = combined_df[["model", "category", "accuracy", "precision", "recall", "f1"]].head(20)
        logger.info("Top models by F1-score:\n%s", ranking.to_string(index=False))

    logger.info("Full experimental pipeline completed.")


if __name__ == "__main__":
    main()
