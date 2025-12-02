"""
High-level utilities for inspecting and comparing model performance.

This module builds on the metrics and result files produced by:

- src.training.train_ml       (classical ML baselines)
- src.training.train_dl       (CNN, LSTM, BiLSTM, RNN baselines)
- src.training.train_bert     (BERT fine-tuning)
- src.evaluation.analysis     (aggregation of ML, DL, BERT results)

It provides helpers to:

- load all results into a single DataFrame
- print model rankings by a chosen metric (e.g., F1-score)
- load confusion matrices for specific models
- pretty-print confusion matrices

Usage examples (Python):

    from src.evaluation.evaluate_models import (
        load_all_results,
        print_model_ranking,
        load_model_confusion_matrix,
        pretty_print_confusion_matrix,
    )

    df = load_all_results()
    print_model_ranking(df, metric="f1", top_k=10)

    cm = load_model_confusion_matrix("bert", category="bert")
    if cm is not None:
        pretty_print_confusion_matrix(cm, labels=["ham", "spam"])
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.training_utils import load_train_config, ensure_dir_exists
from src.evaluation.analysis import aggregate_all_results


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------


def load_all_results(
    train_config_path: str = "config/train.yaml",
    save_aggregated: bool = True,
    aggregated_filename: str = "all_results.csv",
) -> pd.DataFrame:
    """
    Load aggregated results for all models (ML, DL, BERT) as a DataFrame.

    This is a thin wrapper around `aggregate_all_results`, which:
    - reads ML, DL, and BERT metrics files from experiments/results/
    - combines them into a single table
    - optionally saves the combined CSV

    Parameters
    ----------
    train_config_path : str
        Path to config/train.yaml.
    save_aggregated : bool
        Whether to save the aggregated CSV (default: True).
    aggregated_filename : str
        Name of the CSV file for combined results.

    Returns
    -------
    pd.DataFrame
        Combined results with columns:
        ["model", "category", "accuracy", "precision", "recall", "f1", ...].
    """
    df = aggregate_all_results(
        train_config_path=train_config_path,
        save=save_aggregated,
        filename=aggregated_filename,
    )
    return df


# ---------------------------------------------------------------------------
# Rankings and summaries
# ---------------------------------------------------------------------------


def print_model_ranking(
    df: pd.DataFrame,
    metric: str = "f1",
    top_k: int = 10,
) -> None:
    """
    Print a ranking of models by a chosen metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame as returned by `load_all_results` or `aggregate_all_results`.
    metric : str
        Metric to sort by (e.g., "f1", "accuracy", "precision", "recall").
    top_k : int
        Number of top models to display.
    """
    if df.empty:
        print("[evaluate_models] No results found (empty DataFrame).")
        return

    if metric not in df.columns:
        print(f"[evaluate_models] Metric '{metric}' not found in DataFrame columns.")
        print("Available columns:", list(df.columns))
        return

    sorted_df = df.sort_values(metric, ascending=False, na_position="last")
    top_df = sorted_df.head(top_k)

    display_cols = ["model", "category", "accuracy", "precision", "recall", "f1"]
    display_cols = [c for c in display_cols if c in top_df.columns]

    print(f"\nTop {min(top_k, len(top_df))} models by '{metric}':\n")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(top_df[display_cols].to_string(index=False))
    print("")


def get_best_per_category(
    df: pd.DataFrame,
    metric: str = "f1",
) -> pd.DataFrame:
    """
    Return the best model per category (ml, dl, bert) according to a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated results DataFrame.
    metric : str
        Metric to optimize (e.g., "f1").

    Returns
    -------
    pd.DataFrame
        DataFrame containing the best row per category.
    """
    if df.empty:
        return df

    if "category" not in df.columns:
        raise ValueError("DataFrame must contain a 'category' column.")

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")

    best_rows = []
    for category, group in df.groupby("category"):
        best_row = group.sort_values(metric, ascending=False, na_position="last").head(1)
        best_rows.append(best_row)

    if best_rows:
        return pd.concat(best_rows, axis=0, ignore_index=True)
    return pd.DataFrame(columns=df.columns)


# ---------------------------------------------------------------------------
# Confusion matrix helpers
# ---------------------------------------------------------------------------


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file if it exists, otherwise return None.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_results_dir(train_config_path: str) -> str:
    """
    Get the results directory from the train config.
    """
    train_cfg = load_train_config(train_config_path)
    results_dir = train_cfg.get("paths", {}).get("results_dir", "experiments/results")
    ensure_dir_exists(results_dir)
    return results_dir


def load_model_confusion_matrix(
    model_name: str,
    category: str,
    train_config_path: str = "config/train.yaml",
) -> Optional[np.ndarray]:
    """
    Load the confusion matrix for a specific model, if available.

    Parameters
    ----------
    model_name : str
        Identifier of the model (e.g., "random_forest", "cnn", "bert").
    category : str
        One of: "ml", "dl", "bert".
    train_config_path : str
        Path to config/train.yaml (used to locate results_dir).

    Returns
    -------
    Optional[np.ndarray]
        Confusion matrix as a 2D numpy array if found, otherwise None.
    """
    category = category.lower()
    results_dir = _get_results_dir(train_config_path)

    metrics_path: Optional[str] = None

    if category == "ml":
        # Metrics file: metrics_<model_name>.json
        metrics_path = os.path.join(results_dir, f"metrics_{model_name}.json")
    elif category == "dl":
        # Metrics file: metrics_dl_<model_name>.json
        metrics_path = os.path.join(results_dir, f"metrics_dl_{model_name}.json")
    elif category == "bert":
        # Single metrics file: bert_metrics.json
        metrics_path = os.path.join(results_dir, "bert_metrics.json")
    else:
        raise ValueError(
            f"Unknown category '{category}'. Expected one of: 'ml', 'dl', 'bert'."
        )

    metrics_data = _load_json_if_exists(metrics_path)
    if metrics_data is None:
        return None

    cm = metrics_data.get("confusion_matrix", None)
    if cm is None:
        return None

    return np.asarray(cm)


def pretty_print_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str] = ("ham", "spam"),
) -> None:
    """
    Pretty-print a confusion matrix in the console.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    labels : Sequence[str]
        Class labels in the order corresponding to the confusion matrix.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be a square 2D array.")

    n = cm.shape[0]
    if len(labels) != n:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match CM size ({n})."
        )

    print("\nConfusion Matrix:")
    header = [""] + list(labels)
    row_format = "{:>10}" * (len(header))
    print(row_format.format(*header))

    for i in range(n):
        row_values = [labels[i]] + [str(int(v)) for v in cm[i]]
        print(row_format.format(*row_values))
    print("")


# ---------------------------------------------------------------------------
# Simple CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    """
    Simple CLI entry point for quick inspection from the terminal.

    Examples:

        python -m src.evaluation.evaluate_models
        python -m src.evaluation.evaluate_models --metric f1 --top-k 5
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect and compare model performance (ML, DL, BERT)."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="config/train.yaml",
        help="Path to global train config (default: config/train.yaml).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        help="Metric to rank models by (default: f1).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top models to display (default: 10).",
    )
    parser.add_argument(
        "--show-best-per-category",
        action="store_true",
        help="Also show the best model per category (ml, dl, bert).",
    )

    args = parser.parse_args()

    df = load_all_results(train_config_path=args.train_config)
    print_model_ranking(df, metric=args.metric, top_k=args.top_k)

    if args.show_best_per_category:
        best_df = get_best_per_category(df, metric=args.metric)
        if not best_df.empty:
            print("\nBest model per category:\n")
            with pd.option_context("display.max_rows", None, "display.width", 120):
                print(
                    best_df[
                        ["model", "category", "accuracy", "precision", "recall", "f1"]
                    ].to_string(index=False)
                )
            print("")
        else:
            print("[evaluate_models] No per-category results available.")


if __name__ == "__main__":
    _cli()
