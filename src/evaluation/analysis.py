"""
Result aggregation and analysis utilities.

This module provides helpers to:
- load metrics produced by our ML, DL, and BERT training pipelines
- aggregate them into a single comparison table
- save the combined results as a CSV for reporting

This is useful for reproducing the comparative tables in:
"Harnessing BERT for Advanced Email Filtering in Cybersecurity".
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional

import pandas as pd

from src.utils.training_utils import load_train_config, ensure_dir_exists


def _safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file if it exists, otherwise return None.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_load_json_list(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file that contains a list of dictionaries.
    If the file does not exist, return an empty list.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # If it's a single dict, wrap it
    if isinstance(data, dict):
        return [data]
    return []


def _safe_load_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file if it exists, otherwise return None.
    """
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_ml_results(results_dir: str) -> pd.DataFrame:
    """
    Load ML metrics from ml_results.csv (if present).

    Parameters
    ----------
    results_dir : str
        Directory where ML results are stored.

    Returns
    -------
    pd.DataFrame
        DataFrame of ML results; may be empty if no file is found.
    """
    ml_csv = os.path.join(results_dir, "ml_results.csv")
    df = _safe_load_csv(ml_csv)
    if df is None:
        return pd.DataFrame(columns=["model", "accuracy", "precision", "recall", "f1"])
    return df


def load_dl_results(results_dir: str) -> pd.DataFrame:
    """
    Load DL metrics from dl_results.json (if present).

    Parameters
    ----------
    results_dir : str
        Directory where DL results are stored.

    Returns
    -------
    pd.DataFrame
        DataFrame of DL results; may be empty if no file is found.
    """
    dl_json = os.path.join(results_dir, "dl_results.json")
    records = _safe_load_json_list(dl_json)
    if not records:
        return pd.DataFrame(columns=["model", "accuracy", "precision", "recall", "f1"])
    return pd.DataFrame(records)


def load_bert_results(results_dir: str) -> pd.DataFrame:
    """
    Load BERT metrics from bert_metrics.json (if present).

    Parameters
    ----------
    results_dir : str
        Directory where BERT results are stored.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single BERT row; may be empty if no file is found.
    """
    bert_json = os.path.join(results_dir, "bert_metrics.json")
    data = _safe_load_json(bert_json)
    if data is None:
        return pd.DataFrame(columns=["model", "accuracy", "precision", "recall", "f1"])
    return pd.DataFrame([data])


def aggregate_all_results(
    train_config_path: str = "config/train.yaml",
    save: bool = True,
    filename: str = "all_results.csv",
) -> pd.DataFrame:
    """
    Aggregate ML, DL, and BERT results into a single comparison table.

    Parameters
    ----------
    train_config_path : str
        Path to config/train.yaml (used to locate results_dir).
    save : bool
        Whether to save the aggregated results as a CSV file.
    filename : str
        Name of the aggregated results CSV file.

    Returns
    -------
    pd.DataFrame
        Combined results with one row per model and columns:
        ["model", "category", "accuracy", "precision", "recall", "f1", ...].
    """
    train_cfg = load_train_config(train_config_path)
    results_dir = train_cfg["paths"]["results_dir"]
    ensure_dir_exists(results_dir)

    # Load each family of results
    ml_df = load_ml_results(results_dir)
    if not ml_df.empty:
        ml_df["category"] = "ml"

    dl_df = load_dl_results(results_dir)
    if not dl_df.empty:
        dl_df["category"] = "dl"

    bert_df = load_bert_results(results_dir)
    if not bert_df.empty:
        bert_df["category"] = "bert"

    # Normalise core metric columns
    def _ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
        core_cols = ["model", "accuracy", "precision", "recall", "f1"]
        for c in core_cols:
            if c not in df.columns:
                df[c] = None
        return df[["model", "category", "accuracy", "precision", "recall", "f1"] + [c for c in df.columns if c not in core_cols + ["category"]]]

    dfs = []
    if not ml_df.empty:
        dfs.append(_ensure_core_columns(ml_df))
    if not dl_df.empty:
        dfs.append(_ensure_core_columns(dl_df))
    if not bert_df.empty:
        dfs.append(_ensure_core_columns(bert_df))

    if dfs:
        combined = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=["model", "category", "accuracy", "precision", "recall", "f1"]
        )

    # Sort by F1 (descending) if available
    if "f1" in combined.columns:
        combined = combined.sort_values(by="f1", ascending=False, na_position="last")

    # Save combined CSV
    if save:
        out_path = os.path.join(results_dir, filename)
        combined.to_csv(out_path, index=False)

    return combined


if __name__ == "__main__":
    # Small CLI helper: `python -m src.evaluation.analysis`
    df = aggregate_all_results()
    # Print a concise view to stdout
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df[["model", "category", "accuracy", "precision", "recall", "f1"]])
