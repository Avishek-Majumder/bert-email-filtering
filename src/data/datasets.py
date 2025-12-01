"""
Dataset loading utilities for the SMS Spam Collection dataset.

This module is responsible for:
- reading the dataset configuration from config/data.yaml
- loading the raw CSV file into a pandas DataFrame
- normalizing text and label columns to standard names ("text", "label")
- applying basic cleaning (drop NA, drop duplicates) as configured
- mapping string labels (e.g., "ham", "spam") to numeric IDs (0, 1)

The resulting DataFrame is ready to be used by:
- feature extraction (TF–IDF) for classical ML models
- sequence building for CNN/LSTM/BiLSTM/RNN models
- BERT tokenization (with minimal additional preprocessing)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, Any

import pandas as pd
import yaml


DEFAULT_DATA_CONFIG_PATH = "config/data.yaml"


def _load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty or cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty or invalid: {path}")

    return cfg


def load_data_config(config_path: str = DEFAULT_DATA_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and return the full data configuration dictionary.

    Parameters
    ----------
    config_path : str, optional
        Path to the data YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the "dataset", "split", and "preprocessing" sections.
    """
    cfg = _load_yaml(config_path)

    # Provide helpful errors if sections are missing.
    for section in ("dataset", "split", "preprocessing"):
        if section not in cfg:
            raise KeyError(f'Missing "{section}" section in data config: {config_path}')

    return cfg


def get_label_mapping(
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Dict[str, int]:
    """
    Build and return the mapping from string labels to numeric IDs.

    The mapping is derived from the "negative_label" and "positive_label"
    fields under the "dataset" section in config/data.yaml.

    Returns
    -------
    Dict[str, int]
        Mapping from label string to integer ID, e.g. {"ham": 0, "spam": 1}.
    """
    cfg = load_data_config(config_path)
    dataset_cfg = cfg["dataset"]

    negative_label = dataset_cfg.get("negative_label", "ham")
    positive_label = dataset_cfg.get("positive_label", "spam")

    return {
        negative_label: 0,
        positive_label: 1,
    }


def load_sms_dataset(
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load the SMS Spam Collection dataset according to the configuration.

    This function:
    - reads the CSV specified in config/data.yaml
    - ensures the text and label columns exist
    - optionally drops NA text rows and duplicates
    - normalizes columns to standard names: "text", "label"
    - adds a numeric "label_id" column using the label mapping

    Parameters
    ----------
    config_path : str, optional
        Path to the data YAML configuration file.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        A tuple containing:
        - df: DataFrame with columns ["text", "label", "label_id", ...]
        - label_mapping: dict mapping original label strings to integer IDs.

    Raises
    ------
    FileNotFoundError
        If the dataset CSV file cannot be found.
    ValueError
        If required columns are missing or labels are inconsistent.
    """
    cfg = load_data_config(config_path)
    dataset_cfg = cfg["dataset"]

    csv_path = dataset_cfg.get("path", "data/raw/sms_spam_collection.csv")
    text_column = dataset_cfg.get("text_column", "message")
    label_column = dataset_cfg.get("label_column", "label")
    drop_duplicates = bool(dataset_cfg.get("drop_duplicates", True))
    drop_na_text = bool(dataset_cfg.get("drop_na_text", True))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Validate expected columns
    missing_cols = [col for col in (text_column, label_column) if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required column(s) in dataset CSV: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Optionally drop NA text rows
    if drop_na_text:
        df = df.dropna(subset=[text_column])

    # Optionally drop duplicates based on text + label
    if drop_duplicates:
        df = df.drop_duplicates(subset=[text_column, label_column], keep="first")

    # Normalize column names to a standard interface.
    if text_column != "text":
        df = df.rename(columns={text_column: "text"})
    if label_column != "label":
        df = df.rename(columns={label_column: "label"})

    # Ensure text is string type
    df["text"] = df["text"].astype(str)

    # Build label mapping and apply it.
    label_mapping = get_label_mapping(config_path)
    valid_labels = set(label_mapping.keys())

    # Filter to only rows with valid labels
    invalid_mask = ~df["label"].isin(valid_labels)
    num_invalid = invalid_mask.sum()
    if num_invalid > 0:
        # We drop invalid labels but also warn via exception message if all are invalid.
        df = df[~invalid_mask]
        if df.empty:
            raise ValueError(
                "After filtering, no rows remain with valid labels. "
                f"Expected labels: {valid_labels}"
            )

    df["label_id"] = df["label"].map(label_mapping)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df, label_mapping
