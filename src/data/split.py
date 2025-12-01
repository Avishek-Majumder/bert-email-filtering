"""
Train/test splitting utilities for the SMS Spam dataset.

This module provides a simple interface to split a pre-loaded DataFrame
into training and test sets, using the configuration defined in
config/data.yaml ("split" section).

We rely on scikit-learn's train_test_split and support:
- stratified splitting based on the label_id column
- configurable test_size and random_state
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.datasets import load_data_config, DEFAULT_DATA_CONFIG_PATH


def get_split_config(
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Retrieve the 'split' section from the data configuration.

    Parameters
    ----------
    config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    Dict[str, Any]
        Split configuration dictionary.
    """
    cfg = load_data_config(config_path)
    return cfg["split"]


def train_test_split_df(
    df: pd.DataFrame,
    label_column: str = "label_id",
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets according to config/data.yaml.

    This function expects a column with numeric labels (e.g., "label_id")
    to be present for stratification when enabled.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the label_column.
    label_column : str
        Name of the label column to use for stratification.
    config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)

    Raises
    ------
    KeyError
        If the label_column is missing.
    ValueError
        If stratified splitting is requested but the label distribution
        is incompatible (e.g., only one class present).
    """
    if label_column not in df.columns:
        raise KeyError(
            f"Label column '{label_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    split_cfg = get_split_config(config_path)
    test_size = float(split_cfg.get("test_size", 0.3))
    stratify_enabled = bool(split_cfg.get("stratify", True))
    random_state = int(split_cfg.get("random_state", 42))

    stratify_labels = df[label_column] if stratify_enabled else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
        shuffle=True,
    )

    # Reset indices for neatness
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df
