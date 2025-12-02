"""
Basic tests for data loading utilities.

These tests validate that:

- the data configuration can be loaded correctly
- the SMS dataset loader works when the raw CSV is present

We intentionally skip dataset-dependent tests if the raw file is not available,
so that the test suite still runs in a fresh clone without data.
"""

from __future__ import annotations

import os

import pytest

from src.data.datasets import load_data_config, load_sms_dataset


def test_load_data_config_has_required_keys():
    """
    Ensure that config/data.yaml can be loaded and contains core sections.
    """
    cfg = load_data_config("config/data.yaml")

    # Core sections that should exist
    assert "paths" in cfg
    assert "columns" in cfg
    assert "labels" in cfg

    paths = cfg["paths"]
    columns = cfg["columns"]
    labels = cfg["labels"]

    # Paths section basics
    assert "raw_data_path" in paths

    # Columns section basics
    assert "text" in columns
    assert "label" in columns

    # Labels mapping must be non-empty
    assert isinstance(labels, dict)
    assert len(labels) >= 1


@pytest.mark.skipif(
    not os.path.exists(
        load_data_config("config/data.yaml")["paths"]["raw_data_path"]
    ),
    reason="Raw dataset file not found; skipping dataset-dependent test.",
)
def test_load_sms_dataset_returns_nonempty_df():
    """
    If the raw CSV exists, ensure load_sms_dataset returns a non-empty DataFrame
    and a non-empty label mapping.
    """
    df, label_mapping = load_sms_dataset(config_path="config/data.yaml")

    assert not df.empty, "Loaded DataFrame is empty."
    assert "text" in df.columns, "Expected 'text' column in loaded DataFrame."
    assert "label_id" in df.columns, "Expected 'label_id' column in loaded DataFrame."

    assert isinstance(label_mapping, dict)
    assert len(label_mapping) >= 1
