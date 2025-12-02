"""
Smoke tests for the main training pipelines.

We verify that:

- ML training pipeline can run end-to-end when data is available.
- DL training pipeline can run end-to-end when data is available.
- BERT training pipeline can run end-to-end when explicitly enabled
  (since it can be heavy and may require a GPU).

These are *smoke tests*, not full accuracy checks: we only assert that
the code runs and returns outputs of the expected type/shape.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src.data.datasets import load_data_config
from src.training.train_ml import train_and_evaluate_ml_models
from src.training.train_dl import train_and_evaluate_dl_models
from src.training.train_bert import train_and_evaluate_bert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


DATA_CONFIG_PATH = "config/data.yaml"
TRAIN_CONFIG_PATH = "config/train.yaml"
ML_CONFIG_PATH = "config/ml.yaml"
DL_CONFIG_PATH = "config/dl.yaml"
BERT_CONFIG_PATH = "config/bert.yaml"

_DATA_CFG = load_data_config(DATA_CONFIG_PATH)
_RAW_DATA_PATH = _DATA_CFG["paths"]["raw_data_path"]


# ---------------------------------------------------------------------------
# ML pipeline smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(_RAW_DATA_PATH),
    reason="Raw dataset file not found; skipping ML smoke test.",
)
def test_train_ml_smoke():
    """
    Smoke test for classical ML baselines.

    Runs train_and_evaluate_ml_models with the configured dataset and
    asserts that a non-empty DataFrame of metrics is returned.
    """
    metrics_df = train_and_evaluate_ml_models(
        data_config_path=DATA_CONFIG_PATH,
        ml_config_path=ML_CONFIG_PATH,
        train_config_path=TRAIN_CONFIG_PATH,
    )

    assert isinstance(metrics_df, pd.DataFrame)
    assert not metrics_df.empty
    assert "model" in metrics_df.columns
    assert "f1" in metrics_df.columns


# ---------------------------------------------------------------------------
# DL pipeline smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(_RAW_DATA_PATH),
    reason="Raw dataset file not found; skipping DL smoke test.",
)
def test_train_dl_smoke():
    """
    Smoke test for deep learning baselines (CNN, LSTM, BiLSTM, RNN).

    Runs train_and_evaluate_dl_models with the configured dataset and
    asserts that it completes without raising exceptions.
    """
    # train_and_evaluate_dl_models currently returns None; we just ensure it runs.
    result = train_and_evaluate_dl_models(
        data_config_path=DATA_CONFIG_PATH,
        dl_config_path=DL_CONFIG_PATH,
        train_config_path=TRAIN_CONFIG_PATH,
    )

    # The main assertion is "no exception"; we keep an explicit assert for clarity.
    assert result is None or isinstance(result, (dict, list, tuple))


# ---------------------------------------------------------------------------
# BERT pipeline smoke test (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("RUN_BERT_TESTS", "0") != "1",
    reason=(
        "BERT smoke test is disabled by default. "
        "Set RUN_BERT_TESTS=1 in the environment to enable."
    ),
)
@pytest.mark.skipif(
    not os.path.exists(_RAW_DATA_PATH),
    reason="Raw dataset file not found; skipping BERT smoke test.",
)
def test_train_bert_smoke():
    """
    Smoke test for BERT fine-tuning.

    This test is *opt-in* because BERT training can be heavy and may
    require a GPU. To enable it, set:

        RUN_BERT_TESTS=1

    in your environment before running pytest.
    """
    metrics = train_and_evaluate_bert(
        data_config_path=DATA_CONFIG_PATH,
        bert_config_path=BERT_CONFIG_PATH,
        train_config_path=TRAIN_CONFIG_PATH,
    )

    assert isinstance(metrics, dict)
    # We expect at least F1 to be present in the metrics dict.
    assert "f1" in metrics
