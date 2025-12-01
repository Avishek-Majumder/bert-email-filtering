"""
BERT-based classifier for spam detection.

This module provides a simple factory to construct a BERT model for
sequence classification, configured via config/bert.yaml.

We use Hugging Face's `AutoModelForSequenceClassification` with:
- a pretrained backbone (e.g., "bert-base-uncased")
- a classification head producing logits over two classes (ham vs. spam)
"""

from __future__ import annotations

import os
from typing import Dict, Any

import yaml
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.data.bert_dataset import (
    DEFAULT_BERT_CONFIG_PATH,
    load_bert_config,
)


def build_bert_classifier(
    bert_config_path: str = DEFAULT_BERT_CONFIG_PATH,
) -> AutoModelForSequenceClassification:
    """
    Build a BERT-based sequence classification model using config/bert.yaml.

    This helper:
    - loads the BERT configuration (general, model, training sections)
    - constructs a Hugging Face AutoConfig with:
        * pretrained model name
        * number of labels
        * dropout probabilities
    - instantiates AutoModelForSequenceClassification with that config

    Parameters
    ----------
    bert_config_path : str
        Path to the BERT YAML configuration file.

    Returns
    -------
    AutoModelForSequenceClassification
        Pretrained BERT model with a classification head on top.
    """
    cfg = load_bert_config(bert_config_path)
    model_cfg = cfg["model"]

    pretrained_name = model_cfg.get("pretrained_model_name", "bert-base-uncased")
    num_labels = int(model_cfg.get("num_labels", 2))
    dropout = float(model_cfg.get("dropout", 0.3))

    # Build configuration for sequence classification.
    hf_config = AutoConfig.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        config=hf_config,
    )

    return model
