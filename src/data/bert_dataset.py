"""
BERT-specific dataset utilities.

This module provides:
- a function to load the BERT configuration (config/bert.yaml)
- a factory to build a Hugging Face tokenizer
- a PyTorch Dataset that:
    * applies lightweight preprocessing for BERT
    * tokenizes text into input_ids and attention_mask
    * returns labels suitable for binary classification

The dataset is designed to work with the SMS Spam Collection dataset
prepared via src.data.datasets (columns: "text", "label_id").
"""

from __future__ import annotations

import os
from typing import Dict, Any, Sequence, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml
from transformers import AutoTokenizer

from src.data.datasets import DEFAULT_DATA_CONFIG_PATH, load_data_config
from src.features.preprocessing import preprocess_text_for_bert


DEFAULT_BERT_CONFIG_PATH = "config/bert.yaml"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_bert_config(config_path: str = DEFAULT_BERT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and return the full BERT configuration dictionary.

    Parameters
    ----------
    config_path : str
        Path to the BERT YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration with "general", "model", and "training" sections.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty or cannot be parsed.
    KeyError
        If required sections are missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"BERT config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"BERT config file is empty or invalid: {config_path}")

    for section in ("general", "model", "training"):
        if section not in cfg:
            raise KeyError(f'Missing "{section}" section in BERT config: {config_path}')

    return cfg


# ---------------------------------------------------------------------------
# Tokenizer factory
# ---------------------------------------------------------------------------


_tokenizer_cache: Optional[AutoTokenizer] = None


def get_bert_tokenizer(
    bert_config_path: str = DEFAULT_BERT_CONFIG_PATH,
) -> AutoTokenizer:
    """
    Build (or reuse) a Hugging Face tokenizer based on the BERT config.

    Parameters
    ----------
    bert_config_path : str
        Path to the BERT YAML configuration file.

    Returns
    -------
    AutoTokenizer
        A tokenizer instance corresponding to the configured model.
    """
    global _tokenizer_cache

    if _tokenizer_cache is not None:
        return _tokenizer_cache

    cfg = load_bert_config(bert_config_path)
    model_name = cfg["model"]["pretrained_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _tokenizer_cache = tokenizer
    return tokenizer


# ---------------------------------------------------------------------------
# PyTorch Dataset for BERT
# ---------------------------------------------------------------------------


class BERTTextDataset(Dataset):
    """
    PyTorch Dataset for BERT-based spam detection.

    Each item is a dictionary with:
    - "input_ids": LongTensor of shape (max_seq_length,)
    - "attention_mask": LongTensor of shape (max_seq_length,)
    - "label": LongTensor scalar with the class index (0 or 1)
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
        bert_config_path: str = DEFAULT_BERT_CONFIG_PATH,
    ) -> None:
        """
        Parameters
        ----------
        texts : Sequence[str]
            Raw text messages.
        labels : Sequence[int]
            Numeric labels (e.g., 0 for ham, 1 for spam).
        data_config_path : str
            Path to the data YAML configuration (for light BERT preprocessing).
        bert_config_path : str
            Path to the BERT YAML configuration (for model and max_seq_length).
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Number of texts ({len(texts)}) and labels ({len(labels)}) "
                "must be the same."
            )

        self.data_config_path = data_config_path
        self.bert_config = load_bert_config(bert_config_path)
        self.tokenizer = get_bert_tokenizer(bert_config_path)

        self.max_seq_length = int(self.bert_config["model"].get("max_seq_length", 128))

        # Store raw texts and labels; tokenization is done on-the-fly in __getitem__
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_text = self.texts[idx]
        label = self.labels[idx]

        # Lightweight preprocessing for BERT (lowercase + whitespace normalization).
        processed_text = preprocess_text_for_bert(
            raw_text, config_path=self.data_config_path
        )

        encoded = self.tokenizer(
            processed_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # encoded["input_ids"]: (1, L), encoded["attention_mask"]: (1, L)
        input_ids = encoded["input_ids"].squeeze(0)  # shape: (L,)
        attention_mask = encoded["attention_mask"].squeeze(0)  # shape: (L,)

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helper to build dataset from DataFrame
# ---------------------------------------------------------------------------


def build_bert_dataset_from_df(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label_id",
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    bert_config_path: str = DEFAULT_BERT_CONFIG_PATH,
) -> BERTTextDataset:
    """
    Build a BERTTextDataset from a DataFrame with text and label columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least text_column and label_column.
    text_column : str
        Name of the column with raw text.
    label_column : str
        Name of the column with numeric labels.
    data_config_path : str
        Path to the data YAML configuration.
    bert_config_path : str
        Path to the BERT YAML configuration.

    Returns
    -------
    BERTTextDataset
        Constructed dataset instance.
    """
    if text_column not in df.columns:
        raise KeyError(
            f"Text column '{text_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if label_column not in df.columns:
        raise KeyError(
            f"Label column '{label_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(int).tolist()

    return BERTTextDataset(
        texts=texts,
        labels=labels,
        data_config_path=data_config_path,
        bert_config_path=bert_config_path,
    )
