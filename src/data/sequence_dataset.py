"""
Sequence-based dataset utilities for deep learning models (CNN/LSTM/BiLSTM/RNN).

This module provides:
- a Vocabulary class to map tokens <-> integer IDs
- helpers to build a vocabulary from training texts
- functions to convert texts into fixed-length padded sequences
- a PyTorch Dataset for sequence models

The tokenization and preprocessing logic is shared with the rest of the
project via src.features.preprocessing and config/data.yaml.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.datasets import DEFAULT_DATA_CONFIG_PATH, load_data_config
from src.features.preprocessing import preprocess_text_to_tokens
from src.utils.training_utils import load_train_config, ensure_dir_exists


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class Vocabulary:
    """
    Simple vocabulary mapping tokens to integer IDs.

    Special tokens:
    - <PAD> (padding token, ID = 0)
    - <UNK> (unknown token, ID = 1)

    All other tokens are assigned IDs starting from 2.
    """

    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: Iterable[str]) -> List[int]:
        """
        Map a sequence of tokens to a sequence of IDs, using unk_id for
        out-of-vocabulary tokens.
        """
        unk = self.unk_id
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: Iterable[int]) -> List[str]:
        """
        Map a sequence of IDs back to tokens.
        """
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def to_json(self) -> Dict[str, any]:
        """
        Serialize the vocabulary to a JSON-serializable dictionary.
        """
        return {
            "token_to_id": self.token_to_id,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_json(cls, data: Dict[str, any]) -> "Vocabulary":
        """
        Construct a Vocabulary from a dictionary as produced by to_json().
        """
        token_to_id = data["token_to_id"]
        id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        pad_token = data.get("pad_token", PAD_TOKEN)
        unk_token = data.get("unk_token", UNK_TOKEN)
        return cls(token_to_id=token_to_id, id_to_token=id_to_token,
                   pad_token=pad_token, unk_token=unk_token)


DEFAULT_VOCAB_FILENAME = "vocab.json"


def _get_vocab_config(
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Tuple[int, int]:
    """
    Retrieve vocabulary-related parameters from config/data.yaml.

    Returns
    -------
    Tuple[int, int]
        (max_size, min_freq)
    """
    cfg = load_data_config(config_path)
    vocab_cfg = cfg["preprocessing"].get("vocab", {}) or {}
    max_size = int(vocab_cfg.get("max_size", 20000))
    min_freq = int(vocab_cfg.get("min_freq", 2))
    return max_size, min_freq


def build_vocab_from_texts(
    texts: Iterable[str],
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Vocabulary:
    """
    Build a vocabulary from an iterable of raw texts using the
    non-BERT preprocessing pipeline.

    Steps:
    - preprocess each text into tokens
    - count token frequencies
    - keep tokens above min_freq, up to max_size
    - add special tokens <PAD> and <UNK>

    Parameters
    ----------
    texts : Iterable[str]
        Raw text strings (e.g., training messages).
    config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    Vocabulary
        Constructed vocabulary instance.
    """
    max_size, min_freq = _get_vocab_config(config_path)

    freq: Dict[str, int] = {}
    for text in texts:
        tokens = preprocess_text_to_tokens(text, config_path=config_path)
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

    # Sort tokens by frequency (descending), then lexicographically.
    sorted_tokens = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

    # Reserve IDs for special tokens.
    token_to_id: Dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    next_id = 2

    for token, count in sorted_tokens:
        if count < min_freq:
            continue
        if token in token_to_id:
            continue
        if max_size is not None and next_id >= max_size:
            break
        token_to_id[token] = next_id
        next_id += 1

    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)


def save_vocab(
    vocab: Vocabulary,
    artifacts_dir: Optional[str] = None,
    filename: str = DEFAULT_VOCAB_FILENAME,
) -> str:
    """
    Save the vocabulary to a JSON file under the artifacts directory.

    Parameters
    ----------
    vocab : Vocabulary
        Vocabulary instance to save.
    artifacts_dir : Optional[str]
        Directory where vocab should be stored. If None, this will be
        determined from config/train.yaml.
    filename : str
        File name for the saved vocabulary.

    Returns
    -------
    str
        Full path to the saved vocabulary file.
    """
    if artifacts_dir is None:
        train_cfg = load_train_config()
        artifacts_dir = train_cfg["paths"]["artifacts_dir"]

    ensure_dir_exists(artifacts_dir)
    path = os.path.join(artifacts_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_json(), f, ensure_ascii=False, indent=2)

    return path


def load_vocab(
    artifacts_dir: Optional[str] = None,
    filename: str = DEFAULT_VOCAB_FILENAME,
) -> Vocabulary:
    """
    Load a previously saved vocabulary from disk.

    Parameters
    ----------
    artifacts_dir : Optional[str]
        Directory where vocab is stored. If None, this will be determined
        from config/train.yaml.
    filename : str
        File name of the saved vocabulary.

    Returns
    -------
    Vocabulary
        Loaded vocabulary instance.

    Raises
    ------
    FileNotFoundError
        If the vocabulary file does not exist.
    """
    if artifacts_dir is None:
        train_cfg = load_train_config()
        artifacts_dir = train_cfg["paths"]["artifacts_dir"]

    path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vocabulary file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Vocabulary.from_json(data)


# ---------------------------------------------------------------------------
# Text -> sequence conversion
# ---------------------------------------------------------------------------


def _get_sequence_max_length(
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> int:
    """
    Retrieve the maximum sequence length from config/data.yaml.

    Returns
    -------
    int
        Maximum token sequence length.
    """
    cfg = load_data_config(config_path)
    seq_cfg = cfg["preprocessing"].get("sequence", {}) or {}
    max_length = int(seq_cfg.get("max_length", 100))
    return max_length


def text_to_sequence(
    text: str,
    vocab: Vocabulary,
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
    max_length: Optional[int] = None,
) -> List[int]:
    """
    Convert a single raw text into a fixed-length sequence of token IDs.

    Steps:
    - preprocess text into tokens
    - map tokens to IDs via the vocabulary (with unk_id for OOV tokens)
    - truncate or pad to max_length using pad_id

    Parameters
    ----------
    text : str
        Raw text message.
    vocab : Vocabulary
        Vocabulary instance.
    config_path : str
        Path to the data YAML configuration.
    max_length : Optional[int]
        Maximum sequence length. If None, uses value from config.

    Returns
    -------
    List[int]
        Padded/truncated sequence of token IDs.
    """
    if max_length is None:
        max_length = _get_sequence_max_length(config_path)

    tokens = preprocess_text_to_tokens(text, config_path=config_path)
    ids = vocab.encode(tokens)

    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        pad_len = max_length - len(ids)
        ids = ids + [vocab.pad_id] * pad_len

    return ids


def texts_to_sequences(
    texts: Iterable[str],
    vocab: Vocabulary,
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Convert an iterable of raw texts into a 2D numpy array of
    padded/truncated sequences of token IDs.

    Parameters
    ----------
    texts : Iterable[str]
        Raw text messages.
    vocab : Vocabulary
        Vocabulary instance.
    config_path : str
        Path to the data YAML configuration.
    max_length : Optional[int]
        Maximum sequence length.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, max_length) with integer token IDs.
    """
    sequences: List[List[int]] = []
    for text in texts:
        seq = text_to_sequence(
            text=text,
            vocab=vocab,
            config_path=config_path,
            max_length=max_length,
        )
        sequences.append(seq)

    return np.array(sequences, dtype=np.int64)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence-based deep learning models.

    Each item is a dictionary with:
    - "input_ids": LongTensor of shape (max_length,)
    - "label": LongTensor scalar with the class index (0 or 1)
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Vocabulary,
        config_path: str = DEFAULT_DATA_CONFIG_PATH,
        max_length: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        texts : Sequence[str]
            Raw text messages.
        labels : Sequence[int]
            Numeric labels (e.g., 0 for ham, 1 for spam).
        vocab : Vocabulary
            Vocabulary instance.
        config_path : str
            Path to the data YAML configuration.
        max_length : Optional[int]
            Maximum sequence length. If None, uses config value.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Number of texts ({len(texts)}) and labels ({len(labels)}) "
                "must be the same."
            )

        self.vocab = vocab
        self.config_path = config_path
        self.max_length = max_length or _get_sequence_max_length(config_path)

        # Precompute sequences for all texts for simplicity and speed.
        sequences_np = texts_to_sequences(
            texts=texts,
            vocab=vocab,
            config_path=config_path,
            max_length=self.max_length,
        )

        self.input_ids = torch.from_numpy(sequences_np)  # shape: (N, L)
        self.labels = torch.tensor(labels, dtype=torch.long)  # shape: (N,)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "label": self.labels[idx],
        }
