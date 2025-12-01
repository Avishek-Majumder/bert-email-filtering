"""
Deep learning text classification architectures for spam detection.

This module implements the non-transformer models described in
"Harnessing BERT for Advanced Email Filtering in Cybersecurity":

- CNN text classifier
- LSTM-based classifier
- BiLSTM-based classifier
- Simple RNN-based classifier

Architectural and training hyperparameters are configured via
config/dl.yaml. The models consume integer token ID sequences
(e.g., from src.data.sequence_dataset.SequenceDataset) and
output logits over the classes (binary: ham vs. spam).
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


DEFAULT_DL_CONFIG_PATH = "config/dl.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_dl_config(config_path: str = DEFAULT_DL_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and return the deep learning configuration dictionary.

    Parameters
    ----------
    config_path : str
        Path to the DL YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration with "general", "embedding", and model sections.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty or cannot be parsed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DL config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"DL config file is empty or invalid: {config_path}")

    # We keep it permissive here; individual builders will check
    # for their required sections.
    return cfg


# ---------------------------------------------------------------------------
# CNN text classifier
# ---------------------------------------------------------------------------


class CNNTextClassifier(nn.Module):
    """
    1D convolutional text classifier.

    Input:
        input_ids: LongTensor of shape (batch_size, seq_len)

    Architecture:
        - Embedding layer
        - Multiple Conv1d + ReLU + max-over-time pooling branches
        - Concatenation of pooled features
        - Fully connected layers -> logits over classes
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        pad_idx: int = 0,
        num_filters: int = 100,
        filter_sizes: List[int] | None = None,
        dropout: float = 0.5,
        fc_hidden_dim: int = 128,
        activation: str = "relu",
        embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.num_classes = num_classes
        self.activation_name = activation.lower()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        # Initialize embeddings with a small normal distribution.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_init_std)

        # Convolution layers for each filter size.
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )

        conv_output_dim = num_filters * len(filter_sizes)

        self.fc1 = nn.Linear(conv_output_dim, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(fc_hidden_dim, num_classes)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(x)
        # default
        return F.relu(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor
            LongTensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        # input_ids: (B, L)
        embedded = self.embedding(input_ids)  # (B, L, E)
        # Conv1d expects (B, E, L)
        embedded = embedded.transpose(1, 2)  # (B, E, L)

        conv_outputs = []
        for conv in self.convs:
            # conv_out: (B, num_filters, L_out)
            c = conv(embedded)
            c = self._activate(c)
            # Max-over-time pooling over the temporal dimension
            # -> (B, num_filters)
            c = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            conv_outputs.append(c)

        # Concatenate pooled features
        cat = torch.cat(conv_outputs, dim=1)  # (B, num_filters * len(filter_sizes))

        x = self.dropout(self._activate(self.fc1(cat)))
        logits = self.fc_out(x)
        return logits


# ---------------------------------------------------------------------------
# LSTM / BiLSTM classifier
# ---------------------------------------------------------------------------


class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier (uni- or bi-directional).

    Input:
        input_ids: LongTensor of shape (batch_size, seq_len)

    Architecture:
        - Embedding
        - LSTM (optionally bidirectional)
        - Final hidden state(s) -> fully connected layers
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        pad_idx: int = 0,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
        fc_hidden_dim: int = 128,
        embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_init_std)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_dim = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_dim, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(fc_hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor
            LongTensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        embedded = self.embedding(input_ids)  # (B, L, E)
        output, (h_n, c_n) = self.lstm(embedded)

        # h_n: (num_layers * num_directions, B, hidden_size)
        if self.bidirectional:
            # Take the last layer's forward and backward hidden states.
            # Forward hidden state is at index -2, backward at -1.
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat([h_forward, h_backward], dim=1)  # (B, 2*H)
        else:
            # Last layer's hidden state.
            h = h_n[-1, :, :]  # (B, H)

        x = self.dropout(F.relu(self.fc1(h)))
        logits = self.fc_out(x)
        return logits


# ---------------------------------------------------------------------------
# Simple RNN classifier
# ---------------------------------------------------------------------------


class RNNClassifier(nn.Module):
    """
    Simple RNN-based text classifier (tanh or ReLU).

    Input:
        input_ids: LongTensor of shape (batch_size, seq_len)

    Architecture:
        - Embedding
        - RNN (optionally bidirectional)
        - Final hidden state(s) -> fully connected layers
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        pad_idx: int = 0,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        nonlinearity: str = "tanh",
        bidirectional: bool = False,
        fc_hidden_dim: int = 128,
        embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_init_std)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        rnn_output_dim = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(rnn_output_dim, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(fc_hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor
            LongTensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        embedded = self.embedding(input_ids)  # (B, L, E)
        output, h_n = self.rnn(embedded)

        # h_n: (num_layers * num_directions, B, hidden_size)
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat([h_forward, h_backward], dim=1)  # (B, 2*H)
        else:
            h = h_n[-1, :, :]  # (B, H)

        x = self.dropout(F.relu(self.fc1(h)))
        logits = self.fc_out(x)
        return logits


# ---------------------------------------------------------------------------
# Factory functions for building models from config
# ---------------------------------------------------------------------------


def build_cnn_classifier(
    vocab_size: int,
    num_classes: int,
    config_path: str = DEFAULT_DL_CONFIG_PATH,
) -> CNNTextClassifier:
    """
    Build a CNNTextClassifier instance using config/dl.yaml.
    """
    cfg = load_dl_config(config_path)
    emb_cfg = cfg.get("embedding", {}) or {}
    cnn_cfg = cfg.get("cnn", {}) or {}

    embedding_dim = int(emb_cfg.get("dim", 128))
    pad_idx = int(emb_cfg.get("padding_idx", 0))
    init_std = float(emb_cfg.get("init_std", 0.02))

    num_filters = int(cnn_cfg.get("num_filters", 100))
    filter_sizes = cnn_cfg.get("filter_sizes", [3, 4, 5])
    dropout = float(cnn_cfg.get("dropout", 0.5))
    fc_hidden_dim = int(cnn_cfg.get("fc_hidden_dim", 128))
    activation = str(cnn_cfg.get("activation", "relu"))

    return CNNTextClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pad_idx=pad_idx,
        num_filters=num_filters,
        filter_sizes=list(filter_sizes),
        dropout=dropout,
        fc_hidden_dim=fc_hidden_dim,
        activation=activation,
        embedding_init_std=init_std,
    )


def build_lstm_classifier(
    vocab_size: int,
    num_classes: int,
    config_path: str = DEFAULT_DL_CONFIG_PATH,
) -> LSTMClassifier:
    """
    Build a unidirectional LSTMClassifier instance using config/dl.yaml.
    """
    cfg = load_dl_config(config_path)
    emb_cfg = cfg.get("embedding", {}) or {}
    lstm_cfg = cfg.get("lstm", {}) or {}

    embedding_dim = int(emb_cfg.get("dim", 128))
    pad_idx = int(emb_cfg.get("padding_idx", 0))
    init_std = float(emb_cfg.get("init_std", 0.02))

    hidden_size = int(lstm_cfg.get("hidden_size", 128))
    num_layers = int(lstm_cfg.get("num_layers", 1))
    dropout = float(lstm_cfg.get("dropout", 0.5))
    bidirectional = bool(lstm_cfg.get("bidirectional", False))
    fc_hidden_dim = int(lstm_cfg.get("fc_hidden_dim", 128))

    return LSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pad_idx=pad_idx,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        fc_hidden_dim=fc_hidden_dim,
        embedding_init_std=init_std,
    )


def build_bilstm_classifier(
    vocab_size: int,
    num_classes: int,
    config_path: str = DEFAULT_DL_CONFIG_PATH,
) -> LSTMClassifier:
    """
    Build a bidirectional LSTMClassifier instance (BiLSTM) using config/dl.yaml.
    """
    cfg = load_dl_config(config_path)
    emb_cfg = cfg.get("embedding", {}) or {}
    bilstm_cfg = cfg.get("bilstm", {}) or {}

    embedding_dim = int(emb_cfg.get("dim", 128))
    pad_idx = int(emb_cfg.get("padding_idx", 0))
    init_std = float(emb_cfg.get("init_std", 0.02))

    hidden_size = int(bilstm_cfg.get("hidden_size", 128))
    num_layers = int(bilstm_cfg.get("num_layers", 1))
    dropout = float(bilstm_cfg.get("dropout", 0.5))
    bidirectional = bool(bilstm_cfg.get("bidirectional", True))
    fc_hidden_dim = int(bilstm_cfg.get("fc_hidden_dim", 128))

    return LSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pad_idx=pad_idx,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        fc_hidden_dim=fc_hidden_dim,
        embedding_init_std=init_std,
    )


def build_rnn_classifier(
    vocab_size: int,
    num_classes: int,
    config_path: str = DEFAULT_DL_CONFIG_PATH,
) -> RNNClassifier:
    """
    Build a RNNClassifier instance using config/dl.yaml.
    """
    cfg = load_dl_config(config_path)
    emb_cfg = cfg.get("embedding", {}) or {}
    rnn_cfg = cfg.get("rnn", {}) or {}

    embedding_dim = int(emb_cfg.get("dim", 128))
    pad_idx = int(emb_cfg.get("padding_idx", 0))
    init_std = float(emb_cfg.get("init_std", 0.02))

    hidden_size = int(rnn_cfg.get("hidden_size", 128))
    num_layers = int(rnn_cfg.get("num_layers", 1))
    dropout = float(rnn_cfg.get("dropout", 0.5))
    nonlinearity = str(rnn_cfg.get("nonlinearity", "tanh"))
    bidirectional = bool(rnn_cfg.get("bidirectional", False))
    fc_hidden_dim = int(rnn_cfg.get("fc_hidden_dim", 128))

    return RNNClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pad_idx=pad_idx,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        nonlinearity=nonlinearity,
        bidirectional=bidirectional,
        fc_hidden_dim=fc_hidden_dim,
        embedding_init_std=init_std,
    )


def build_dl_model(
    architecture: str,
    vocab_size: int,
    num_classes: int,
    config_path: str = DEFAULT_DL_CONFIG_PATH,
) -> nn.Module:
    """
    Generic factory to build a DL model by name.

    Parameters
    ----------
    architecture : str
        One of: "cnn", "lstm", "bilstm", "rnn".
    vocab_size : int
        Size of the vocabulary (number of tokens).
    num_classes : int
        Number of output classes (2 for ham vs. spam).
    config_path : str
        Path to the DL YAML configuration file.

    Returns
    -------
    nn.Module
        Instantiated PyTorch model.
    """
    name = architecture.lower()
    if name == "cnn":
        return build_cnn_classifier(vocab_size, num_classes, config_path=config_path)
    if name == "lstm":
        return build_lstm_classifier(vocab_size, num_classes, config_path=config_path)
    if name == "bilstm":
        return build_bilstm_classifier(vocab_size, num_classes, config_path=config_path)
    if name == "rnn":
        return build_rnn_classifier(vocab_size, num_classes, config_path=config_path)

    raise ValueError(
        f"Unknown DL architecture '{architecture}'. "
        "Supported values: 'cnn', 'lstm', 'bilstm', 'rnn'."
    )
