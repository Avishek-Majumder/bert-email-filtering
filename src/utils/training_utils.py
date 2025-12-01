"""
Training and utility helpers.

This module centralizes common functionality used across the project:

- loading the global training configuration (config/train.yaml)
- ensuring directories exist before writing files
- setting random seeds for reproducibility
- selecting the appropriate device (CPU/GPU)
- constructing loggers that respect config/logging settings

All training pipelines (ML, DL, BERT) rely on these utilities.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


DEFAULT_TRAIN_CONFIG_PATH = "config/train.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_train_config(
    config_path: str = DEFAULT_TRAIN_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Load and return the global training configuration dictionary.

    Parameters
    ----------
    config_path : str
        Path to the train YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration with sections such as "general", "paths",
        "logging", "evaluation", and "save".

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty or cannot be parsed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Train config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Train config file is empty or invalid: {config_path}")

    # We keep this permissive: downstream code will access the keys it needs.
    return cfg


# ---------------------------------------------------------------------------
# Filesystem utilities
# ---------------------------------------------------------------------------


def ensure_dir_exists(path: str) -> None:
    """
    Ensure that a directory exists (create it if necessary).

    Parameters
    ----------
    path : str
        Directory path.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility utilities
# ---------------------------------------------------------------------------


def seed_everything(
    seed: int = 42,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducible experiments.

    Parameters
    ----------
    seed : int
        Global random seed.
    deterministic : bool
        If True and PyTorch is available, enable deterministic behavior
        in cuDNN where possible.
    cudnn_benchmark : bool
        If True and PyTorch is available, enable cuDNN benchmarking (may
        improve speed but can affect determinism).
    """
    random.seed(seed)
    np.random.seed(seed)

    if not _TORCH_AVAILABLE:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN settings
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device(train_cfg: Dict[str, Any]) -> "torch.device":
    """
    Select the appropriate device (CPU or GPU) based on configuration
    and availability.

    Parameters
    ----------
    train_cfg : Dict[str, Any]
        Global training configuration (typically from config/train.yaml).

    Returns
    -------
    torch.device
        Selected device.
    """
    if not _TORCH_AVAILABLE:
        # Fallback: treat as CPU-only environment.
        return "cpu"  # type: ignore[return-value]

    general_cfg = train_cfg.get("general", {}) or {}
    preferred = str(general_cfg.get("device", "cuda")).lower()

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------


def _parse_log_level(level_str: str) -> int:
    """
    Convert a string log level into a logging module constant.

    Parameters
    ----------
    level_str : str
        One of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" (case-insensitive).

    Returns
    -------
    int
        Corresponding logging level.
    """
    level_str = (level_str or "INFO").upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(level_str, logging.INFO)


def get_logger(
    name: str,
    config: Dict[str, Any],
    log_file_suffix: Optional[str] = None,
) -> logging.Logger:
    """
    Construct and return a logger that respects the logging section of
    the global training config.

    Parameters
    ----------
    name : str
        Logger name.
    config : Dict[str, Any]
        Global training configuration.
    log_file_suffix : Optional[str]
        Optional suffix appended to the log file name (e.g., "ml", "dl").

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, assume it's already configured.
    if logger.handlers:
        return logger

    logging_cfg = config.get("logging", {}) or {}
    paths_cfg = config.get("paths", {}) or {}

    level_str = logging_cfg.get("level", "INFO")
    level = _parse_log_level(level_str)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    to_file = bool(logging_cfg.get("to_file", True))
    if to_file:
        logs_dir = paths_cfg.get("logs_dir", "experiments/logs")
        ensure_dir_exists(logs_dir)

        file_prefix = logging_cfg.get("file_prefix", "training_log")
        if log_file_suffix:
            filename = f"{file_prefix}_{log_file_suffix}.log"
        else:
            filename = f"{file_prefix}.log"

        file_path = os.path.join(logs_dir, filename)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
