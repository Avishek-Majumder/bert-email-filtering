"""
Training and evaluation pipeline for deep learning (non-BERT) models.

This module reproduces the CNN/LSTM/BiLSTM/RNN experiments from
"Harnessing BERT for Advanced Email Filtering in Cybersecurity" by:

- loading the SMS Spam Collection dataset
- performing a stratified 70/30 train–test split
- building a token vocabulary from training texts
- converting texts into padded integer sequences
- training the following architectures:
    * CNNTextClassifier
    * LSTMClassifier
    * BiLSTMClassifier
    * RNNClassifier
- evaluating each model via accuracy, precision, recall, F1-score
- saving metrics and model weights under experiments/

The actual model architectures are defined in src.models.dl_architectures.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data.datasets import load_sms_dataset, DEFAULT_DATA_CONFIG_PATH
from src.data.split import train_test_split_df
from src.data.sequence_dataset import (
    Vocabulary,
    build_vocab_from_texts,
    save_vocab,
    load_vocab,
    SequenceDataset,
)
from src.models.dl_architectures import (
    load_dl_config,
    build_dl_model,
)
from src.utils.training_utils import (
    load_train_config,
    ensure_dir_exists,
    seed_everything,
    get_logger,
    get_device,
)
from src.evaluation.metrics import compute_classification_metrics


# ---------------------------------------------------------------------------
# Helpers: dataset / loaders
# ---------------------------------------------------------------------------


def _build_vocab_and_datasets(
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    dl_config_path: str = "config/dl.yaml",
) -> Tuple[SequenceDataset, SequenceDataset, Vocabulary]:
    """
    Load the dataset, split train/test, build a vocabulary from training texts,
    and construct SequenceDataset instances for train and test.

    Parameters
    ----------
    data_config_path : str
        Path to config/data.yaml.
    dl_config_path : str
        Path to config/dl.yaml (used indirectly for sequence length).

    Returns
    -------
    Tuple[SequenceDataset, SequenceDataset, Vocabulary]
        (train_dataset, test_dataset, vocab)
    """
    # Load full dataset and split
    df, label_mapping = load_sms_dataset(config_path=data_config_path)
    train_df, test_df = train_test_split_df(
        df,
        label_column="label_id",
        config_path=data_config_path,
    )

    # Build vocabulary from training texts
    vocab = build_vocab_from_texts(
        texts=train_df["text"].astype(str).tolist(),
        config_path=data_config_path,
    )

    # Persist vocabulary to artifacts dir
    save_vocab(vocab)

    # Build SequenceDataset for train and test
    train_dataset = SequenceDataset(
        texts=train_df["text"].astype(str).tolist(),
        labels=train_df["label_id"].astype(int).tolist(),
        vocab=vocab,
        config_path=data_config_path,
        max_length=None,  # use config value
    )

    test_dataset = SequenceDataset(
        texts=test_df["text"].astype(str).tolist(),
        labels=test_df["label_id"].astype(int).tolist(),
        vocab=vocab,
        config_path=data_config_path,
        max_length=None,
    )

    return train_dataset, test_dataset, vocab


def _build_dataloaders(
    train_dataset: SequenceDataset,
    test_dataset: SequenceDataset,
    dl_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Build DataLoaders for train, optional validation, and test sets.

    Parameters
    ----------
    train_dataset : SequenceDataset
        Full training dataset.
    test_dataset : SequenceDataset
        Test dataset.
    dl_cfg : Dict[str, Any]
        Deep learning configuration (config/dl.yaml).
    train_cfg : Dict[str, Any]
        Global training configuration (config/train.yaml).

    Returns
    -------
    Tuple[DataLoader, Optional[DataLoader], DataLoader]
        (train_loader, val_loader, test_loader)
    """
    general_cfg = dl_cfg.get("general", {}) or {}
    batch_size = int(general_cfg.get("batch_size", 32))
    use_val = bool(general_cfg.get("use_validation_split", True))
    val_split = float(general_cfg.get("validation_split", 0.1))

    num_workers = int(train_cfg["general"].get("num_workers", 2))
    pin_memory = True if train_cfg["general"].get("device", "cuda") == "cuda" else False

    # Optional train/validation split
    if use_val and 0.0 < val_split < 1.0 and len(train_dataset) > 1:
        val_size = int(len(train_dataset) * val_split)
        if val_size < 1:
            val_size = 1
        train_size = len(train_dataset) - val_size

        generator = torch.Generator()
        generator.manual_seed(int(train_cfg["general"].get("random_state", 42)))

        train_subset, val_subset = random_split(
            train_dataset,
            lengths=[train_size, val_size],
            generator=generator,
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # No validation set
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training + evaluation for a single architecture
# ---------------------------------------------------------------------------


def _build_optimizer_and_scheduler(
    model: nn.Module,
    dl_cfg: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Construct optimizer and optional LR scheduler from dl_cfg.

    Parameters
    ----------
    model : nn.Module
        Model to optimize.
    dl_cfg : Dict[str, Any]
        Deep learning configuration.

    Returns
    -------
    Tuple[Optimizer, Optional[_LRScheduler]]
        (optimizer, scheduler)
    """
    general_cfg = dl_cfg.get("general", {}) or {}
    opt_cfg = dl_cfg.get("optimization", {}) or {}

    lr = float(general_cfg.get("learning_rate", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    opt_name = str(opt_cfg.get("optimizer", "adam")).lower()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    scheduler_cfg = opt_cfg.get("scheduler", {}) or {}
    if not bool(scheduler_cfg.get("enabled", False)):
        return optimizer, None

    sched_type = str(scheduler_cfg.get("type", "step_lr")).lower()
    if sched_type == "step_lr":
        step_size = int(scheduler_cfg.get("step_size", 5))
        gamma = float(scheduler_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        # Fallback: no scheduler
        scheduler = None

    return optimizer, scheduler


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: Optional[float] = None,
) -> float:
    """
    Train model for a single epoch.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)  # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


def _evaluate_dl_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on a validation or test set.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, total_batches)
    metrics = compute_classification_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        average="binary",
    )
    return avg_loss, metrics


def train_and_evaluate_single_dl_model(
    architecture: str,
    train_dataset: SequenceDataset,
    test_dataset: SequenceDataset,
    vocab: Vocabulary,
    dl_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    results_dir: str,
    models_dir: str,
    device: torch.device,
    logger,
) -> Dict[str, Any]:
    """
    Train and evaluate a single DL architecture (cnn, lstm, bilstm, rnn).

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary including the model name.
    """
    num_classes = 2  # ham vs spam
    vocab_size = vocab.size

    model = build_dl_model(
        architecture=architecture,
        vocab_size=vocab_size,
        num_classes=num_classes,
        config_path="config/dl.yaml",
    ).to(device)

    general_cfg = dl_cfg.get("general", {}) or {}
    epochs = int(general_cfg.get("epochs", 10))
    max_grad_norm = general_cfg.get("max_grad_norm", 5.0)
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

    logger.info("Training DL model '%s' for %d epochs.", architecture, epochs)

    # DataLoaders (train/val/test)
    train_loader, val_loader, test_loader = _build_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        dl_cfg=dl_cfg,
        train_cfg=train_cfg,
    )

    # Criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = _build_optimizer_and_scheduler(model, dl_cfg)

    best_val_loss = math.inf
    log_every = int(train_cfg["logging"].get("log_every_n_steps", 50))

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
        )

        msg = f"[{architecture}] Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}"

        if val_loader is not None:
            val_loss, val_metrics = _evaluate_dl_model(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
            )
            msg += (
                f", val_loss: {val_loss:.4f}, "
                f"val_acc: {val_metrics['accuracy']:.4f}, "
                f"val_f1: {val_metrics['f1']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        logger.info(msg)

        if scheduler is not None:
            scheduler.step()

    # Final evaluation on test set
    test_loss, test_metrics = _evaluate_dl_model(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
    )

    logger.info(
        "[%s] Test - loss: %.4f, acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f",
        architecture,
        test_loss,
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    # Save metrics
    metrics_with_name = {
        "model": architecture,
        "test_loss": float(test_loss),
        **test_metrics,
    }
    metrics_path = os.path.join(results_dir, f"metrics_dl_{architecture}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_with_name, f, indent=2)
    logger.info("Saved DL metrics for '%s' to %s", architecture, metrics_path)

    # Save model weights
    save_models = bool(train_cfg["save"].get("save_models", True))
    if save_models:
        ensure_dir_exists(models_dir)
        model_path = os.path.join(models_dir, f"dl_{architecture}.pt")
        overwrite = bool(train_cfg["save"].get("overwrite_existing", False))
        if not os.path.exists(model_path) or overwrite:
            torch.save(model.state_dict(), model_path)
            logger.info("Saved DL model '%s' weights to %s", architecture, model_path)
        else:
            logger.info(
                "DL model file already exists and overwrite_existing is False: %s",
                model_path,
            )

    return metrics_with_name


# ---------------------------------------------------------------------------
# Orchestrator: train all DL models
# ---------------------------------------------------------------------------


def train_and_evaluate_dl_models(
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    dl_config_path: str = "config/dl.yaml",
    train_config_path: str = "config/train.yaml",
) -> None:
    """
    End-to-end pipeline to train and evaluate all DL models:
    CNN, LSTM, BiLSTM, and RNN.

    Parameters
    ----------
    data_config_path : str
        Path to config/data.yaml.
    dl_config_path : str
        Path to config/dl.yaml.
    train_config_path : str
        Path to config/train.yaml.
    """
    # Load configs
    dl_cfg = load_dl_config(dl_config_path)
    train_cfg = load_train_config(train_config_path)

    seed = int(train_cfg["general"].get("random_state", 42))
    deterministic = bool(train_cfg["general"].get("deterministic", True))
    cudnn_benchmark = bool(train_cfg["general"].get("cudnn_benchmark", False))
    seed_everything(seed=seed, deterministic=deterministic, cudnn_benchmark=cudnn_benchmark)

    logger = get_logger(
        name="train_dl",
        config=train_cfg,
        log_file_suffix="dl",
    )

    device = get_device(train_cfg)
    logger.info("Using device: %s", device)

    # Build datasets & vocab
    train_dataset, test_dataset, vocab = _build_vocab_and_datasets(
        data_config_path=data_config_path,
        dl_config_path=dl_config_path,
    )

    logger.info(
        "Train dataset size: %d, Test dataset size: %d, Vocab size: %d",
        len(train_dataset),
        len(test_dataset),
        vocab.size,
    )

    # Prepare output directories
    results_dir = train_cfg["paths"]["results_dir"]
    models_dir = train_cfg["paths"]["models_dir"]
    ensure_dir_exists(results_dir)
    ensure_dir_exists(models_dir)

    architectures = ["cnn", "lstm", "bilstm", "rnn"]
    aggregated_metrics: List[Dict[str, Any]] = []

    for arch in architectures:
        logger.info("=" * 80)
        logger.info("Starting training for DL architecture: %s", arch)

        metrics = train_and_evaluate_single_dl_model(
            architecture=arch,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            vocab=vocab,
            dl_cfg=dl_cfg,
            train_cfg=train_cfg,
            results_dir=results_dir,
            models_dir=models_dir,
            device=device,
            logger=logger,
        )
        aggregated_metrics.append(metrics)

    # Save aggregated results
    aggregated_path = os.path.join(results_dir, "dl_results.json")
    with open(aggregated_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_metrics, f, indent=2)
    logger.info("Saved aggregated DL metrics to %s", aggregated_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point when running this module as a script.
    """
    train_and_evaluate_dl_models()


if __name__ == "__main__":
    main()
