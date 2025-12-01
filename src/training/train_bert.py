"""
Training and evaluation pipeline for the BERT-based spam classifier.

This module reproduces the BERT experiments from
"Harnessing BERT for Advanced Email Filtering in Cybersecurity" by:

- loading the SMS Spam Collection dataset
- performing a stratified 70/30 train–test split
- building BERT-ready datasets (input_ids, attention_mask, labels)
- fine-tuning a pretrained BERT model for 3 epochs with:
    * batch size = 16
    * learning rate = 2e-5
- evaluating on the test set using accuracy, precision, recall, F1-score
- saving metrics and the best model under experiments/

We implement a clean PyTorch + Hugging Face training loop instead of
using the higher-level Trainer, to keep dependencies minimal and
logic transparent.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW,
)

from src.data.datasets import load_sms_dataset, DEFAULT_DATA_CONFIG_PATH
from src.data.split import train_test_split_df
from src.data.bert_dataset import (
    build_bert_dataset_from_df,
    load_bert_config,
    DEFAULT_BERT_CONFIG_PATH,
)
from src.models.bert_classifier import build_bert_classifier
from src.utils.training_utils import (
    load_train_config,
    ensure_dir_exists,
    seed_everything,
    get_logger,
    get_device,
)
from src.evaluation.metrics import compute_classification_metrics


# ---------------------------------------------------------------------------
# Data / DataLoader helpers
# ---------------------------------------------------------------------------


def _build_bert_datasets(
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    bert_config_path: str = DEFAULT_BERT_CONFIG_PATH,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load the dataset, perform a stratified 70/30 train–test split, and
    build BERTTextDataset instances for train and test.

    Parameters
    ----------
    data_config_path : str
        Path to config/data.yaml.
    bert_config_path : str
        Path to config/bert.yaml.

    Returns
    -------
    Tuple[Dataset, Dataset]
        (train_dataset, test_dataset)
    """
    df, label_mapping = load_sms_dataset(config_path=data_config_path)
    train_df, test_df = train_test_split_df(
        df,
        label_column="label_id",
        config_path=data_config_path,
    )

    train_dataset = build_bert_dataset_from_df(
        train_df,
        text_column="text",
        label_column="label_id",
        data_config_path=data_config_path,
        bert_config_path=bert_config_path,
    )

    test_dataset = build_bert_dataset_from_df(
        test_df,
        text_column="text",
        label_column="label_id",
        data_config_path=data_config_path,
        bert_config_path=bert_config_path,
    )

    return train_dataset, test_dataset


def _build_bert_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    bert_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Build DataLoaders for train, optional validation, and test sets.

    Train/validation split is controlled by:
      - bert_cfg["general"]["use_validation_split"]
      - bert_cfg["general"]["validation_split"]

    Parameters
    ----------
    train_dataset : Dataset
        Full training dataset.
    test_dataset : Dataset
        Test dataset.
    bert_cfg : Dict[str, Any]
        BERT configuration (config/bert.yaml).
    train_cfg : Dict[str, Any]
        Global training configuration (config/train.yaml).

    Returns
    -------
    Tuple[DataLoader, Optional[DataLoader], DataLoader]
        (train_loader, val_loader, test_loader)
    """
    general_cfg = bert_cfg.get("general", {}) or {}
    train_settings = bert_cfg.get("training", {}) or {}

    batch_size = int(train_settings.get("batch_size", 16))
    use_val = bool(general_cfg.get("use_validation_split", True))
    val_split = float(general_cfg.get("validation_split", 0.1))

    num_workers = int(train_cfg["general"].get("num_workers", 2))
    pin_memory = True if train_cfg["general"].get("device", "cuda") == "cuda" else False

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
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def _build_bert_optimizer_and_scheduler(
    model: torch.nn.Module,
    bert_cfg: Dict[str, Any],
    num_train_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Build AdamW optimizer and linear warmup scheduler for BERT fine-tuning.

    Parameters
    ----------
    model : nn.Module
        BERT model to optimize.
    bert_cfg : Dict[str, Any]
        BERT configuration (config/bert.yaml).
    num_train_steps : int
        Total number of training steps (len(train_loader) * num_epochs).

    Returns
    -------
    Tuple[Optimizer, _LRScheduler]
        (optimizer, scheduler)
    """
    train_cfg = bert_cfg.get("training", {}) or {}

    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    num_warmup_steps = int(num_train_steps * warmup_ratio)

    # Standard AdamW for BERT fine-tuning
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    return optimizer, scheduler


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    max_grad_norm: float,
    fp16: bool,
    logger,
    epoch: int,
    num_epochs: int,
    logging_steps: int,
) -> float:
    """
    Train BERT for a single epoch.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_steps = 0

    if fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        if step % logging_steps == 0:
            avg_loss_so_far = total_loss / max(1, total_steps)
            logger.info(
                "[BERT] Epoch %d/%d, step %d/%d, loss: %.4f",
                epoch,
                num_epochs,
                step,
                len(loader),
                avg_loss_so_far,
            )

    return total_loss / max(1, total_steps)


def _evaluate_bert(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate BERT model on a validation or test set.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_steps += 1

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, total_steps)
    metrics = compute_classification_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        average="binary",
    )
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Orchestrator: train + evaluate BERT
# ---------------------------------------------------------------------------


def train_and_evaluate_bert(
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    bert_config_path: str = DEFAULT_BERT_CONFIG_PATH,
    train_config_path: str = "config/train.yaml",
) -> Dict[str, Any]:
    """
    End-to-end pipeline to fine-tune BERT and evaluate on the test set.

    Parameters
    ----------
    data_config_path : str
        Path to config/data.yaml.
    bert_config_path : str
        Path to config/bert.yaml.
    train_config_path : str
        Path to config/train.yaml.

    Returns
    -------
    Dict[str, Any]
        Dictionary of test metrics (including model name).
    """
    # Load configs
    bert_cfg = load_bert_config(bert_config_path)
    train_cfg = load_train_config(train_config_path)

    seed = int(train_cfg["general"].get("random_state", 42))
    deterministic = bool(train_cfg["general"].get("deterministic", True))
    cudnn_benchmark = bool(train_cfg["general"].get("cudnn_benchmark", False))
    seed_everything(seed=seed, deterministic=deterministic, cudnn_benchmark=cudnn_benchmark)

    logger = get_logger(
        name="train_bert",
        config=train_cfg,
        log_file_suffix="bert",
    )

    device = get_device(train_cfg)
    logger.info("Using device: %s", device)

    # Prepare datasets and loaders
    train_dataset, test_dataset = _build_bert_datasets(
        data_config_path=data_config_path,
        bert_config_path=bert_config_path,
    )

    logger.info(
        "Train dataset size: %d, Test dataset size: %d",
        len(train_dataset),
        len(test_dataset),
    )

    train_loader, val_loader, test_loader = _build_bert_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        bert_cfg=bert_cfg,
        train_cfg=train_cfg,
    )

    # Build model
    model = build_bert_classifier(bert_config_path=bert_config_path)
    model.to(device)

    training_cfg = bert_cfg.get("training", {}) or {}
    num_epochs = int(training_cfg.get("epochs", 3))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    fp16 = bool(training_cfg.get("fp16", False))
    logging_steps = int(training_cfg.get("logging_steps", 50))
    eval_steps = int(training_cfg.get("eval_steps", 200))

    total_train_steps = len(train_loader) * num_epochs
    optimizer, scheduler = _build_bert_optimizer_and_scheduler(
        model=model,
        bert_cfg=bert_cfg,
        num_train_steps=total_train_steps,
    )

    logger.info(
        "Starting BERT fine-tuning for %d epochs (%d total steps).",
        num_epochs,
        total_train_steps,
    )

    # Best-model tracking
    save_best = bool(training_cfg.get("save_best_model", True))
    metric_name = training_cfg.get("metric_for_best_model", "f1")
    greater_is_better = bool(training_cfg.get("greater_is_better", True))
    best_metric_value: Optional[float] = None
    best_model_state: Optional[Dict[str, torch.Tensor]] = None

    # Training loop
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            fp16=fp16,
            logger=logger,
            epoch=epoch,
            num_epochs=num_epochs,
            logging_steps=logging_steps,
        )

        logger.info(
            "[BERT] Epoch %d/%d finished. Train loss: %.4f",
            epoch,
            num_epochs,
            train_loss,
        )

        # Optional validation evaluation
        if val_loader is not None:
            val_loss, val_metrics = _evaluate_bert(
                model=model,
                loader=val_loader,
                device=device,
            )
            logger.info(
                "[BERT] Epoch %d/%d - Val loss: %.4f, acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f",
                epoch,
                num_epochs,
                val_loss,
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
            )

            if save_best:
                current_metric_value = float(val_metrics.get(metric_name, 0.0))
                if best_metric_value is None:
                    best_metric_value = current_metric_value
                    best_model_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    improved = (
                        current_metric_value > best_metric_value
                        if greater_is_better
                        else current_metric_value < best_metric_value
                    )
                    if improved:
                        best_metric_value = current_metric_value
                        best_model_state = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }

    # If we tracked a best model on validation, load it before test.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(
            "Loaded best BERT model (based on '%s', value=%.4f) for final test evaluation.",
            metric_name,
            best_metric_value,
        )

    # Final test evaluation
    test_loss, test_metrics = _evaluate_bert(
        model=model,
        loader=test_loader,
        device=device,
    )

    logger.info(
        "[BERT] Test - loss: %.4f, acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f",
        test_loss,
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    # Prepare output directories
    paths_cfg = train_cfg.get("paths", {}) or {}
    results_dir = paths_cfg.get("results_dir", "experiments/results")
    models_dir = paths_cfg.get("models_dir", "experiments/models")
    ensure_dir_exists(results_dir)
    ensure_dir_exists(models_dir)

    bert_general = bert_cfg.get("general", {}) or {}
    bert_output_dir = bert_general.get("output_dir", "experiments/bert")
    ensure_dir_exists(bert_output_dir)

    # Save metrics
    metrics_with_name = {
        "model": "bert",
        "test_loss": float(test_loss),
        **test_metrics,
    }

    metrics_path = os.path.join(results_dir, "bert_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_with_name, f, indent=2)
    logger.info("Saved BERT metrics to %s", metrics_path)

    # Save model (both as state dict and Hugging Face format)
    save_models = bool(train_cfg["save"].get("save_models", True))
    if save_models:
        # PyTorch state dict
        state_dict_path = os.path.join(models_dir, "bert_classifier.pt")
        overwrite = bool(train_cfg["save"].get("overwrite_existing", False))
        if not os.path.exists(state_dict_path) or overwrite:
            torch.save(model.state_dict(), state_dict_path)
            logger.info("Saved BERT state dict to %s", state_dict_path)

        # Hugging Face format (config + weights + tokenizer vocab etc.)
        model.save_pretrained(bert_output_dir)
        logger.info("Saved BERT model in Hugging Face format to %s", bert_output_dir)

    return metrics_with_name


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point when running this module as a script.
    """
    _ = train_and_evaluate_bert()


if __name__ == "__main__":
    main()
