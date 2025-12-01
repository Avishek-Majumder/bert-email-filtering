"""
Evaluation metrics for spam detection experiments.

This module centralizes the computation of core classification metrics
used in our work "Harnessing BERT for Advanced Email Filtering in Cybersecurity":

- accuracy
- precision
- recall
- F1-score
- confusion matrix

The helper functions here are used by:
- src.training.train_ml
- src.training.train_dl
- src.training.train_bert
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


ArrayLike = Union[Sequence[int], np.ndarray]


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
    labels: Optional[Sequence[int]] = None,
    output_confusion_matrix: bool = True,
) -> Dict[str, Any]:
    """
    Compute standard classification metrics for a predicted label set.

    Parameters
    ----------
    y_true : ArrayLike
        Ground-truth labels (e.g., 0 for ham, 1 for spam).
    y_pred : ArrayLike
        Predicted labels, same shape as y_true.
    average : str
        Averaging mode for precision/recall/F1 when there are multiple classes.
        Common options:
            - "binary"   (default for two-class problems: ham vs spam)
            - "macro"
            - "micro"
            - "weighted"
    labels : Optional[Sequence[int]]
        Optional list of label indices to include in the confusion matrix.
        If None, labels are inferred from y_true and y_pred.
    output_confusion_matrix : bool
        If True, also compute and include the confusion matrix.

    Returns
    -------
    Dict[str, Any]
        Dictionary with at least the following keys:
            - "accuracy"
            - "precision"
            - "recall"
            - "f1"

        If output_confusion_matrix is True, also includes:
            - "confusion_matrix": 2D list representation of the confusion matrix.
    """
    # Convert to numpy arrays for consistency
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Accuracy
    acc = accuracy_score(y_true_arr, y_pred_arr)

    # Precision, recall, F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average=average,
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    if output_confusion_matrix:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
        # Convert numpy array to plain list for JSON/CSV friendliness
        metrics["confusion_matrix"] = cm.tolist()

    return metrics
