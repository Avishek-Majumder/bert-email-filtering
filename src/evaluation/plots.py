"""
Plotting utilities for spam detection experiments.

This module provides convenient helpers to visualize:

- Model-level metrics (e.g., F1 scores across ML, DL, BERT models)
- Confusion matrices for individual models

These helpers are meant for analysis and reporting of our work:

    "Harnessing BERT for Advanced Email Filtering in Cybersecurity"
    IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11058531

The functions are designed to be light-weight and optional:
if matplotlib is not installed, an informative ImportError is raised.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:  # pragma: no cover
    # Fallback: allow import of this module even without matplotlib,
    # but plotting functions will raise an informative error.
    plt = None  # type: ignore
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_matplotlib() -> None:
    """
    Ensure matplotlib is available, otherwise raise an informative error.
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with `pip install matplotlib`."
        )


# ---------------------------------------------------------------------------
# Bar plots for model metrics
# ---------------------------------------------------------------------------


def plot_metric_bar(
    results_df: pd.DataFrame,
    metric: str = "f1",
    top_k: Optional[int] = None,
    category: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    rotate_xticks: int = 45,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a bar chart of a chosen metric (e.g., F1) across models.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame as produced by `aggregate_all_results`, containing
        columns like ["model", "category", "accuracy", "precision",
        "recall", "f1"].
    metric : str
        Metric column to plot (default: "f1").
    top_k : Optional[int]
        If provided, only the top_k models (sorted by metric descending)
        are shown.
    category : Optional[str]
        If provided, filter the DataFrame to this category
        (e.g., "ml", "dl", "bert").
    figsize : Tuple[float, float]
        Figure size in inches.
    rotate_xticks : int
        Rotation angle for x-axis tick labels.
    title : Optional[str]
        Title for the plot. If None, a default is constructed.
    out_path : Optional[str]
        If provided, save the figure to this path (e.g., PNG).
    show : bool
        If True, call plt.show(). If False, just return the figure/axes.

    Returns
    -------
    (fig, ax)
        Matplotlib Figure and Axes objects.
    """
    _ensure_matplotlib()

    df = results_df.copy()

    if df.empty:
        raise ValueError("results_df is empty; nothing to plot.")

    if category is not None:
        df = df[df.get("category", "").str.lower() == category.lower()]

    if df.empty:
        raise ValueError(
            f"No rows to plot after filtering for category={category!r} "
            f"and metric={metric!r}."
        )

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame columns. "
            f"Available columns: {list(df.columns)}"
        )

    # Sort descending by the metric
    df_sorted = df.sort_values(metric, ascending=False, na_position="last")

    if top_k is not None and top_k > 0:
        df_sorted = df_sorted.head(top_k)

    # Ensure numeric
    scores = pd.to_numeric(df_sorted[metric], errors="coerce")
    models = df_sorted["model"].astype(str)

    # Build bar plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(models, scores)

    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Model")

    if title is None:
        if category is not None:
            ax.set_title(f"Top models ({category}) by {metric.upper()}")
        else:
            ax.set_title(f"Model comparison by {metric.upper()}")
    else:
        ax.set_title(title)

    ax.set_ylim(0.0, float(scores.max()) * 1.05 if not scores.isna().all() else 1.0)

    # Rotate xticks for readability
    ax.set_xticklabels(models, rotation=rotate_xticks, ha="right")

    # Annotate bars with metric values
    for i, v in enumerate(scores):
        if np.isnan(v):
            continue
        ax.text(
            i,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_f1_scores(
    results_df: pd.DataFrame,
    top_k: Optional[int] = None,
    category: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    rotate_xticks: int = 45,
    out_path: Optional[str] = None,
    show: bool = True,
):
    """
    Convenience wrapper for `plot_metric_bar` specialized to F1-scores.

    Parameters
    ----------
    results_df : pd.DataFrame
        Aggregated results DataFrame.
    top_k : Optional[int]
        Show only the top_k models by F1-score.
    category : Optional[str]
        Filter by category: "ml", "dl", or "bert".
    figsize : Tuple[float, float]
        Figure size in inches.
    rotate_xticks : int
        Rotation angle for x-axis tick labels.
    out_path : Optional[str]
        If provided, save the figure to this path.
    show : bool
        If True, show the plot; otherwise, just return fig/ax.

    Returns
    -------
    (fig, ax)
        Matplotlib Figure and Axes objects.
    """
    return plot_metric_bar(
        results_df=results_df,
        metric="f1",
        top_k=top_k,
        category=category,
        figsize=figsize,
        rotate_xticks=rotate_xticks,
        title="Model comparison by F1-score",
        out_path=out_path,
        show=show,
    )


# ---------------------------------------------------------------------------
# Confusion matrix plots
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str] = ("ham", "spam"),
    normalize: bool = False,
    figsize: Tuple[float, float] = (6.0, 5.0),
    title: Optional[str] = None,
    out_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes), where
        rows correspond to true labels and columns to predicted labels.
    labels : Sequence[str]
        Class labels in the order corresponding to the confusion matrix.
    normalize : bool
        If True, normalize each row to sum to 1.0.
    figsize : Tuple[float, float]
        Figure size in inches.
    title : Optional[str]
        Plot title. If None, a default is chosen based on `normalize`.
    out_path : Optional[str]
        If provided, save the figure to this path.
    show : bool
        If True, show the plot; otherwise, just return fig/ax.

    Returns
    -------
    (fig, ax)
        Matplotlib Figure and Axes objects.
    """
    _ensure_matplotlib()

    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be a square 2D array.")

    n_classes = cm.shape[0]
    if len(labels) != n_classes:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match CM size ({n_classes})."
        )

    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm, row_sums, where=row_sums != 0)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    # Tick labels
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    if title is None:
        title = "Normalized confusion matrix" if normalize else "Confusion matrix"
    ax.set_title(title)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell
    thresh = cm_display.max() / 2.0 if cm_display.size > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_display[i, j]
            text_color = "white" if value > thresh else "black"
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color=text_color,
            )

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
