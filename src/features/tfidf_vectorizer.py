"""
TF–IDF feature extraction utilities for classical ML models.

This module provides helpers to:
- preprocess raw text using the non-BERT pipeline
- fit a scikit-learn TfidfVectorizer on the training data
- persist and reload the fitted vectorizer
- transform new text data into TF–IDF feature matrices

The configuration for preprocessing is shared with the rest of the
project via config/data.yaml. Persisted vectorizers are stored under
the artifacts directory defined in config/train.yaml.
"""

from __future__ import annotations

import os
from typing import Tuple, Iterable, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.datasets import DEFAULT_DATA_CONFIG_PATH, load_data_config
from src.features.preprocessing import preprocess_series_to_string
from src.utils.training_utils import load_train_config, ensure_dir_exists


DEFAULT_VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"


def _build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Construct a TfidfVectorizer instance with sensible defaults for
    short text messages (SMS/email).

    Returns
    -------
    TfidfVectorizer
        Unfitted TF–IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(
        # We already perform tokenization and stopword removal in the
        # preprocessing pipeline, so we treat input as pre-tokenized
        # strings and use a simple tokenizer.
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,  # already handled in preprocessing
        ngram_range=(1, 2),  # unigrams + bigrams often work well for spam
        min_df=1,
        max_df=1.0,
        max_features=None,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    return vectorizer


def fit_tfidf_from_series(
    train_texts: pd.Series,
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    artifacts_dir: Optional[str] = None,
    save: bool = True,
    filename: str = DEFAULT_VECTORIZER_FILENAME,
) -> TfidfVectorizer:
    """
    Fit a TF–IDF vectorizer from a pandas Series of raw training texts.

    This function:
    - applies the non-BERT preprocessing pipeline to the texts
    - fits a TfidfVectorizer on the processed strings
    - optionally saves the fitted vectorizer to disk

    Parameters
    ----------
    train_texts : pd.Series
        Series of raw training texts.
    data_config_path : str
        Path to the data YAML configuration (for preprocessing).
    artifacts_dir : Optional[str]
        Directory where the vectorizer should be saved. If None, this
        will be determined from config/train.yaml.
    save : bool
        Whether to persist the fitted vectorizer to disk.
    filename : str
        File name for the saved vectorizer.

    Returns
    -------
    TfidfVectorizer
        Fitted TF–IDF vectorizer.
    """
    # Preprocess the training texts.
    processed = preprocess_series_to_string(
        train_texts, config_path=data_config_path
    )

    vectorizer = _build_tfidf_vectorizer()
    vectorizer.fit(processed.values)

    if save:
        if artifacts_dir is None:
            train_cfg = load_train_config()
            artifacts_dir = train_cfg["paths"]["artifacts_dir"]
        ensure_dir_exists(artifacts_dir)
        path = os.path.join(artifacts_dir, filename)
        joblib.dump(vectorizer, path)

    return vectorizer


def load_tfidf_vectorizer(
    artifacts_dir: Optional[str] = None,
    filename: str = DEFAULT_VECTORIZER_FILENAME,
) -> TfidfVectorizer:
    """
    Load a previously saved TF–IDF vectorizer from disk.

    Parameters
    ----------
    artifacts_dir : Optional[str]
        Directory where the vectorizer is stored. If None, this will be
        determined from config/train.yaml.
    filename : str
        File name of the saved vectorizer.

    Returns
    -------
    TfidfVectorizer
        Loaded TF–IDF vectorizer.

    Raises
    ------
    FileNotFoundError
        If the vectorizer file does not exist.
    """
    if artifacts_dir is None:
        train_cfg = load_train_config()
        artifacts_dir = train_cfg["paths"]["artifacts_dir"]

    path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"TF–IDF vectorizer not found at: {path}")

    vectorizer: TfidfVectorizer = joblib.load(path)
    return vectorizer


def transform_texts_to_tfidf(
    texts: Iterable[str],
    vectorizer: TfidfVectorizer,
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> np.ndarray:
    """
    Transform an iterable of raw texts into a TF–IDF feature matrix
    using a fitted vectorizer.

    Parameters
    ----------
    texts : Iterable[str]
        Iterable of raw text strings.
    vectorizer : TfidfVectorizer
        Fitted TF–IDF vectorizer.
    data_config_path : str
        Path to the data YAML configuration (for preprocessing).

    Returns
    -------
    np.ndarray
        TF–IDF feature matrix of shape (n_samples, n_features).
    """
    series = pd.Series(list(texts), dtype=str)
    processed = preprocess_series_to_string(series, config_path=data_config_path)
    features = vectorizer.transform(processed.values)
    return features


def fit_transform_tfidf_from_series(
    train_texts: pd.Series,
    data_config_path: str = DEFAULT_DATA_CONFIG_PATH,
    artifacts_dir: Optional[str] = None,
    save: bool = True,
    filename: str = DEFAULT_VECTORIZER_FILENAME,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Convenience function that:
    - fits a TF–IDF vectorizer on the training texts
    - transforms the same training texts into TF–IDF features
    - optionally saves the fitted vectorizer

    Parameters
    ----------
    train_texts : pd.Series
        Series of raw training texts.
    data_config_path : str
        Path to the data YAML configuration (for preprocessing).
    artifacts_dir : Optional[str]
        Directory where the vectorizer should be saved.
    save : bool
        Whether to persist the fitted vectorizer to disk.
    filename : str
        File name for the saved vectorizer.

    Returns
    -------
    Tuple[np.ndarray, TfidfVectorizer]
        A tuple containing the TF–IDF feature matrix and the fitted vectorizer.
    """
    vectorizer = fit_tfidf_from_series(
        train_texts=train_texts,
        data_config_path=data_config_path,
        artifacts_dir=artifacts_dir,
        save=save,
        filename=filename,
    )
    features = transform_texts_to_tfidf(
        texts=train_texts.values,
        vectorizer=vectorizer,
        data_config_path=data_config_path,
    )
    return features, vectorizer
