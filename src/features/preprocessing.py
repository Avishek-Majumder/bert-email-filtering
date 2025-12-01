"""
Text preprocessing utilities for spam detection.

This module implements the preprocessing pipeline described in
"Harnessing BERT for Advanced Email Filtering in Cybersecurity":

- lowercasing
- punctuation removal
- number removal
- extra whitespace normalization
- tokenization
- stopword removal
- stemming / lemmatization (for non-BERT models)

We provide helpers that operate on individual strings as well as on
pandas Series. Configuration is driven by config/data.yaml, so the
pipeline can be tweaked without changing this code.
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Set, Dict, Any, Optional

import pandas as pd

from src.data.datasets import load_data_config, DEFAULT_DATA_CONFIG_PATH

# Optional NLTK-based support (stopwords, stemming, lemmatization).
# If NLTK is not installed, we fall back to simpler behavior.
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _NLTK_AVAILABLE = False

# Fallback stopwords (from scikit-learn) if NLTK is unavailable.
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_EN_STOPWORDS

    _SKLEARN_STOPWORDS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _SKLEARN_STOPWORDS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------------------------


def _clean_text_basic(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """
    Apply basic normalization to a raw text string.

    Parameters
    ----------
    text : str
        Raw input text.
    lowercase : bool
        Convert text to lowercase if True.
    remove_punctuation : bool
        Remove punctuation characters if True.
    remove_numbers : bool
        Remove numeric characters if True.
    strip_whitespace : bool
        Collapse multiple spaces and strip leading/trailing spaces.

    Returns
    -------
    str
        Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # Replace punctuation with space so we don't accidentally join words.
        punct_table = str.maketrans({ch: " " for ch in string.punctuation})
        text = text.translate(punct_table)

    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    if strip_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _tokenize_whitespace(text: str) -> List[str]:
    """
    Simple whitespace-based tokenizer.

    Parameters
    ----------
    text : str
        Text string (assumed to be pre-cleaned).

    Returns
    -------
    List[str]
        List of tokens.
    """
    if not text:
        return []
    return text.split()


def tokenize_text(text: str, method: str = "whitespace") -> List[str]:
    """
    Tokenize a text string using the specified method.

    Currently supported:
    - "whitespace": simple .split() on whitespace.

    Parameters
    ----------
    text : str
        Text string (already cleaned).
    method : str
        Tokenization method.

    Returns
    -------
    List[str]
        List of tokens.
    """
    method = (method or "whitespace").lower()
    if method == "whitespace":
        return _tokenize_whitespace(text)
    else:
        # Fallback to whitespace for unknown methods.
        return _tokenize_whitespace(text)


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------


def _get_stopword_set(language: str = "english") -> Set[str]:
    """
    Build a set of stopwords for the given language.

    We prefer NLTK stopwords if available, falling back to
    scikit-learn's English stopwords when necessary.

    Parameters
    ----------
    language : str
        Language name, e.g. "english".

    Returns
    -------
    Set[str]
        Set of stopwords.
    """
    lang = (language or "english").lower()

    # Try NLTK first.
    if _NLTK_AVAILABLE:
        try:
            # Ensure the stopwords corpus is available.
            # Users may need to call:
            #   nltk.download("stopwords")
            sw = set(nltk_stopwords.words(lang))
            return sw
        except Exception:  # pragma: no cover - runtime environment dependent
            pass

    # Fallback to sklearn English stopwords if available and language is English.
    if lang == "english" and _SKLEARN_STOPWORDS_AVAILABLE:
        return set(SKLEARN_EN_STOPWORDS)

    # As a last resort, return an empty set (i.e., no stopword removal).
    return set()


def remove_stopwords(tokens: Iterable[str], stopword_set: Set[str]) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Parameters
    ----------
    tokens : Iterable[str]
        Input tokens.
    stopword_set : Set[str]
        Set of words to remove.

    Returns
    -------
    List[str]
        Tokens with stopwords removed.
    """
    if not stopword_set:
        return list(tokens)
    return [t for t in tokens if t not in stopword_set]


# ---------------------------------------------------------------------------
# Stemming and lemmatization
# ---------------------------------------------------------------------------


def _build_stemmer(algorithm: str = "porter"):
    """
    Build a stemming object based on the chosen algorithm.

    Parameters
    ----------
    algorithm : str
        Name of the stemming algorithm: "porter" or "snowball".

    Returns
    -------
    object or None
        Stemmer object with a .stem(token) method, or None if not available.
    """
    if not _NLTK_AVAILABLE:
        return None

    algo = (algorithm or "porter").lower()
    try:
        if algo == "porter":
            return PorterStemmer()
        elif algo == "snowball":
            return SnowballStemmer("english")
        else:
            return PorterStemmer()
    except Exception:  # pragma: no cover - very unlikely
        return None


def _build_lemmatizer(model: str = "wordnet"):
    """
    Build a lemmatizer object.

    Parameters
    ----------
    model : str
        Lemmatizer model name; currently only "wordnet" is supported.

    Returns
    -------
    object or None
        Lemmatizer object with a .lemmatize(token) method, or None if not available.
    """
    if not _NLTK_AVAILABLE:
        return None

    try:
        # Users may need to call:
        #   nltk.download("wordnet")
        return WordNetLemmatizer()
    except Exception:  # pragma: no cover - very unlikely
        return None


def stem_tokens(tokens: Iterable[str], algorithm: str = "porter") -> List[str]:
    """
    Apply stemming to a list of tokens.

    Parameters
    ----------
    tokens : Iterable[str]
        Input tokens.
    algorithm : str
        Stemming algorithm ("porter" or "snowball").

    Returns
    -------
    List[str]
        Stemmed tokens. If no stemmer is available, returns tokens unchanged.
    """
    stemmer = _build_stemmer(algorithm)
    if stemmer is None:
        return list(tokens)
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: Iterable[str], model: str = "wordnet") -> List[str]:
    """
    Apply lemmatization to a list of tokens.

    Parameters
    ----------
    tokens : Iterable[str]
        Input tokens.
    model : str
        Lemmatizer model ("wordnet").

    Returns
    -------
    List[str]
        Lemmatized tokens. If no lemmatizer is available, returns tokens unchanged.
    """
    lemmatizer = _build_lemmatizer(model)
    if lemmatizer is None:
        return list(tokens)
    return [lemmatizer.lemmatize(t) for t in tokens]


# ---------------------------------------------------------------------------
# High-level preprocessing functions
# ---------------------------------------------------------------------------


def _get_preprocessing_cfg(
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Retrieve the 'preprocessing' section from the data configuration.

    Parameters
    ----------
    config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    Dict[str, Any]
        Preprocessing configuration dictionary.
    """
    cfg = load_data_config(config_path)
    return cfg["preprocessing"]


def preprocess_text_to_tokens(
    text: str,
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> List[str]:
    """
    Full preprocessing pipeline for non-BERT models, returning tokens.

    The pipeline is controlled via the 'preprocessing' section in
    config/data.yaml and typically includes:

    - lowercasing
    - punctuation removal
    - number removal
    - whitespace normalization
    - tokenization (whitespace)
    - stopword removal (if enabled)
    - stemming OR lemmatization (if enabled)

    If both stemming.enabled and lemmatization.enabled are True, we
    give priority to lemmatization.

    Parameters
    ----------
    text : str
        Raw input text.
    config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    List[str]
        Preprocessed tokens.
    """
    cfg = _get_preprocessing_cfg(config_path)

    # Basic cleaning
    text_clean = _clean_text_basic(
        text=text,
        lowercase=bool(cfg.get("lowercase", True)),
        remove_punctuation=bool(cfg.get("remove_punctuation", True)),
        remove_numbers=bool(cfg.get("remove_numbers", True)),
        strip_whitespace=bool(cfg.get("strip_whitespace", True)),
    )

    # Tokenization
    tokenize_cfg = cfg.get("tokenize", {}) or {}
    method = tokenize_cfg.get("method", "whitespace")
    tokens = tokenize_text(text_clean, method=method)

    if not tokens:
        return []

    # Stopwords
    sw_cfg = cfg.get("stopwords", {}) or {}
    if bool(sw_cfg.get("enabled", True)):
        language = sw_cfg.get("language", "english")
        sw_set = _get_stopword_set(language)
        tokens = remove_stopwords(tokens, sw_set)

    if not tokens:
        return []

    # Lemmatization vs stemming
    lemma_cfg = cfg.get("lemmatization", {}) or {}
    stem_cfg = cfg.get("stemming", {}) or {}

    if bool(lemma_cfg.get("enabled", False)):
        model = lemma_cfg.get("model", "wordnet")
        tokens = lemmatize_tokens(tokens, model=model)
    elif bool(stem_cfg.get("enabled", True)):
        algorithm = stem_cfg.get("algorithm", "porter")
        tokens = stem_tokens(tokens, algorithm=algorithm)

    return tokens


def preprocess_text_to_string(
    text: str,
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> str:
    """
    Preprocess a text string for non-BERT models and return a single
    cleaned string, suitable for TF–IDF vectorization.

    This function simply joins the tokens produced by
    `preprocess_text_to_tokens` with a single space.

    Parameters
    ----------
    text : str
        Raw input text.
    config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    str
        Preprocessed, tokenized, and normalized text.
    """
    tokens = preprocess_text_to_tokens(text, config_path=config_path)
    return " ".join(tokens)


def preprocess_series_to_string(
    series: pd.Series,
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> pd.Series:
    """
    Apply the non-BERT preprocessing pipeline to a pandas Series of text
    and return a new Series of processed strings.

    Parameters
    ----------
    series : pd.Series
        Series of raw text values.
    config_path : str
        Path to the data YAML configuration.

    Returns
    -------
    pd.Series
        Series of preprocessed text strings.
    """
    return series.astype(str).apply(
        lambda x: preprocess_text_to_string(x, config_path=config_path)
    )


def preprocess_text_for_bert(
    text: str,
    config_path: str = DEFAULT_DATA_CONFIG_PATH,
) -> str:
    """
    Lightweight preprocessing for BERT-based models.

    For BERT, we intentionally apply a minimal transformation to avoid
    disrupting the pretrained tokenizer's vocabulary and casing too much.

    We typically:
    - lowercase (for 'bert-base-uncased')
    - normalize whitespace (strip and collapse)

    We DO NOT:
    - remove punctuation
    - remove numbers
    - remove stopwords
    - apply stemming / lemmatization

    Parameters
    ----------
    text : str
        Raw input text.
    config_path : str
        Path to the data YAML configuration (for consistency; currently
        only used to respect lowercase/whitespace flags).

    Returns
    -------
    str
        Lightly normalized text string suitable for BERT tokenization.
    """
    cfg = _get_preprocessing_cfg(config_path)
    lowercase = bool(cfg.get("lowercase", True))

    if not isinstance(text, str):
        text = str(text)

    if lowercase:
        text = text.lower()

    # Only normalize whitespace here.
    text = re.sub(r"\s+", " ", text).strip()
    return text
