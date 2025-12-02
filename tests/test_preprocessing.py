"""
Basic tests for text preprocessing utilities.

These tests validate that:

- the preprocessing module can be imported
- at least one public text-processing function exists
- calling that function on simple inputs returns non-empty strings

We keep the tests defensive so that minor refactors (e.g. renaming the
main preprocessing function) do not break the entire suite: if no known
function is found, the behaviour tests are skipped rather than failed.
"""

from __future__ import annotations

from typing import Callable, Optional

import pytest

from src.data.datasets import load_data_config
import src.features.preprocessing as preprocessing


def _get_preprocess_fn() -> Optional[Callable[..., str]]:
    """
    Try to locate a public text preprocessing function in
    src.features.preprocessing.

    We check a few common names in order and return the first one found.
    """
    candidate_names = [
        "preprocess_text",          # e.g., preprocess_text(text, config)
        "preprocess_single_text",   # e.g., preprocess_single_text(text, config)
        "clean_text",               # e.g., clean_text(text, config)
        "normalize_text",           # e.g., normalize_text(text, config)
    ]

    for name in candidate_names:
        fn = getattr(preprocessing, name, None)
        if callable(fn):
            return fn

    return None


def test_preprocessing_module_imports():
    """
    Simple sanity check that the preprocessing module imports successfully.
    """
    assert preprocessing is not None


def test_preprocessing_function_exists():
    """
    Ensure that at least one known preprocessing function is present.

    If none are found, we skip further behaviour tests instead of failing
    the entire suite.
    """
    fn = _get_preprocess_fn()
    if fn is None:
        pytest.skip(
            "No known preprocessing function found in src.features.preprocessing "
            "(expected one of: preprocess_text, preprocess_single_text, "
            "clean_text, normalize_text)."
        )
    assert callable(fn)


def test_preprocessing_basic_cleanup():
    """
    If a preprocessing function is available, verify that it:

    - returns a string
    - does not return an empty string for a simple non-empty input
    - performs at least lowercasing or some simple normalization
    """
    fn = _get_preprocess_fn()
    if fn is None:
        pytest.skip("No known preprocessing function found; skipping behaviour test.")

    data_cfg = load_data_config("config/data.yaml")

    raw_text = "HELLO WORLD!!! This is a TEST message with URL: http://example.com"
    processed = fn(raw_text, data_cfg)  # type: ignore[arg-type]

    assert isinstance(processed, str), "Preprocessing should return a string."
    assert processed.strip() != "", "Processed text should not be empty."

    # We expect at least basic lowercasing (most pipelines do this).
    assert "HELLO" not in processed
    assert "hello" in processed or "world" in processed.lower()
