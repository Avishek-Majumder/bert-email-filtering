"""
Classical machine learning model builders for spam detection.

This module provides helper functions to construct all ML baselines
used in our work:

- Random Forest (RF)
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- XGBoost (XGB)
- Gradient Boosting (GB)
- Naive Bayes (NB)
- K-Nearest Neighbors (KNN)

Hyperparameters are read from config/ml.yaml so they can be tuned
without modifying code. The training pipeline (fit/predict, scaling,
metrics) is implemented in src/training/train_ml.py.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC

# XGBoost is an external dependency; we import it lazily and handle
# the case where it is unavailable.
try:
    from xgboost import XGBClassifier  # type: ignore

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    _XGB_AVAILABLE = False


DEFAULT_ML_CONFIG_PATH = "config/ml.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_ml_config(config_path: str = DEFAULT_ML_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and return the ML configuration dictionary.

    Parameters
    ----------
    config_path : str
        Path to the ML YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration with "general" and "ml_models" sections.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty or cannot be parsed.
    KeyError
        If required sections are missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ML config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"ML config file is empty or invalid: {config_path}")

    for section in ("general", "ml_models"):
        if section not in cfg:
            raise KeyError(f'Missing "{section}" section in ML config: {config_path}')

    return cfg


# ---------------------------------------------------------------------------
# Model builder helpers
# ---------------------------------------------------------------------------


def _class_weight_or_none(use_balanced: bool) -> Any:
    """
    Helper to decide class_weight parameter value for supported models.

    Parameters
    ----------
    use_balanced : bool
        If True, returns "balanced", otherwise None.

    Returns
    -------
    Any
        "balanced" or None.
    """
    return "balanced" if use_balanced else None


def build_random_forest(cfg: Dict[str, Any], use_balanced: bool) -> RandomForestClassifier:
    mcfg = cfg["ml_models"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=int(mcfg.get("n_estimators", 200)),
        criterion=str(mcfg.get("criterion", "gini")),
        max_depth=mcfg.get("max_depth", None),
        min_samples_split=int(mcfg.get("min_samples_split", 2)),
        min_samples_leaf=int(mcfg.get("min_samples_leaf", 1)),
        max_features=mcfg.get("max_features", "sqrt"),
        n_jobs=int(mcfg.get("n_jobs", -1)),
        random_state=int(cfg["general"].get("random_state", 42)),
        class_weight=_class_weight_or_none(use_balanced),
    )


def build_logistic_regression(cfg: Dict[str, Any], use_balanced: bool) -> LogisticRegression:
    mcfg = cfg["ml_models"]["logistic_regression"]
    return LogisticRegression(
        penalty=str(mcfg.get("penalty", "l2")),
        C=float(mcfg.get("C", 1.0)),
        solver=str(mcfg.get("solver", "liblinear")),
        max_iter=int(mcfg.get("max_iter", 1000)),
        fit_intercept=bool(mcfg.get("fit_intercept", True)),
        random_state=int(cfg["general"].get("random_state", 42)),
        class_weight=_class_weight_or_none(use_balanced),
    )


def build_svm(cfg: Dict[str, Any], use_balanced: bool) -> SVC:
    mcfg = cfg["ml_models"]["svm"]
    return SVC(
        kernel=str(mcfg.get("kernel", "linear")),
        C=float(mcfg.get("C", 1.0)),
        gamma=mcfg.get("gamma", "scale"),
        probability=bool(mcfg.get("probability", True)),
        class_weight=_class_weight_or_none(use_balanced),
        random_state=int(cfg["general"].get("random_state", 42)),
    )


def build_xgboost(cfg: Dict[str, Any]) -> XGBClassifier:
    if not _XGB_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Please install it or remove XGBoost "
            "from the ML model list."
        )

    mcfg = cfg["ml_models"]["xgboost"]
    return XGBClassifier(
        n_estimators=int(mcfg.get("n_estimators", 200)),
        max_depth=int(mcfg.get("max_depth", 6)),
        learning_rate=float(mcfg.get("learning_rate", 0.1)),
        subsample=float(mcfg.get("subsample", 0.8)),
        colsample_bytree=float(mcfg.get("colsample_bytree", 0.8)),
        objective=str(mcfg.get("objective", "binary:logistic")),
        eval_metric=str(mcfg.get("eval_metric", "logloss")),
        reg_lambda=float(mcfg.get("reg_lambda", 1.0)),
        reg_alpha=float(mcfg.get("reg_alpha", 0.0)),
        n_jobs=int(mcfg.get("n_jobs", -1)),
        random_state=int(cfg["general"].get("random_state", 42)),
        use_label_encoder=False,
    )


def build_gradient_boosting(cfg: Dict[str, Any]) -> GradientBoostingClassifier:
    mcfg = cfg["ml_models"]["gradient_boosting"]
    return GradientBoostingClassifier(
        n_estimators=int(mcfg.get("n_estimators", 200)),
        learning_rate=float(mcfg.get("learning_rate", 0.1)),
        max_depth=int(mcfg.get("max_depth", 3)),
        subsample=float(mcfg.get("subsample", 1.0)),
        min_samples_split=int(mcfg.get("min_samples_split", 2)),
        min_samples_leaf=int(mcfg.get("min_samples_leaf", 1)),
        random_state=int(cfg["general"].get("random_state", 42)),
    )


def build_naive_bayes(cfg: Dict[str, Any]):
    """
    Build a Naive Bayes classifier instance.

    Depending on config.ml_models.naive_bayes.type, we return either
    MultinomialNB or BernoulliNB. For TF–IDF text features, MultinomialNB
    is a common choice.
    """
    mcfg = cfg["ml_models"]["naive_bayes"]
    nb_type = str(mcfg.get("type", "multinomial")).lower()
    alpha = float(mcfg.get("alpha", 1.0))
    fit_prior = bool(mcfg.get("fit_prior", True))

    if nb_type == "bernoulli":
        return BernoulliNB(alpha=alpha, fit_prior=fit_prior)
    else:
        return MultinomialNB(alpha=alpha, fit_prior=fit_prior)


def build_knn(cfg: Dict[str, Any]) -> KNeighborsClassifier:
    mcfg = cfg["ml_models"]["knn"]
    return KNeighborsClassifier(
        n_neighbors=int(mcfg.get("n_neighbors", 5)),
        weights=str(mcfg.get("weights", "distance")),
        metric=str(mcfg.get("metric", "minkowski")),
        p=int(mcfg.get("p", 2)),
        n_jobs=int(mcfg.get("n_jobs", -1)),
    )


# ---------------------------------------------------------------------------
# Public factory: build all models
# ---------------------------------------------------------------------------


def build_all_ml_models(
    config_path: str = DEFAULT_ML_CONFIG_PATH,
) -> Dict[str, object]:
    """
    Build all configured ML models and return them in a dictionary.

    Parameters
    ----------
    config_path : str
        Path to the ML YAML configuration.

    Returns
    -------
    Dict[str, object]
        Dictionary mapping model names to constructed sklearn/xgboost estimators.
        Keys:
            - "random_forest"
            - "logistic_regression"
            - "svm"
            - "xgboost"
            - "gradient_boosting"
            - "naive_bayes"
            - "knn"
    """
    cfg = load_ml_config(config_path)
    general_cfg = cfg["general"]
    use_balanced = bool(general_cfg.get("use_class_weight_balanced", True))

    models: Dict[str, object] = {}

    models["random_forest"] = build_random_forest(cfg, use_balanced)
    models["logistic_regression"] = build_logistic_regression(cfg, use_balanced)
    models["svm"] = build_svm(cfg, use_balanced)

    # XGBoost is optional; skip if not available.
    if _XGB_AVAILABLE:
        models["xgboost"] = build_xgboost(cfg)

    models["gradient_boosting"] = build_gradient_boosting(cfg)
    models["naive_bayes"] = build_naive_bayes(cfg)
    models["knn"] = build_knn(cfg)

    return models
