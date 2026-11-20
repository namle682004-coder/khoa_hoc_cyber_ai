# src/lab/model.py

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier


class ModelError(Exception):
    """Custom exception for model building errors."""
    pass


AVAILABLE_MODELS = {
    "lr": LogisticRegression,
    "rf": RandomForestClassifier,
    "gbm": GradientBoostingClassifier,
    "xgb": XGBClassifier,
}


# ---------------- Default Parameters ----------------
DEFAULT_PARAMS = {

    "lr": dict(
        max_iter=3000,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    ),

    "rf": dict(
        n_estimators=600,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "gbm": dict(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    ),

    "xgb": dict(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
}


def build_model(
    model_name: str = "xgb",
    params: dict | None = None
) -> ClassifierMixin:
    """
    Build and return a classification model.

    Parameters
    ----------
    model_name : str
        Model name ["lr", "rf", "gbm", "xgb"]

    params : dict | None
        Optional parameters (e.g. from hyperparameter tuning)

    Returns
    -------
    ClassifierMixin
        sklearn-compatible classifier
    """

    model_name = model_name.lower()

    if model_name not in AVAILABLE_MODELS:
        raise ModelError(
            f"Unknown model name '{model_name}'. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )

    # load default params
    model_params = DEFAULT_PARAMS[model_name].copy()

    # override with tuned params
    if params:
        model_params.update(params)

    model_class = AVAILABLE_MODELS[model_name]

    try:
        model = model_class(**model_params)
    except Exception as e:
        raise ModelError(f"Failed to build {model_name} model: {e}")

    return model