"""
This module contains functions for performing Logistic Regression classification and related utility functions.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, BaseCrossValidator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def logistic_regression(data: pd.DataFrame, labels: pd.Series, vars_selected: list,
                        cv_strategy: BaseCrossValidator = None) -> Pipeline:
    """
    Conducts Logistic Regression classification.

    Args:
        data: The feature matrix with features in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Selected features for classification.
        cv_strategy: CV strategy to use. If None, defaults to RepeatedStratifiedKFold(n_splits=10, n_repeats=3).

    Returns:
        A trained LogisticRegression model pipeline.
    """
    label_codes, unique_labels = pd.factorize(labels, sort=True)
    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").fillna(0)
    data = data[vars_selected]

    if not cv_strategy:
        logger.info("Running Repeated Stratified K-Fold Cross-Validation with 10 folds")
        cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    param_grid = {
        "C": np.logspace(-4, 4, 20),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro")
    }

    scaler = StandardScaler()
    grid_search = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=cv_strategy, scoring=scoring,
                               refit="accuracy")
    pipeline = make_pipeline(
        scaler,
        grid_search
    )
    pipeline.fit(data, label_codes)
    logger.info("Optimal parameters: %s", grid_search.best_params_)

    return make_pipeline(scaler, grid_search.best_estimator_)
