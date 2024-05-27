"""
This module contains functions for performing Random Forest classification and related utility functions.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def rf(data: pd.DataFrame, labels: pd.Series, vars_selected: list,
       cv_strategy: BaseCrossValidator = None) -> Pipeline:
    """
    Conducts Random Forest classification.

    Args:
        data: The expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Selected genes for classification. Can be DEGs or a custom list.
        cv_strategy: CV strategy to use. If None, defaults to RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    Returns:
        A trained RandomForestClassifier model pipeline.
    """
    label_codes, unique_labels = pd.factorize(labels, sort=True)
    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").fillna(0)
    data = data[vars_selected]

    if not cv_strategy:
        logger.info("Running Repeated Stratified K-Fold Cross-Validation with 10 folds")
        cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    param_grid = {"n_estimators": [30, 75, 100], "max_features": ["auto", "sqrt", "log2"], "max_depth": [4, 6, 8, 10]}
    scoring = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score, average="macro"),
               "recall": make_scorer(recall_score, average="macro"),
               "f1_score": make_scorer(f1_score, average="macro")}

    scaler = StandardScaler()
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=cv_strategy, scoring=scoring, refit="accuracy")
    pipeline = make_pipeline(
        scaler,
        grid_search
    )
    pipeline.fit(data, label_codes)
    logger.info("Best parameters: %s", grid_search.best_params_)

    return make_pipeline(scaler, grid_search.best_estimator_)
