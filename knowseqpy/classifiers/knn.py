"""
This module contains functions for performing k-Nearest Neighbors (k-NN) classification and related utility functions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def knn(data: pd.DataFrame, labels: pd.Series, vars_selected: list, cv_strategy: BaseCrossValidator = None) -> dict:
    """
    Conducts k-NN classification.

    Args:
        data: The expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Selected genes for classification. Can be DEGs or a custom list.
        cv_strategy: CV strategy to use. If None, defaults to RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    Returns:
        A dictionary containing the following key metrics:
        - model: The trained KNeighborsClassifier model.
        - confusion_matrix: Array, representing the confusion matrix, showing true and false predictions for each class.
        - accuracy: Float, the mean accuracy of the model on the given test data and labels.
        - specificity: Float, the specificity of the model. Measures the proportion of true negatives identified.
        - sensitivity: Float, also known as recall. Measures the proportion of true positives identified.
        - precision: Float, the precision of the model. Represents the ratio of true positives to all positives.
        - f1_score: Float, the F1 score of the model. A balance between precision and recall.
        - y_pred: Array, the predictions made by the model on the dataset.

    Optimization Note:
        Using Leave-One-Out Cross-Validation (loocv) can be resource-intensive, particularly for large datasets.
    """
    label_codes, unique_labels = pd.factorize(labels)
    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").fillna(0)
    data = data[vars_selected]
    scaled_data = StandardScaler().fit_transform(data)

    if not cv_strategy:
        logger.info("Running Repeated Stratified K-Fold Cross-Validation with 10 folds")
        cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    param_grid = {"n_neighbors": range(1, int(np.sqrt(len(scaled_data))))}
    scoring = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score, average="macro"),
               "recall": make_scorer(recall_score, average="macro"),
               "f1_score": make_scorer(f1_score, average="macro")}

    scaler = StandardScaler()
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv_strategy, scoring=scoring, refit="accuracy")
    pipeline = make_pipeline(
        scaler,
        grid_search
    )
    pipeline.fit(scaled_data, label_codes)

    logger.info("Optimal k: %s", grid_search.best_estimator_.n_neighbors)
    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_
    y_pred = grid_search.best_estimator_.predict(scaled_data)

    conf_mat = confusion_matrix(label_codes, y_pred)

    return {
        "model": make_pipeline(scaler, grid_search.best_estimator_),
        "confusion_matrix": conf_mat,
        "accuracy": grid_search.best_score_,
        "f1_score": cv_results["mean_test_f1_score"][best_index],
        "specificity": _calculate_specificity(conf_mat),
        "precision": cv_results["mean_test_precision"][best_index],
        "sensitivity": cv_results["mean_test_recall"][best_index],
        "y_pred": y_pred,
        "unique_labels": unique_labels
    }


def _calculate_specificity(conf_matrix: np.array) -> float:
    """
    Calculates specificity for each class in a binary or multiclass classification and returns the average.

    Args:
        conf_matrix: The confusion matrix of the model.

    Returns:
        The average specificity across all classes.
    """
    class_specificities = []
    for i, row in enumerate(conf_matrix):
        true_negatives = sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))
        false_positives = sum(np.delete(row, i))
        total_negatives = true_negatives + false_positives
        class_specificity = true_negatives / total_negatives if total_negatives > 0 else 0
        class_specificities.append(class_specificity)

    return np.mean(class_specificities)
