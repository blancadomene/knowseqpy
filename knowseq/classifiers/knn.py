"""
This module contains functions for performing k-Nearest Neighbors (k-NN) classification and related utility functions.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, LeaveOneOut, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def knn(data: pd.DataFrame, labels, vars_selected, num_fold: int = 10, loocv: bool = False):
    """
    Conducts k-NN classification.

    Args:
        data (DataFrame or ndarray): The expression matrix with genes in columns and samples in rows.
        labels (array-like): Labels for each sample.
        vars_selected (list): Selected genes for classification. Can be DEGs or a custom list.
        num_fold: Number of folds for cross-validation. Default is 10.
        loocv: If True, use Leave-One-Out cross-validation. Otherwise, use KFold. Defaults to False.

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
        Using Leave-One-Out Cross-Validation (LOOCV) can be resource-intensive, particularly for large datasets.
    """
    label_codes, unique_labels = pd.factorize(labels)

    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").fillna(0)
    data = data[vars_selected]
    scaled_data = StandardScaler().fit_transform(data)

    if loocv:
        logger.info("Running Leave One Out Cross-Validation")
        cv_strategy = LeaveOneOut()
    else:
        logger.info("Running Repeated Stratified K-Fold Cross-Validation with %s folds", num_fold)
        cv_strategy = RepeatedStratifiedKFold(n_splits=num_fold, n_repeats=3)

    results = _knn_model_evaluation(scaled_data, label_codes, cv_strategy)

    logger.info("Classification completed successfully")
    results["unique_labels"] = unique_labels
    return results


def _knn_model_evaluation(data: pd.DataFrame, label_codes, cv: BaseCrossValidator):
    """
    Tunes the best k value for the k-NN classifier using grid search with the provided cross-validation strategy

    Args:
        data: The expression matrix.
        cv (obj): Cross-validation strategy.
        label_codes (array-like): label_codes for each sample.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    param_grid = {"n_neighbors": range(1, int(np.sqrt(len(data))))}
    scoring = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score, average="macro"),
               "recall": make_scorer(recall_score, average="macro"),
               "f1_score": make_scorer(f1_score, average="macro")}

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring=scoring, refit="accuracy")
    grid_search.fit(data, label_codes)

    logger.info("Optimal k: %s", grid_search.best_estimator_.n_neighbors)
    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_
    y_pred = grid_search.best_estimator_.predict(data)

    conf_mat = confusion_matrix(label_codes, y_pred)

    return {
        "model": grid_search.best_estimator_,
        "confusion_matrix": conf_mat,
        "accuracy": grid_search.best_score_,
        "f1_score": cv_results["mean_test_f1_score"][best_index],
        "specificity": _calculate_specificity(conf_mat),
        "precision": cv_results["mean_test_precision"][best_index],
        "sensitivity": cv_results["mean_test_recall"][best_index],
        "y_pred": y_pred
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
