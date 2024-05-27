"""
This module provides a function to evaluate models on test data.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def evaluate_model(model: Pipeline, data_test: pd.DataFrame, labels_test: pd.Series, vars_selected: list):
    """
    Evaluates a trained model pipeline on the test data and returns performance metrics.

    Args:
        model: A trained model pipeline that includes preprocessing and the classifier.
        data_test: Test data (DataFrame) on which to evaluate the model.
        labels_test: True labels for the test data.
        vars_selected: Selected genes for classification. Can be DEGs or a custom list.

    Returns:
        A dictionary containing key performance metrics: accuracy, precision, recall, F1 score, and confusion matrix.
    """
    data_test = data_test[vars_selected]

    y_pred_test = model.predict(data_test)
    y_true_test, unique_labels = pd.factorize(labels_test, sort=True)

    return {
        "accuracy": accuracy_score(y_true_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_true_test, y_pred_test),
        "precision": precision_score(y_true_test, y_pred_test, average="macro"),
        "recall": recall_score(y_true_test, y_pred_test, average="macro"),
        "f1_score": f1_score(y_true_test, y_pred_test, average="macro"),
        "y_pred_test": y_pred_test,
        "unique_labels": unique_labels
    }
