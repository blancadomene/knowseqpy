import logging

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# TODO: change loocv variable name?
#   Parameter names, descriptions, etc
def knn_classifier(data, labels, vars_selected, num_fold=10, loocv=False):
    """
    Conduct k-NN classification with cross-validation.

    Parameters:
        data (DataFrame or ndarray): The expression matrix with genes in columns and samples in rows.
        labels (array-like): Labels for each sample.
        vars_selected (list): Genes calibreselected for classification. Can be DEGs or a custom list.
        num_fold (int): Number of folds for cross-validation. Default is 10.
        loocv (bool): If True, use Leave-One-Out cross-validation. Otherwise, use KFold.

    Returns:
        dict: A dictionary containing confusion matrices, accuracy, sensitivity, specificity, F1 scores,
              best k value for k-NN, and predictions.
    """
    label_codes, unique_labels = pd.factorize(labels)

    data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0)
    data = data[vars_selected]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    logging.info("Tuning the optimal K...") # TODO: quitar?

    if not loocv:
        cv = RepeatedStratifiedKFold(n_splits=num_fold, n_repeats=3)
        results = _tune_k(data, label_codes, cv)
    else:
        logging.info("Running Leave One Out Cross-Validation...")
        loo = LeaveOneOut()
        results = _tune_k(data, label_codes, loo)

    logging.info("Classification completed successfully.")
    results["unique_labels"] = unique_labels
    return results


def _tune_k(data, label_codes, cv=5):
    """
    Perform k-fold cross-validation., tuning the best K value for k-NN using Grid Search.

    Parameters:
        data (pandas DataFrame): The expression matrix.
        cv (obj): Cross-validation strategy.
        label_codes (array-like): label_codes for each sample.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    max_k = min(20,
                len(np.unique(label_codes)) - 1)  # TODO: when we have 2 classes, always maxed by 1 (overfit train data)
    param_grid = {'n_neighbors': range(1, max_k + 1)}
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='macro'),
               'recall': make_scorer(recall_score, average='macro'),
               'f1_score': make_scorer(f1_score, average='macro')}

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring=scoring, refit="accuracy")
    grid_search.fit(data, label_codes)

    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_
    y_pred = grid_search.best_estimator_.predict(data)

    return {
        'model': grid_search.best_estimator_,
        'confusion_matrix': confusion_matrix(label_codes, y_pred),
        'accuracy': grid_search.best_score_,
        'precision': cv_results['mean_test_precision'][best_index],
        'recall': cv_results['mean_test_recall'][best_index],
        'f1_score': cv_results['mean_test_f1_score'][best_index],
        'y_pred': y_pred
    }
