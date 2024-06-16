"""
This module contains functions for performing Keras neural network classification.
"""
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from typing import Tuple
import pandas as pd
from keras import layers, Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def create_model(meta: dict, dense_layers: Tuple[int, ...] = (), activation_func: str = "sigmoid",
                 dropout: bool = False) -> Sequential:
    """
    Creates a Keras Sequential model based on provided parameters.

    Args:
        meta: Metadata containing information about the input features.
        dense_layers: Tuple specifying the number of units in each dense layer.
        activation_func: Activation function to use in the dense layers.
        dropout: Whether to include dropout layers after each dense layer.

    Returns:
        A compiled Keras Sequential model.
    """

    model = Sequential()
    n_features_in_ = meta["n_features_in_"]
    model.add(layers.Input(shape=(n_features_in_,)))

    for i in range(len(dense_layers)):
        model.add(layers.Dense(units=dense_layers[i], activation=activation_func))
        if dropout:
            model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=1, activation="sigmoid"))

    return model


def neural_network(data: pd.DataFrame, labels: pd.Series, vars_selected: list,
                   cv_strategy: RepeatedStratifiedKFold = None) -> Pipeline:
    """
    Conducts classification using a Keras neural network model with grid search.

    Args:
        data: The expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Selected genes for classification. Can be DEGs or a custom list.
        cv_strategy: CV strategy to use. If None, defaults to RepeatedStratifiedKFold(n_splits=10, n_repeats=3).

    Returns:
        A trained KerasClassifier model pipeline.
    """
    label_codes, unique_labels = pd.factorize(labels, sort=True)
    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").fillna(0)
    data = data[vars_selected]

    if not cv_strategy:
        logger.info("Running Repeated Stratified K-Fold Cross-Validation with 10 folds")
        cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    # TODO
    param_grid = {
        "model__dense_layers": [(64,)],
        "model__activation_func": ["relu"],
        "model__dropout": [True],
        "optimizer__learning_rate": [0.5],
        "batch_size": [16],
        "epochs": [50]
    }

    scoring = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score, average="macro"),
               "recall": make_scorer(recall_score, average="macro"),
               "f1_score": make_scorer(f1_score, average="macro")}

    scaler = StandardScaler()
    model = KerasClassifier(model=create_model, loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam",
                            verbose=False)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy, scoring=scoring,
                               refit="accuracy")
    pipeline = make_pipeline(
        scaler,
        grid_search
    )
    pipeline.fit(data, label_codes)
    logger.info("Best parameters: %s", grid_search.best_params_)

    return make_pipeline(scaler, grid_search.best_estimator_)
