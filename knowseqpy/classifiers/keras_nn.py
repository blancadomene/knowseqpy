import os

import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from knowseqpy.utils import get_logger

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logger = get_logger().getChild(__name__)


def create_nn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def nn(data: pd.DataFrame, labels: pd.Series, vars_selected: list, cv_strategy: BaseCrossValidator = None) -> Pipeline:
    """
    Conducts neural network classification using Keras.

    Args:
        data: The expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Selected genes for classification. Can be DEGs or a custom list.
        cv_strategy: CV strategy to use. If None, defaults to RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    Returns:
        A trained neural network model pipeline.
    """
    label_codes, unique_labels = pd.factorize(labels, sort=True)
    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").fillna(0)
    data = data[vars_selected]

    if not cv_strategy:
        logger.info("Running Repeated Stratified K-Fold Cross-Validation with 10 folds")
        cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    input_dim = data.shape[1]
    output_dim = len(unique_labels)
    nn_model = KerasClassifier(model=create_nn_model, input_dim=input_dim, output_dim=output_dim, epochs=100,
                               batch_size=32, verbose=0)

    scaler = StandardScaler()
    pipeline = Pipeline([
        ('standardscaler', scaler),
        ('nn_model', nn_model)
    ])

    # Fit the model
    for train_index, test_index in cv_strategy.split(data, label_codes):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = pd.get_dummies(label_codes[train_index]), pd.get_dummies(label_codes[test_index])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        logger.info("Fold accuracy: %s", score)

    return pipeline
