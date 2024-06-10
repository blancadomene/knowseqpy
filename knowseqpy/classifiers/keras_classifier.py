import pandas as pd
from keras.src.layers import TimeDistributed
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.optimizers import Adam

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def create_model(lookback=0, n_features=0, n_steps=0, n_series=0, lstm_layers=(), dense_layers=(), use_dropout=False,
                 activation_func="sigmoid", optimizer=Adam, lr=0.01):
    assert lookback != 0 or n_features != 0 or n_steps != 0 or n_series != 0

    # Initialize the constructor
    model = Sequential()

    # LSTM Layers
    for i in range(len(lstm_layers)):
        return_sequences = (i < len(lstm_layers) - 1)

        if i == 0:
            model.add(LSTM(lstm_layers[i],
                           activation=activation_func, input_shape=(lookback, n_features),
                           return_sequences=return_sequences))
        else:
            model.add(LSTM(units=lstm_layers[i], activation=activation_func, return_sequences=return_sequences))

        if use_dropout:
            model.add(Dropout(0.2))

    model.add(RepeatVector(n_steps))

    # Dense layers
    for i in range(len(dense_layers)):
        model.add(TimeDistributed(Dense(dense_layers[i], activation=activation_func)))

        if use_dropout:
            model.add(Dropout(0.2))

    # Output layer
    model.add(TimeDistributed(Dense(n_series, activation="relu")))

    model.compile(optimizer=optimizer(learning_rate=lr), loss="mse")

    return model


def keras_classifier(data: pd.DataFrame, labels: pd.Series, vars_selected: list,
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
    input_dim = data.shape[1]  # TODO: Set input_dim based on the number of features

    if not cv_strategy:
        logger.info("Running Repeated Stratified K-Fold Cross-Validation with 10 folds")
        cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    param_grid = {
        "build_fn__optimizer": ["adam", "rmsprop"],
        "build_fn__dropout_rate": [0.0, 0.1, 0.2],
        "build_fn__init_mode": ["uniform", "lecun_uniform", "normal"],
        "batch_size": [10, 20, 40],
        "epochs": [50, 100]
    }
    scoring = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score, average="macro"),
               "recall": make_scorer(recall_score, average="macro"),
               "f1_score": make_scorer(f1_score, average="macro")}

    scaler = StandardScaler()
    model = KerasClassifier(build_fn=create_model, input_dim=input_dim, verbose=0)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy, scoring=scoring,
                               refit="accuracy")
    pipeline = make_pipeline(
        scaler,
        grid_search
    )
    pipeline.fit(data, label_codes)
    logger.info("Best parameters: %s", grid_search.best_params_)

    return make_pipeline(scaler, grid_search.best_estimator_)
