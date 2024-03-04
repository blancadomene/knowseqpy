import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import quantile_transform

VALID_LENGTH_METHOD = {"smooth", "fixed"}

logger = logging.getLogger(__name__)


def _todo_cqn(counts: pd.DataFrame,
              x: pd.Series,
              lengths: pd.Series,
              size_factors: pd.Series = None,
              sub_index: list = None,
              tau: float = 0.5,
              sqn: bool = True,
              length_method: str = "smooth") -> tuple:
    """
    Performs conditional quantile normalization (CQN) on the given counts' matrix.

    Args:
        counts: The counts matrix with genes in rows and samples in columns.
        x: The predictor variable (e.g., GC content).
        lengths: The lengths of the genes.
        size_factors: The size factors for normalization. If None, calculated from 'counts'.
        sub_index: The subset of rows to use for robust fitting. If None, all rows are used.
        tau: The quantile for Quantile Regression.
        sqn: Whether to perform secondary quantile normalization.
        length_method: Either "smooth" or "fixed" for the type of length adjustment.

    Returns:
        dict: A dictionary containing various normalized and transformed data.
    """
    if length_method not in VALID_LENGTH_METHOD:
        err_msg = "Expected values for `length_method` parameter are: 'smooth' or 'fixed'."
        logger.error(err_msg)
        raise ValueError(err_msg)

    if size_factors is None:
        size_factors = counts.sum(axis=0)

    if sub_index is None:
        sub_index = counts.index

    # Log transform and length adjustment
    y = np.log2(counts + 1) - np.log2(size_factors / 1e6)

    # TODO: offset does not give the results it should
    if length_method == "fixed":
        y -= np.log2(lengths / 1e3)

    # Sub-setting for robust fitting
    y_fit = y.loc[sub_index]

    # Quantile normalization using Quantile Regression from statsmodels
    logging.info("Performing quantile normalization...")

    fitted = []
    for col in y_fit.columns:
        model = sm.QuantReg(y_fit[col], sm.add_constant(x.loc[sub_index]))
        res = model.fit(q=tau)
        fitted.append(res.predict(sm.add_constant(x)))

    fitted_df = pd.DataFrame(fitted).T
    fitted_df.columns = y_fit.columns
    fitted_df.index = y.index

    logging.info("Quantile normalization completed.")

    residuals = y - fitted_df

    # Secondary Quantile Normalization (SQN) if needed
    if sqn:
        residuals = pd.DataFrame(quantile_transform(residuals, axis=0, copy=True), index=residuals.index,
                                 columns=residuals.columns)

    offset = residuals - y

    return y, offset
