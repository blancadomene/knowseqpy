import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import quantile_transform


def cqn(counts: pd.DataFrame,
        x: pd.Series,
        lengths: pd.Series,
        size_factors: pd.Series = None,
        sub_index: list = None,
        tau: float = 0.5,
        sqn: bool = True,
        length_method: str = "smooth"):
    """
    Performs conditional quantile normalization (CQN) on the given counts' matrix.

    :param counts: The counts matrix with genes in rows and samples in columns.
    :param x: The predictor variable (e.g., GC content).
    :param lengths: The lengths of the genes.
    :param size_factors: The size factors for normalization. If None, calculated from 'counts'.
    :param sub_index: The subset of rows to use for robust fitting. If None, all rows are used.
    :param tau: The quantile for Quantile Regression.
    :param sqn: Whether to perform secondary quantile normalization.
    :param length_method: Either "smooth" or "fixed" for the type of length adjustment.

    :return dict: A dictionary containing various normalized and transformed data.
    """

    if size_factors is None:
        size_factors = counts.sum(axis=0)

    if sub_index is None:
        sub_index = counts.index

    # Log transform and length adjustment
    y = np.log2(counts + 1) - np.log2(size_factors / 1e6)
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

    # Calculate residuals and offset
    residuals = y - fitted_df

    # Secondary Quantile Normalization (SQN) if needed
    if sqn:
        residuals = pd.DataFrame(quantile_transform(residuals, axis=0, copy=True), index=residuals.index,
                                 columns=residuals.columns)

    offset = residuals - y

    # Output
    output = {
        'counts': counts,
        'lengths': lengths,
        'size_factors': size_factors,
        'y': y,
        'x': x,
        'offset': offset
    }

    return output
