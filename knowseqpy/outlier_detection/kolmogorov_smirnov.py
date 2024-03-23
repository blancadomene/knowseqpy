"""
This module provides functionality for detecting outliers in gene expression data sets using
the `kolmogorov_smirnov` test, comparing each sample's gene expression distribution against the empirical cumulative
distribution function (ECDF) derived from the entire dataset.
"""

import numpy as np
import pandas as pd
from scipy.stats import kstest

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def kolmogorov_smirnov(gene_expression: pd.DataFrame) -> list:
    """
    Identify outliers in an expression DataFrame using the Kolmogorov-Smirnov (KS) test. Applies the KS test to each
    sample in the DataFrame to compare the distribution of gene expression values against an empirical distribution
    function (ECDF) derived from the dataset.

    Args:
        gene_expression: DataFrame containing gene expression values with genes as rows and samples as columns.

    Returns:
        list: A list of sample indices considered outliers based on the KS test.
    """
    if gene_expression.empty:
        raise ValueError("Input DataFrame is empty.")

    def ks_statistic(sample):
        def cdf(x):
            return np.interp(x, x_ecdf, y_ecdf, left=0, right=1)

        return kstest(sample, cdf).statistic

    flat_values = gene_expression.values.flatten()
    x_ecdf, y_ecdf = _ecdf_1d(flat_values)

    results = gene_expression.apply(ks_statistic)
    q3, q1 = np.percentile(results, [75, 25])
    threshold = q3 + 1.5 * (q3 - q1)

    return results[results > threshold].index.tolist()


def _ecdf_1d(data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the empirical cumulative distribution function (ECDF) for a one-dimensional array of data. This is a step
    function that increases by 1/n at each data point, where n is the number of data points.

    Args:
        data: The one-dimensional array of data for which to compute the ECDF.

    Returns:
        tuple: Two numpy arrays, where x contains the sorted data and y contains the ECDF values for each data point.
    """
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y
