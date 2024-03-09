"""
This module provides functionalities for performing quality analysis on RNA sequencing data.
It includes methods for detecting outliers in gene expression data using various statistical techniques,
such as the Kolmogorov-Smirnov test, Median Absolute Deviation, and Manhattan distance analysis.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kstest, median_abs_deviation

from src.log import get_logger

logger = get_logger().getChild(__name__)


def rna_seq_qa(gene_expression_df: pd.DataFrame) -> [pd.DataFrame, list]:
    """
    Perform the quality analysis of an expression matrix.

    Args:
        gene_expression_df: A DataFrame that contains the gene expression values.

    Returns:
        dict: A dictionary containing found outliers for each realized test or corrected data if to_removal is True.
    """
    # TODO: Check if there are NA values (manually removed them from the golden_breast while testing)

    ks_outliers = _ks_outliers(gene_expression_df)
    mad_outliers = _mad_outliers(gene_expression_df)
    manhattan_outliers = _manhattan_distances_outliers(gene_expression_df)

    # Get common outliers at least between two of three methods
    common_outliers = set(ks_outliers) & set(mad_outliers) | set(ks_outliers) & set(manhattan_outliers) | set(
        mad_outliers) & set(manhattan_outliers)

    return gene_expression_df.drop(columns=list(common_outliers)), list(common_outliers)


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


def _ks_outliers(gene_expression_df: pd.DataFrame) -> list:
    """
    Identify outliers in an expression DataFrame using the Kolmogorov-Smirnov (KS) test. Applies the KS test to each
    sample in the DataFrame to compare the distribution of gene expression values against an empirical distribution
    function (ECDF) derived from the dataset.

    Args:
        gene_expression_df: DataFrame containing gene expression values with genes as rows and samples as columns.

    Returns:
        list: A list of sample indices considered outliers based on the KS test.
    """

    def ks_statistic(sample):
        def cdf(x):
            return np.interp(x, x_ecdf, y_ecdf, left=0, right=1)

        return kstest(sample, cdf).statistic

    flat_values = gene_expression_df.values.flatten()
    x_ecdf, y_ecdf = _ecdf_1d(flat_values)

    results = gene_expression_df.apply(ks_statistic)
    q3, q1 = np.percentile(results, [75, 25])
    threshold = q3 + 1.5 * (q3 - q1)

    return results[results > threshold].index.tolist()


def _manhattan_distances_outliers(gene_expression_df: pd.DataFrame) -> list:
    """
    Identify outliers based on the Manhattan distances between samples in an expression DataFrame.

    Args:
        gene_expression_df: DataFrame containing gene expression values with genes as rows and samples as columns.

    Returns:
        list: A list of sample indices considered outliers based on their Manhattan distances to other samples.
    """

    # Transpose the matrix to have samples as rows and genes as columns, as `pdist` function expects them
    gene_expression_df_t = gene_expression_df.transpose()

    # Calculate the Manhattan distances between samples and normalize them by the number of genes
    manhattan_distances = pdist(gene_expression_df_t.values, metric='cityblock')
    manhattan_distance_matrix = squareform(manhattan_distances) / gene_expression_df_t.shape[1]
    distance_sum = np.sum(manhattan_distance_matrix, axis=0)

    q3, q1 = np.percentile(distance_sum, [75, 25])
    threshold = q3 + 1.5 * (q3 - q1)

    return gene_expression_df_t.index[distance_sum > threshold].tolist()


def _mad_outliers(gene_expression_df: pd.DataFrame) -> list:
    """
    Identify outliers in an expression DataFrame based on the Median Absolute Deviation (MAD).

    Args:
        gene_expression_df: DataFrame containing gene expression values with genes as rows and samples as columns.

    Returns:
        list: A list of sample indices considered outliers based on the MAD criterion.
    """

    outliers = []
    row_expression = gene_expression_df.iloc[:, 1:].mean()

    # Calculate the median expression value across all genes for each sample, and identify outliers as
    # those samples whose expression is beyond 3 times the MAD from the median expression level
    for i in range(len(row_expression)):
        expr_matrix = row_expression.drop(row_expression.index[i])

        upper_bound = expr_matrix.median() + 3 * median_abs_deviation(expr_matrix, scale=1)
        lower_bound = expr_matrix.median() - 3 * median_abs_deviation(expr_matrix, scale=1)

        if row_expression.iloc[i] < lower_bound or row_expression.iloc[i] > upper_bound:
            outliers.append(row_expression.index[i])

    return outliers
