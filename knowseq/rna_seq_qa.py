import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kstest, median_abs_deviation


def rna_seq_qa(expression_df: pd.DataFrame, output_dir: str = "SamplesQualityAnalysis") -> [pd.DataFrame, list]:
    """
    Perform the quality analysis of an expression matrix.

    Args:
        expression_df: A DataFrame that contains the gene expression values.
        output_dir: The output directory to store the analysis results.

    Returns:
        dict: A dictionary containing found outliers for each realized test or corrected data if to_removal is True.
    """
    # TODO: Check if there are NA values (manually removed them from the golden while testing)
    # TODO: Improve performance
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    distance_outliers = _manhattan_distances_outliers(expression_df)
    ks_outliers = _ks_outliers(expression_df)
    mad_outliers = _mad_outliers(expression_df)

    # Get common outliers at least between two of three methods
    common_outliers = set(distance_outliers) & set(ks_outliers) | set(distance_outliers) & set(mad_outliers) | set(
        ks_outliers) & set(mad_outliers)

    return expression_df.drop(columns=list(common_outliers)), list(common_outliers)


def _ks_outliers(expression_df: pd.DataFrame) -> list:
    def ecdf_1d(data):
        n = len(data)  # Number of data points
        x = np.sort(data)  # X-data for the ECDF
        y = np.arange(1, n + 1) / n  # Y-data for the ECDF
        return x, y

    flat_values = expression_df.values.flatten()
    x_ecdf, y_ecdf = ecdf_1d(flat_values)

    def ks_statistic(sample):
        cdf = lambda x: np.interp(x, x_ecdf, y_ecdf, left=0, right=1)
        return kstest(sample, cdf).statistic

    ks_results = expression_df.apply(ks_statistic)
    q3 = ks_results.quantile(0.75)
    iqr = ks_results.quantile(0.75) - ks_results.quantile(0.25)
    ks_threshold = q3 + 1.5 * iqr
    return ks_results[ks_results > ks_threshold].index.tolist()


def _manhattan_distances_outliers(expression_df: pd.DataFrame) -> list:
    # Transpose the matrix to have samples as rows and genes as columns, as `pdist` function expects them
    expression_df_t = expression_df.copy().transpose()

    # Calculate the Manhattan distances between samples
    manhattan_distances = pdist(expression_df_t.values, metric='cityblock')
    manhattan_distance_matrix = squareform(manhattan_distances) / expression_df_t.shape[1]
    distance_sum = np.sum(manhattan_distance_matrix, axis=0)

    # TODO: only upper threshold for some reason? if not, just threshold
    # Calculate the threshold for outliers (Q3 + 1.5 * IQR)
    q3, iqr = np.percentile(distance_sum, 75), np.subtract(*np.percentile(distance_sum, [75, 25]))
    upper_distance_outlier_threshold = q3 + 1.5 * iqr

    # Detect distance-based outliers
    outliers = expression_df_t.index[distance_sum > upper_distance_outlier_threshold].tolist()

    return outliers


def _mad_outliers(expression_df: pd.DataFrame) -> list:
    outliers = []
    row_expression = expression_df.iloc[:, 1:].mean()
    for i in range(len(row_expression)):
        expr_matrix = row_expression.drop(row_expression.index[i])

        upper_bound = expr_matrix.median() + 3 * median_abs_deviation(expr_matrix, scale=1)
        lower_bound = expr_matrix.median() - 3 * median_abs_deviation(expr_matrix, scale=1)

        if row_expression.iloc[i] < lower_bound or row_expression.iloc[i] > upper_bound:
            outliers.append(row_expression.index[i])

    return outliers
