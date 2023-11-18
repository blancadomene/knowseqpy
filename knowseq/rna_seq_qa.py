import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kstest, median_abs_deviation


def rna_seq_qa(expression_df: pd.DataFrame, output_dir: str = "SamplesQualityAnalysis") -> [pd.DataFrame, list]:
    """
    Perform the quality analysis of an expression matrix.

    TODO: This function in R generates different plots over expression data in order.
          We want to return as a stats dict (min, max, etc) in case we want to plot using a separate func

    Parameters:
    expression_df: A DataFrame that contains the gene expression values.
    output_dir: The output directory to store the analysis results.

    Returns:
    dict: A dictionary containing found outliers for each realized test or corrected data if to_removal is True.
    """
    # TODO: Check if there are NA values (manually removed them from the golden while testing)
    # TODO: Improve performance
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Transpose the matrix to have samples as rows and genes as columns
    # `pdist` function expects each sample to be in the rows of the input matrix, not the columns.
    expression_df = expression_df.transpose()

    outliers = {'Distance': [], 'KS': [], 'MAD': []}

    # --- SAMPLES DISTANCES OUTLIERS DETECTION --- #
    # Calculate the Manhattan distances between samples
    manhattan_distances = pdist(expression_df.values, metric='cityblock')
    manhattan_distance_matrix = squareform(manhattan_distances) / expression_df.shape[1]
    distance_sum = np.sum(manhattan_distance_matrix, axis=0)

    # TODO: only upper threshold for some reason?

    # Calculate the threshold for outliers (Q3 + 1.5 * IQR)
    q3 = np.percentile(distance_sum, 75)
    iqr = np.subtract(*np.percentile(distance_sum, [75, 25]))
    upper_distance_outlier_threshold = q3 + 1.5 * iqr

    # Detect distance-based outliers
    distance_outliers = expression_df.index[distance_sum > upper_distance_outlier_threshold].tolist()
    outliers['Distance'] = {'limit': upper_distance_outlier_threshold, 'outliers': distance_outliers}

    # --- KOLMOGOROV-SMIRNOV OUTLIERS DETECTION --- #
    # Transpose df again to have genes as columns and samples as rows
    expression_df = expression_df.transpose()

    # Function to calculate the Empirical Cumulative Distribution Function (ECDF) for one-dimensional data
    def ecdf_1d(data):
        n = len(data)  # Number of data points
        x = np.sort(data)  # X-data for the ECDF
        y = np.arange(1, n + 1) / n  # Y-data for the ECDF
        return x, y

    # Flatten the expression matrix to calculate the ECDF across all values
    flat_values = expression_df.values.flatten()
    x_ecdf, y_ecdf = ecdf_1d(flat_values)

    # Function to compute the KS statistic against the ECDF
    def ks_statistic(sample):
        cdf = lambda x: np.interp(x, x_ecdf, y_ecdf, left=0, right=1)
        return kstest(sample, cdf).statistic

    # Apply the KS test to each column (gene)
    ks_results = expression_df.apply(ks_statistic)

    # Identify outliers based on the KS test
    q3 = ks_results.quantile(0.75)
    iqr = ks_results.quantile(0.75) - ks_results.quantile(0.25)
    ks_threshold = q3 + 1.5 * iqr

    # Identify outliers based on the KS test
    ks_outliers = ks_results[ks_results > ks_threshold].index.tolist()
    outliers['KS'] = {'limit': ks_threshold, 'outliers': ks_outliers}

    # --- MAD OUTLIERS DETECTION --- #
    # Calculate the median absolute deviation
    """median_expression = np.median(expression_df, axis=0)
    mad_scores = np.median(np.abs(expression_df - median_expression), axis=0)

    # Detect MAD-based outliers
    mad_outliers = expression_df.columns[mad_scores > 3].tolist()
    outliers['MAD'] = {'limit': 3, 'outliers': mad_outliers}"""

    mad_outliers = []
    row_expression = expression_df.iloc[:, 1:].mean()
    for i in range(len(row_expression)):
        expr_matrix = row_expression.drop(row_expression.index[i])

        upper_bound = expr_matrix.median() + 3 * median_abs_deviation(expr_matrix, scale=1)
        lower_bound = expr_matrix.median() - 3 * median_abs_deviation(expr_matrix, scale=1)

        if row_expression.iloc[i] < lower_bound or row_expression.iloc[i] > upper_bound:
            mad_outliers.append(row_expression.index[i])

    outliers['MAD'] = {'limit': "-", 'outliers': mad_outliers}

    # Get common outliers at least between two of three methods
    common_outliers = set(distance_outliers) & set(ks_outliers) | set(distance_outliers) & set(mad_outliers) | set(
        ks_outliers) & set(mad_outliers)

    quality_matrix = expression_df.drop(columns=list(common_outliers))

    return quality_matrix, list(common_outliers)
