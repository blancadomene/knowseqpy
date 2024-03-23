"""
This module identifies outliers in gene expression data using the Median Absolute Deviation (MAD). It offers a
straightforward approach to flag samples that deviate significantly from the median expression level,
aiding in the quality control of the datasets.
"""

import pandas as pd
from scipy.stats import median_abs_deviation

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def median_deviation(gene_expression_df: pd.DataFrame) -> list:
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
