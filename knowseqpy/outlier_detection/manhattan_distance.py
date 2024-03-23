"""
This module facilitates outlier identification in gene expression datasets by utilizing Manhattan distances.
It calculates the city block distances between all pairs of samples, identifying those that significantly diverge
from the norm as potential outliers. This method helps in refining datasets for more accurate downstream analysis.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def manhattan_distance(gene_expression_df: pd.DataFrame) -> list:
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
    manhattan_distances = pdist(gene_expression_df_t.values, metric="cityblock")
    manhattan_distance_matrix = squareform(manhattan_distances) / gene_expression_df_t.shape[1]
    distance_sum = np.sum(manhattan_distance_matrix, axis=0)

    q3, q1 = np.percentile(distance_sum, [75, 25])
    threshold = q3 + 1.5 * (q3 - q1)

    return gene_expression_df_t.index[distance_sum > threshold].tolist()
