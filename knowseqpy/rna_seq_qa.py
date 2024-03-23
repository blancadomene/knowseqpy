"""
This module provides functionalities for performing quality analysis on RNA sequencing data.
It includes methods for detecting outliers in gene expression data using various statistical techniques,
such as the Kolmogorov-Smirnov test, Median Absolute Deviation, and Manhattan distance analysis.
"""

import pandas as pd

from .outlier_detection import kolmogorov_smirnov, manhattan_distance, median_deviation
from .utils import get_logger

logger = get_logger().getChild(__name__)


def rna_seq_qa(gene_expression_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Perform the quality analysis of an expression df.

    Args:
        gene_expression_df: A DataFrame that contains the gene expression values.

    Returns:
        tuple: A tuple where the first element is a DataFrame with outliers removed based on the
               commonality between at least two of three methods (KS test, MAD, and Manhattan distances),
               and the second element is a list containing the identifiers of the detected outliers.
    """
    ks_outliers = kolmogorov_smirnov(gene_expression_df)
    logger.info(f"KS outliers detected: {len(ks_outliers)} samples")
    mad_outliers = manhattan_distance(gene_expression_df)
    logger.info(f"MAD outliers detected: {len(mad_outliers)} samples")
    manhattan_outliers = median_deviation(gene_expression_df)
    logger.info(f"Manhattan distance outliers detected: {len(manhattan_outliers)} samples")

    # Get common outliers at least between two of three methods
    common_outliers = set(ks_outliers) & set(mad_outliers) | set(ks_outliers) & set(manhattan_outliers) | set(
        mad_outliers) & set(manhattan_outliers)
    logger.info(f"{len(common_outliers)} common outliers identified by at least two methods.")

    return gene_expression_df.drop(columns=list(common_outliers)), list(common_outliers)
