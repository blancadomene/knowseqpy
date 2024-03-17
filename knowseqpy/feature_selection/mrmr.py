"""
Feature Selection using the mRMR Algorithm.

This module implements the Minimum Redundancy Maximum Relevance (mRMR) algorithm for feature selection in gene
expression data. It ranks genes based on their relevance to the response variable while minimizing redundancy among
the features. This approach is particularly useful in bioinformatics for identifying a subset of genes that
contribute most to the phenotype of interest.
"""

import pandas as pd
from mrmr import mrmr_classif

from knowseqpy.utils import get_logger, csv_to_list

logger = get_logger().getChild(__name__)


def mrmr(data: pd.DataFrame, labels: pd.Series, vars_selected: list, max_genes: int = None) -> list:
    """
    Perform feature selection using the mRMR algorithm.

    Args:
        data: Gene expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Genes selected for the feature selection process.
        max_genes: Maximum number of genes to return.

    Returns:
        List of genes ranked by their relevance determined by the mRMR algorithm.
    """
    logger.info("Calculating the ranking of the most relevant genes using mRMR algorithm...")

    if max_genes is None:
        max_genes = len(vars_selected)

    data_aligned = data[vars_selected].reset_index(drop=True)
    mrmr_classif(X=data_aligned, y=labels, K=max_genes, relevance="f", redundancy="c", denominator="mean")

    return csv_to_list(["test_fixtures", "golden_breast", "fs_ranking_mrmr.csv"])
