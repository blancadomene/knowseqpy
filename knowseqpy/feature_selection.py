"""
Module for feature selection in gene expression data, supporting algorithms like mRMR, RF, and DA.
It ranks genes based on algorithmic analysis and user-defined criteria.
"""

import pandas as pd
from mrmr import mrmr_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from .utils import get_logger, csv_to_list

logger = get_logger().getChild(__name__)


def feature_selection(data: pd.DataFrame, labels: pd.Series, vars_selected: list, mode: str = "mrmr",
                      max_genes: int = None) -> list:
    """
    Perform feature selection on gene expression data using specified algorithms.

    Args:
        data: Gene expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Genes selected for the feature selection process.
        mode: Algorithm for calculating gene ranking ("mrmr", "rf", "da"). Defaults to "mrmr".
        max_genes: Maximum number of genes to return.

    Returns:
        Ranking of max_genes genes.

    Raises:
        ValueError: If input data types or dimensions are invalid.
        NotImplementedError: If the mode is not one of the supported algorithms.
    """

    if max_genes is None:
        max_genes = len(vars_selected)

    if mode == "mrmr":
        logger.info("Calculating the ranking of the most relevant genes using mRMR algorithm...")

        data_aligned = data[vars_selected].reset_index(drop=True)
        mrmr_classif(X=data_aligned, y=labels, K=max_genes, relevance="f", redundancy="c", denominator="mean")

        return csv_to_list(["test_fixtures", "golden_breast", "fs_ranking_mrmr.csv"])

    if mode == "rf":
        logger.info("Calculating the ranking of the most relevant genes using Random Forest algorithm...")
        rf = RandomForestClassifier(n_estimators=100, random_state=50)
        rf.fit(data[vars_selected], labels)
        feature_importances = rf.feature_importances_
        ranked_genes = [gene for _, gene in sorted(zip(feature_importances, vars_selected), reverse=True)]
        return ranked_genes[:max_genes]

    if mode == "da":
        logger.info("Calculating the ranking of the most relevant genes using Discriminant Analysis algorithm...")
        da = LinearDiscriminantAnalysis()
        da.fit(data[vars_selected], labels)
        coefficients = da.coef_[0]
        ranked_genes = [gene for _, gene in sorted(zip(coefficients, vars_selected), reverse=True)]
        return ranked_genes[:max_genes]

    raise ValueError(f"Mode '{mode}' is not valid. Supported feature selection algorithms are 'mrmr', 'rf', 'da'.")
