"""
Feature Selection using the Random Forest Algorithm.

This module leverages the Random Forest classifier for feature selection in gene expression data.
By analyzing the importance of each feature (gene) in the construction of the random forest, it ranks genes according
to their contribution to sample classification. This method is effective in scenarios where the relationship between
genes and the phenotype is nonlinear and complex.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def random_forest(data: pd.DataFrame, labels: pd.Series, vars_selected: list, max_genes: int = None) -> list:
    """
    Perform feature selection using the Random Forest algorithm.

    Args:
        data: Gene expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Genes selected for the feature selection process.
        max_genes: Maximum number of genes to return.

    Returns:
        List of genes ranked by their importance determined by the Random Forest algorithm.
    """
    if max_genes is None:
        max_genes = len(vars_selected)

    logger.info("Calculating the ranking of the most relevant genes using Random Forest algorithm...")
    rf = RandomForestClassifier(n_estimators=100, random_state=50)
    rf.fit(data[vars_selected], labels)
    feature_importance = rf.feature_importances_
    ranked_genes = [gene for _, gene in sorted(zip(feature_importance, vars_selected), reverse=True)]
    return ranked_genes[:max_genes]
