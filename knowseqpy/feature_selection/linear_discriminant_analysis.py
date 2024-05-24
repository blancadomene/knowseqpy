"""
Feature Selection using Discriminant Analysis.

This module applies Linear Discriminant Analysis (LDA) for feature selection in gene expression data.
It ranks genes based on the coefficients obtained from LDA, which aims to find a linear combination of features that
characterizes or separates two or more classes of samples. This technique is suited for situations where the goal is to
maximize the separability among known categories or phenotypes.
"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def linear_discriminant_analysis(data: pd.DataFrame, labels: pd.Series, vars_selected: list, max_genes: int = None) -> list:
    """
    Perform feature selection using Linear Discriminant Analysis.

    Args:
        data: Gene expression matrix with genes in columns and samples in rows.
        labels: Labels for each sample.
        vars_selected: Genes selected for the feature selection process.
        max_genes: Maximum number of genes to return.

    Returns:
        List of genes ranked by their coefficients determined by the Discriminant Analysis.
    """
    if max_genes is None:
        max_genes = len(vars_selected)

    logger.info("Calculating the ranking of the most relevant genes using Discriminant Analysis algorithm...")
    da = LinearDiscriminantAnalysis()
    pipeline = make_pipeline(
        StandardScaler(),
        da
    )
    pipeline.fit(data[vars_selected], labels)
    coefficients = da.coef_[0]
    ranked_genes = [gene for _, gene in sorted(zip(coefficients, vars_selected), reverse=True)]
    return ranked_genes[:max_genes]
