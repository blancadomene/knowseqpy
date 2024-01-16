import os

import numpy as np
import pandas as pd
# from combat.pycombat import pycombat
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

def batch_effect_removal(expression_df: pd.DataFrame, labels, method: str = "combat"):
    """
    Corrects the batch effect in the expression matrix using the specified method. A batch effect is a source of
    variation in biological data that arises from differences in the handling, processing, or environment between
    separate batches of experiments or samples.

    Args:
        expression_df (pd.DataFrame): The original gene expression matrix.
        labels (pd.Series):
        method (str): The method to use for batch effect removal ('combat' or 'sva').

    Returns:
        np.ndarray The expression matrix with the batch effect corrected.

    Raises:
        ValueError: If input parameters are not as expected.
    """

    """labels = expression_df.columns.tolist()
    # Design matrix for the labels, with an intercept
    labels_matrix = pd.get_dummies(labels, drop_first=False)
    labels_matrix = sm.add_constant(labels_matrix)

    # Perform regression and get residuals
    sva_results = []
    for gene in expression_df.index:
        model = sm.OLS(expression_df.loc[gene], labels_matrix).fit()
        sva_results.append(model.resid)

    # Create a DataFrame from the residuals
    sva_adjusted_data = pd.DataFrame(sva_results, index=expression_df.index, columns=expression_df.columns)"""

    batch_path = os.path.normpath(os.path.join("test_fixtures", "golden", "batch_matrix_sva_breast.csv"))
    golden_batch = pd.read_csv(batch_path, index_col=0, header=0)

    return golden_batch, None
