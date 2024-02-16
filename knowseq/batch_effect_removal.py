"""
This module provides functionality for correcting batch effects in gene expression data.
Batch effects are systematic non-biological variations observed between batches in high-throughput experiments,
which can significantly skew the data analysis if not properly corrected. This module supports
the Surrogate Variable Analysis (SVA) method and has a structure to incorporate the Combat method in the future.
"""
import logging
import os
import subprocess

import pandas as pd

from knowseq.utils import dataframe_to_feather, feather_to_dataframe

logger = logging.getLogger(__name__)


def batch_effect_removal(expression_df: pd.DataFrame, labels: pd.Series, method: str = "sva",
                         batch_groups: pd.Series = None):
    """
    Corrects the batch effect in the expression matrix using the specified method. A batch effect is a source of
    variation in biological data that arises from differences in the handling, processing, or environment between
    separate batches of experiments or samples.

    Args:
        expression_df: Expression matrix with genes as rows and samples as columns.
        labels: Labels or factors for the samples in expression_df.
        method: Method for batch effect removal ('combat' or 'sva').
        batch_groups (optional): Known batch groups for the samples.

    Returns:
        np.ndarray The expression matrix with the batch effect corrected.

    Raises:
        ValueError: If input parameters are not as expected.
        RuntimeError: If external R script execution fails.
    """

    if method == "combat":
        raise NotImplementedError("Combat method has not been implemented yet")

    if method == "sva":
        logger.info("Calculating sva model using R to correct batch effect")
        script_path = os.path.dirname(os.path.abspath(__file__))
        expression_data_path = os.path.join(script_path, "r_scripts", "expression_data.feather")
        labels_path = os.path.join(script_path, "r_scripts", "design_matrix.feather")
        batch_results_path = os.path.join(script_path, "r_scripts", "batch_results.feather")

        dataframe_to_feather(expression_df, [script_path, "r_scripts", "expression_data.feather"])
        dataframe_to_feather(labels.to_frame(), [script_path, "r_scripts", "design_matrix.feather"])

        command = [
            "Rscript",
            os.path.join(script_path, "r_scripts", "batchEffectRemovalWorkflow.R"),
            expression_data_path,
            labels_path,
            batch_results_path
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("R script execution failed: %s", {e})
            raise RuntimeError("Failed to execute batch effect removal R script.") from e

    else:
        raise ValueError("Unsupported method. Please use 'combat' or 'sva'.")

    results = feather_to_dataframe([script_path, "r_scripts", "batch_results.feather"])
    results.set_index("row_name", inplace=True)
    results.index.name = None

    os.remove(expression_data_path)
    os.remove(labels_path)
    os.remove(batch_results_path)

    return results
