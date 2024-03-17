"""
This module provides functionality for correcting batch effects in gene expression data.
Batch effects are systematic non-biological variations observed between batches in high-throughput experiments,
which can significantly skew the data analysis if not properly corrected. This module supports
the Surrogate Variable Analysis (SVA) method and has a structure to incorporate the Combat method in the future.
"""
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .utils import dataframe_to_feather, feather_to_dataframe, get_logger, get_project_directory

logger = get_logger().getChild(__name__)


def batch_effect_removal(expression_df: pd.DataFrame, labels: pd.Series, method: str = "sva",
                         batch_groups: pd.Series = None) -> pd.DataFrame:
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
    if method == "sva":
        return _sva(expression_df, labels)
    elif method == "combat":
        return _combat(expression_df, labels, batch_groups)
    else:
        raise ValueError("Unsupported method. Please use 'combat' or 'sva'.")


def _sva(expression_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    logger.info("Calculating sva model using R to correct batch effect")
    with tempfile.TemporaryDirectory() as temp_dir:
        expression_data_path = Path(temp_dir, "expression_data.feather")
        labels_path = Path(temp_dir, "design_matrix.feather")
        batch_results_path = Path(temp_dir, "batch_results.feather")

        dataframe_to_feather(expression_df, expression_data_path)
        dataframe_to_feather(labels.to_frame(), labels_path)

        try:
            subprocess.run([
                "Rscript",
                get_project_directory() / "knowseqpy" / "r_scripts" / "batchEffectRemovalWorkflow.R",
                expression_data_path,
                labels_path,
                batch_results_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute batch effect removal R script. {e}") from e

        results = feather_to_dataframe(batch_results_path)
        results.set_index("row_name", inplace=True)
        results.index.name = None

        return results


def _combat(expression_df: pd.DataFrame, labels: pd.Series, batch_groups: pd.Series = None) -> pd.DataFrame:
    raise NotImplementedError("Combat method has not been implemented yet")
