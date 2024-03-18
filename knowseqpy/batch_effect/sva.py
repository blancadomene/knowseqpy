"""
This module provides functionality for correcting batch effects in gene expression data using SVA.
Batch effects are systematic non-biological variations observed between batches in high-throughput experiments,
which can significantly skew the data analysis if not properly corrected.
"""
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from knowseqpy.utils import dataframe_to_feather, feather_to_dataframe, get_logger, get_project_path

logger = get_logger().getChild(__name__)


def sva(expression_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Corrects the batch effect in the expression matrix using sva method.

    Args:
        expression_df: Expression df with genes as rows and samples as columns.
        labels: Labels or factors for the samples in expression_df.

    Returns:
        pd.DataFrame The expression matrix with the batch effect corrected.

    Raises:
        RuntimeError: If external R script execution fails.
    """
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
                get_project_path() / "knowseqpy" / "r_scripts" / "batchEffectRemovalWorkflow.R",
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
