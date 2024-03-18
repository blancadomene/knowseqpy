"""
This module provides utilities for processing and normalizing gene expression data from RNA-Seq samples.
It implements a function for calculating Counts Per Million (CPM), a common method for normalizing
gene expression to account for differences in sequencing depth across samples. This allows for the
comparative analysis of gene expression data from different samples or experimental conditions.
"""

import pandas as pd

from knowseqpy.utils import get_logger

logger = get_logger().getChild(__name__)


def cpm(counts: pd.DataFrame, remove_non_numeric: bool = True) -> pd.DataFrame:
    """
    Computes counts per million (CPM), also known as RPM (Reads per million), normalizing gene expression data for
    sequencing depth. This enables comparison across samples by adjusting for the total number of reads.

    Args:
        counts: A DataFrame with mapped reads counts for each gene across samples.
                                  Rows represent genes and columns represent samples. Assumes no null values.
        remove_non_numeric: If True, non-numeric columns are automatically removed. If False, a ValueError
                                   is raised when non-numeric columns are present. Defaults to True.

    Returns:
        A pandas DataFrame with the normalized gene expression values (normalizes only for sequencing depth).

    Raises:
        ValueError: If `remove_non_numeric` is False and non-numeric columns are found.
        ValueError: If total counts for one or more samples are zero, which prevents computation of CPM.
    """
    logger.info("Performing CPM on gene expression data")

    is_numeric = counts.apply(lambda col: pd.to_numeric(col, errors="coerce").notnull().all())
    if not is_numeric.all():
        if not remove_non_numeric:
            non_numeric_cols = list(counts.columns[~is_numeric])
            err = f"DataFrame contains non-numeric columns: {non_numeric_cols}. " \
                  f"Set remove_non_numeric=True to remove them automatically."
            logger.error(err)
            raise ValueError(err)

        counts = counts.select_dtypes(include=["number"])
        logger.info("Non-numeric columns removed")

    if counts.sum().eq(0).any():
        err = "Total counts for one or more samples are zero, cannot compute CPM."
        logger.error(err)
        raise ValueError(err)

    return (counts * 1e6) / counts.sum()
