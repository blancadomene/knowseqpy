"""
This module provides utilities for processing and normalizing gene expression data from RNA-Seq samples.
It implements a function for calculating Counts Per Million (CPM), a common method for normalizing
gene expression to account for differences in sequencing depth across samples. This allows for the
comparative analysis of gene expression data from different samples or experimental conditions.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def cpm(counts_df: pd.DataFrame, remove_non_numeric: bool = True) -> pd.DataFrame:
    """
    Computes counts per million (CPM), also known as RPM (Reads per million), normalizing gene expression data for
    sequencing depth. This enables comparison across samples by adjusting for the total number of reads.

    Args:
        counts_df (pd.DataFrame): A DataFrame with mapped reads counts for each gene across samples.
                                  Rows represent genes and columns represent samples. Assumes no null values.
        remove_non_numeric (bool): If True, non-numeric columns are automatically removed. If False, a ValueError
                                   is raised when non-numeric columns are present. Defaults to True.

    Returns:
        A pandas DataFrame with the normalized gene expression values (normalizes only for sequencing depth).

    Raises:
        ValueError: If `remove_non_numeric` is False and non-numeric columns are found.
        ValueError: If total counts for one or more samples are zero, which prevents computation of CPM.

    """
    logger.info("Performing CPM on gene expression data")

    counts_df = counts_df.dropna()

    is_numeric = counts_df.apply(lambda col: pd.to_numeric(col, errors="coerce").notnull().all())
    if not is_numeric.all():
        if remove_non_numeric:
            counts_df = counts_df.select_dtypes(include=["number"])
            logger.info("Non-numeric columns removed")
        else:
            non_numeric_cols = list(counts_df.columns[~is_numeric])
            err_msg = (f"DataFrame contains non-numeric columns: {non_numeric_cols}. "
                       f"Set remove_non_numeric=True to remove them automatically.")
            logger.error(err_msg)
            raise ValueError(err_msg)

    if counts_df.sum().eq(0).any():
        err_msg = "Total counts for one or more samples are zero, cannot compute CPM."
        logger.error(err_msg)
        raise ValueError(err_msg)

    return (counts_df * 1e6) / counts_df.sum()
