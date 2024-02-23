"""
This module contains functions for processing RNA-Seq gene expression count files, compiling gene count data from
multiple files into a single DataFrame, using CPM and filtering out unwanted data based on specified criteria.
"""

import logging

import pandas as pd

from knowseq.normalization import cpm
from knowseq.read_dge import read_dge

logger = logging.getLogger(__name__)

DEFAULT_ROWS_TO_SKIP = ["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique"]


# TODO: R bug removes first row
def counts_to_matrix(info_path: str, counts_path: str, sep: str = ",", ext: str = ".count",
                     rows_to_skip: list = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns a dataframe with the merged information of all count files found in `file_name["Path"]`.

    Args:
        info_path: path to CSV file that holds metadata for biological samples. Each row represents a unique sample
                   with various identifiers and descriptors. The expected columns are `Internal.ID` and `Sample.Type`.
        counts_path: The directory path where `.count` files are stored. Each file contains gene expression data
                     in count format for a sample identified by Internal.ID.
        sep: The separator character of the `info_path` content.
        ext: The extension of the count file.
        rows_to_skip: List of rows to skip during processing. Defaults to common RNA-Seq metadata rows.

    Returns:
        Pandas dataframe with gene ids (ensemble ID) as rows and sample ids of each count file as columns.
    """
    rows_to_skip = rows_to_skip if rows_to_skip is not None else DEFAULT_ROWS_TO_SKIP

    data_info_df = pd.read_csv(info_path, sep=sep, dtype="str", usecols=["Internal.ID", "Sample.Type"])

    counts_df = read_dge(data_info=data_info_df, counts_path=counts_path, ext=ext)
    counts_per_million_df = cpm(counts_df)

    # Unwanted rows are the ones which column-wise cpm sum is < 1, and also the ones in rows_to_skip
    rows_to_keep_df = counts_per_million_df[counts_per_million_df > 1]
    counts_filtered_df = counts_df.loc[rows_to_keep_df.sum(axis=1) >= 1]

    # Ignore errors in case any row from rows_to_skip is not present in our dataset
    counts_filtered_df.drop(rows_to_skip, inplace=True, errors='ignore')

    # Truncate row-names (ex: ENSG00000000005.5 to ENSG00000000005)
    counts_filtered_df.index = [row_name.split(".")[0] for row_name in counts_filtered_df.index]

    labels_ser = data_info_df.set_index("Internal.ID")["Sample.Type"]

    return counts_filtered_df, labels_ser
