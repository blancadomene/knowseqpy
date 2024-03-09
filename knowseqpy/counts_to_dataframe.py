"""
This module contains functions for processing RNA-Seq gene expression count files, compiling gene count data from
multiple files into a single DataFrame, using CPM and filtering out unwanted data based on specified criteria.
"""

import pandas as pd

from .normalization import cpm
from .read_dge import read_dge
from .utils import get_logger

logger = get_logger().getChild(__name__)

DEFAULT_ROWS_TO_SKIP = ["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique"]


def counts_to_dataframe(info_path: str, counts_path: str, sep: str = ",", ext: str = ".count",
                        rows_to_skip: list = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Merges the information of all count files found in `counts_path`, which ID correspond to the one found
    in `info_path["Internal.ID"]`.

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
        Pandas series with the classes that correspond to each sample.
    """
    rows_to_skip = rows_to_skip if rows_to_skip is not None else DEFAULT_ROWS_TO_SKIP

    data_info = pd.read_csv(info_path, sep=sep, dtype="str", usecols=["Internal.ID", "Sample.Type"])

    counts = read_dge(data_info=data_info, counts_path=counts_path, ext=ext)
    counts_per_million = cpm(counts)

    # Unwanted rows are the ones which column-wise cpm sum is < 1, and also the ones in rows_to_skip
    counts_to_keep = counts_per_million[counts_per_million > 1]
    counts_filtered = counts.loc[counts_to_keep.sum(axis=1) >= 1]

    # Ignore errors in case any row from rows_to_skip is not present in our dataset
    counts_filtered.drop(rows_to_skip, inplace=True, errors="ignore")

    # Truncate row-names (ex: ENSG00000000005.5 to ENSG00000000005) as annotation in next steps provides truncated name
    counts_filtered.index = [row_name.split(".")[0] for row_name in counts_filtered.index]

    # Match labels index with counts index, so we can match them later on
    labels = data_info.set_index("Internal.ID")["Sample.Type"]
    labels.index = labels.index.map(lambda x: f"{x}{ext}")

    return counts_filtered, labels
