import logging
import os

import pandas as pd

from knowseq.normalization import cpm
from knowseq.read_dge import read_dge

logger = logging.getLogger(__name__)


# TODO: R bug removes first row
def counts_to_matrix(file_name: str, sep: str = ",", ext: str = "", rows_to_skip: list = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns a dataframe with the merged information of all count files found in `file_name["Path"]`.

    Args:
        file_name: CSV-like file containing the name and path to each of the count files.
                   The expected columns are Run, Path, and Class.
        sep: The separator character of the file_name content.
        ext: The extension of the count file.
        rows_to_skip: List of rows to skip during processing. Defaults to common RNA-Seq metadata rows.


    Returns:
        Pandas dataframe with the ensemble ID in the rows and all the samples of each count files in the columns.

    Raises:
        FileNotFoundError: If `file_name` path doesn't exist.
        ValueError: If input file doesn't contain the expected columns.
        ValueError: If row name truncation fails.
    """

    if rows_to_skip is None:
        rows_to_skip = ["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique"]

    # TODO: lo de los paths. no hace falta try catch porque ya petara. usar mi funcion tb supongo
    data_info = pd.read_csv(file_name, sep=sep, dtype="str")

    # Check that the expected columns are in the dataset
    for expected_col in ["Run", "Path", "Class"]:
        if expected_col not in data_info.columns:
            err_msg = f"Couldn't find expected column: {expected_col}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    # Build count files' path
    count_files = data_info.apply(lambda row: os.path.join(row["Path"], row["Run"]) + ext, axis=1)
    logging.info(f"Merging {len(data_info)} counts files...")

    # Read digital gene expression (DGE) count files and remove unwanted rows.
    # Unwanted rows are the ones which column-wise sum is < 1, and also the ones in rows_to_skip
    counts = read_dge(count_files)
    counts_per_million = cpm(counts)

    rows_to_keep = counts_per_million[counts_per_million > 1]
    counts = counts[rows_to_keep.sum(axis=1) >= 1]
    counts.drop(rows_to_skip, inplace=True)

    # Truncate row-names (ex: ENSG00000000005.5 to ENSG00000000005)
    counts.index = [row_name.split(".")[0] for row_name in counts.index]

    return counts, data_info["Class"]
