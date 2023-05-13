import pandas as pd

from cpm import cpm
from read_dge import read_dge


# TODO: remove intermediate CSV and the hardcoded part (from DELETE_TEST)
# TODO: change counts_to_matrix name
# TODO: modify doc
def counts_to_matrix(file_name: str, sep: str = ",", ext: str = ""):
    """
    Returns a dataframe with the merged information of all count files.

    :param file_name: CSV-like file containing the name and path to each of the count files.
    The expected columns are Run, Path and Class.
    :param sep: The separator character of the file_name content. Set to "," by default.
    :param ext: The extension of the count file. Set to "" by default.

    :return: Pandas dataframe with the ensemble ID in the rows and all the samples of each count files in the columns.

    :raises FileNotFoundError: If file_name doesn't exist.
    :raises Exception: If input file doesn't contain the expected columns.
    """

    data = pd.read_csv(file_name, sep=sep, dtype="str")

    # Check that the expected columns are in the dataset
    for expected_col in ["Run", "Path", "Class"]:
        if expected_col not in data.columns:
            raise Exception("Couldn't find expected column: " + expected_col)

    # Build count files' path
    count_files = data["Path"] + "/" + data["Run"] + ext
    print("\nMerging " + str(len(data)) + " counts files...\n")  # TODO log

    # Read digital gene expression (DGE) count files and remove unwanted rows.
    # Unwanted rows are the ones which column-wise sum is < 1, and also the ones in rows_to_skip
    counts = read_dge(count_files)
    counts.drop("ENSG00000000003.13", inplace=True)  # TODO remove
    counts_per_million = cpm(counts)

    keep = counts_per_million[counts_per_million > 1]
    counts = counts[keep.sum(axis=1) >= 1]
    rows_to_skip = ["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique"]
    counts.drop(rows_to_skip, inplace=True)

    # Truncate row-names (ex: ENSG00000000005.5 to ENSG00000000005)
    counts.index = [row_name.split(".")[0] for row_name in counts.index.values]

    return counts, data["Class"]  # TODO only return counts, data[Class] is already in provided name
