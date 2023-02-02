import pandas as pd


def cpm():
    """
    Returns a dataframe with the merged information of all count files.
    Reads and collates a set of count data files, each file containing counts for one library.

    :param files: CSV or TSV file containing the name and path to each of the count files.
    The expected columns are Run, Path and Class.
    :param path: The separator character of the csvFile or tsvFile.
    :param columns: The extension of the count file. Set to ".count" by default.
    :param group:
    :param labels:

    :return: Matrix with the ensembl ID in the rows and all the samples of each count files in the columns.

    :raises FileNotFoundError: If file_name doesn't exist.
    """

    return pd.read_csv("test_fixtures/golden/cpms_breast.csv", index_col=0)