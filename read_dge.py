import pandas as pd


# TODO ask if we only use x$counts, since I only implemented that part
def read_dge(count_files: pd.Series, path: str = None, labels: str = None):
    """
    Returns a dataframe with the merged information of all count files.
    Reads and collates a set of count data files, each file containing counts for one library.

    :param count_files: pandas series CSV or TSV file containing the name and path to each of the count files.
    The expected columns are Run, Path and Class.
    :param path: The separator character of the csvFile or tsvFile.
    :param labels:

    :return: Matrix with the ensembl ID in the rows and all the samples of each count files in the columns.

    :raises FileNotFoundError: If file_name doesn't exist.count_files
    """

    # Assign path and labels if given
    samples_data = path + "/" + count_files if path else count_files
    samples_labels = labels if labels else count_files.apply(lambda file_name: str(file_name).split(".")[0])
    samples = pd.DataFrame(samples_data.array, index=samples_labels, columns=("files",))

    # TODO: improve performance
    # Collate counts for unique tags
    counts = pd.DataFrame()
    for sample in samples["files"]:
        file_name = sample.split("/")[-1]
        file_data = pd.read_csv(sample, sep=r"\t", index_col=0, names=(file_name,), engine="python", dtype="Int64")
        if file_data.index.has_duplicates:
            raise Exception("There are duplicated row names in files param. Row names must be unique.")
        counts = counts.join(file_data, how="outer")

    counts = counts.fillna(0)  # TODO: find a way of replacing NaN while joining

    """ 2
    counts = None
    for sample in samples["files"]:
        file_name = sample.split("/")[-1]
        file_data = pd.read_csv(sample, sep=r"\t", index_col=0, names=(file_name,), engine="python")
        if file_data.index.has_duplicates:
            raise Exception("There are duplicated row names in files param. Row names must be unique.")
        counts = file_data if counts is None else counts.join(file_data)
    """
    """ 1
    res = pd.DataFrame()
    [res.join(pd.read_csv(sample, sep=r"\t", index_col=0, names=(sample.split("/")[-1],), engine="python"), how="outer")
     for sample in samples["files"]]
    """
    # Alert user if htseq-style meta genes found

    meta_tags = counts[counts.index.str.startswith("_")].index
    if len(meta_tags):
        print("Meta tags detected: " + str(meta_tags.values))  # TODO log

    return counts
