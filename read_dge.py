import logging

import pandas as pd


# TODO ask if we only use x$counts, since I only implemented that part
# TODO: change labels type (should not be str but a vector of strings)
def read_dge(count_files: pd.Series, path: str = None, labels: str = None):
    """
    Reads and merges a set of text files containing gene expression counts.

    :param count_files: Pandas Series of filenames that contain sample information.
    :param path: string giving the directory containing the files. Set to None by default (current working directory).
    :param labels: names to associate with the files. Set to None by default (file names).

    :return: Pandas Dataframe, containing a row for each unique tag found in the input files and a column for each
    input file.

    :raises Exception: If row names are not unique within the row-names.
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
        logging.info(f"Meta tags detected: {meta_tags.values}")

    return counts
