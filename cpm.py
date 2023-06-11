import pandas as pd


def cpm(counts: pd.DataFrame):
    """
    Returns a DataFrame with the normalized gene expression. Calculated by dividing the mapped reads count by a per
    million scaling factor of total mapped reads.

    :param counts: A pandas DataFrame containing the mapped reads counts for each gene.

    :return: A pandas DataFrame with the normalized gene expression.
    """
    # TODO: Check for non-numeric values or any other edge case?
    # TODO: Consider using counts.to_numpy() for potentially faster computation

    return (counts * 1e6) / counts.sum()