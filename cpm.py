import pandas as pd


def cpm(counts: pd.DataFrame):
    """
    Returns a DataFrame with the normalized gene expression. Calculated by dividing the mapped reads count by a per
    million scaling factor of total mapped reads.

    :param counts: ?????.

    :return: ????.
    """
    # TODO: check for non-numeric values or any other edge case?
    # TODO: counts.to_numpty()? Faster
    return (counts * 1e6) / counts.sum()