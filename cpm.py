import numpy as np
import pandas as pd


def cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the normalized gene expression. Calculated by dividing the mapped reads count by a per
    million scaling factor of total mapped reads.

    :param counts: A pandas DataFrame containing the mapped reads counts for each gene.
                   Assumes all columns are numeric and there are no null values.

    :return: A pandas DataFrame with the normalized gene expression.
    """
    # TODO: Consider using counts.to_numpy() or other solutions for potentially faster computation
    counts = counts.dropna()
    counts = counts.select_dtypes(include=[np.number])

    normalized_counts = (counts * 1e6) / counts.sum()
    return normalized_counts
