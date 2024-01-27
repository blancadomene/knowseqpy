import logging

import pandas as pd

logger = logging.getLogger(__name__)


# TODO: Ask if I should remove non-numerical cols or just raise an exception and let the the user check that out
def cpm(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes counts per million (CPM), also known as RPM (Reads per million). Returns a DataFrame with the normalized
    gene expression (normalizes only for sequencing depth). Calculated by dividing the mapped reads count by a per
    million scaling factor of total mapped reads.

    Args:
        counts_df: A pandas DataFrame containing the mapped reads counts for each gene.
                   Assumes all columns are numeric and there are no null values.

    Returns:
        A pandas DataFrame with the normalized gene expression.
    """

    # Ensure the DataFrame only contains numerical values and no nulls
    counts_df = counts_df.dropna()
    for col in counts_df.columns:
        if counts_df[col].apply(pd.to_numeric, errors='coerce').isna().sum() > 0:
            raise ValueError(f"Dataframe col {col} contains non-numeric values.")

    return (counts_df * 1e6) / counts_df.sum()
