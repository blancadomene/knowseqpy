"""
This module is designed to handle the processing and analysis of gene expression count data,
primarily sourced from RNA sequencing experiments. It provides functionality to read count data from
text files, merge multiple samples into a comprehensive pandas DataFrame, and perform basic preprocessing
such as normalization and identification of metadata tags.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

logger = logging.getLogger(__name__)


def read_dge(data_info: pd.DataFrame, counts_path: str, ext: str = ".count", labels: pd.Series = None) -> pd.DataFrame:
    """
    Reads and merges a set of text files containing gene expression counts.

    Args:
        data_info: Pandas DataFrame of filenames that contain sample information.
        counts_path: The directory path where `.count` files are stored. Each file contains gene expression data
                     in count format for a sample identified by Internal.ID.
        ext: The extension of the count file.
        labels: Custom labels to associate with the files. Defaults to filenames if None.

    Returns:
        A pandas DataFrame, containing a row for each unique gene identifier found in the input files and
        a column for each sample, with expression counts as values.

    Raises:
        Exception: If row names (gene identifiers) are not unique within any file.
    """
    count_files = [os.path.join(counts_path, f"{row['Internal.ID']}{ext}") for _, row in data_info.iterrows()]
    labels = labels if labels is not None else data_info["Internal.ID"] + ext

    logger.info("Merging %s counts files into a pandas DataFrame...", {len(count_files)})

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_read_count_file, count_files))

    counts_df = pd.concat(results, axis=1)
    counts_df.columns = labels
    counts_df.columns.name = None

    # Replace NaN with 0 for absent counts
    counts_df.fillna(0, inplace=True)

    # Detect and log any meta tags
    meta_tags_idx = counts_df.index[counts_df.index.str.startswith("_")]
    if meta_tags_idx.any():
        logger.info("Meta tags detected: %s", meta_tags_idx.values)

    return counts_df


def _read_count_file(sample_path: str) -> pd.DataFrame:
    """
    Reads a single gene expression count file.

    Args:
        sample_path: The file path to read.

    Returns:
        A pandas DataFrame with the gene expression counts from the file.

    Raises:
        ValueError: If row names are not unique within the file.
    """
    file_name = os.path.basename(sample_path)
    file_data = pd.read_csv(sample_path, sep='\t', index_col=0, names=[file_name], engine='python', dtype='Int64')
    if file_data.index.has_duplicates:
        err_msg = f"Duplicated row names in {file_name}. Row names must be unique."
        logger.error(err_msg)
        raise ValueError(err_msg)
    return file_data
