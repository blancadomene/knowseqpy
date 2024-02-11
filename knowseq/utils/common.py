"""
Common utility functions for working with CSV files and data structures.
"""
import csv
import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def csv_to_dataframe(path_components: list[str], index_col=None, header=None, **kwargs: Any) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        path_components: List of components in the file path.
        index_col: Column(s) to use as the row labels (index). Defaults to None.
        header: Row number(s) to use as the column names. Defaults to None.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv().

    Returns:
        pandas DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the CSV file is not found at the specified path.
        pd.errors.ParserError: If there is an issue with CSV parsing.
    """
    filepath = os.path.normpath(os.path.join(*path_components))
    logger.info("Loading CSV file from %s into a dataframe", filepath)
    return pd.read_csv(filepath, index_col=index_col, header=header, **kwargs)


def csv_to_list(path_components: list[str]) -> list:
    """
    Loads a CSV file into a list.

    Args:
        path_components: List of components in the file path.

    Returns:
        A list where each element is a row of the CSV.
    """
    filepath = os.path.normpath(os.path.join(*path_components))
    logger.info("Loading CSV file from %s into a list", filepath)
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)


def dataframe_to_feather(data: pd.DataFrame, path_components: list[str], **kwargs: Any) -> None:
    """
    Saves a pandas DataFrame to a Feather file.

    Args:
        data: pandas DataFrame with data.
        path_components: List of components in the file path.
        **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_feather().
    """
    filepath = os.path.normpath(os.path.join(*path_components))
    data.to_feather(filepath, **kwargs)
    logger.info("Exporting Dataframe to Feather file at %s", filepath)


def feather_to_dataframe(path_components: list, **kwargs: Any) -> pd.DataFrame:
    """
    Loads a Feather file into a pandas DataFrame.

    Args:
        path_components: List of components in the file path.
        **kwargs: Additional keyword arguments to pass to pandas.read_feather().

    Returns:
        pandas DataFrame containing the Feather file data.
    """
    filepath = os.path.normpath(os.path.join(*path_components))
    logger.info("Loading Feather file from %s into a dataframe", filepath)
    return pd.read_feather(filepath, **kwargs)


def get_nested_value(data_dict: dict, keys: list[str], default: str = None) -> Any:
    """
    Access nested elements in a data structure.

    Args:
        data_dict: The data structure to access.
        keys: Sequence of keys to access the nested element.
        default: Default string value to return if any key is not found. Defaults to None.

    Returns:
        The value from the nested data structure or the default value if not found.

    Raises:
        KeyError: If a key is not found and default is None.
    """
    current_level = data_dict
    for key in keys:
        try:
            current_level = current_level[key]
        except (KeyError, TypeError, IndexError) as e:
            if default is None:
                raise KeyError(f"Key not found: {e}") from None
            return default
    return current_level
