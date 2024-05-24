"""
Common utility functions for working with CSV files and data structures.
"""
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logger import get_logger

logger = get_logger().getChild(__name__)


def get_project_path() -> Path:
    """
    Retrieves the directory of the project.

    Returns:
        Path: A Path object pointing to the project's root directory.
    """
    return Path(__file__).resolve().parent.parent.parent


def get_test_path() -> Path:
    """
    Retrieves the directory of the project's test folder.

    Returns:
        Path: A Path object pointing to the project's test directory.
    """
    project_path = Path(__file__).resolve().parent.parent.parent
    return project_path / "tests"


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
    filepath = Path(*path_components)
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
    filepath = Path(*path_components)
    logger.info("Loading CSV file from %s into a list", filepath)
    with filepath.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)


def dataframe_to_feather(data: pd.DataFrame, filepath: Path, **kwargs) -> None:
    """
    Saves a pandas DataFrame to a Feather file using pathlib paths.

    Args:
        data: pandas DataFrame with data.
        filepath: Pathlib object representing the path to the file.
        **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_feather().
    """
    data.to_feather(filepath, **kwargs)
    logger.info("Exporting Dataframe to Feather file at %s", filepath)


def feather_to_dataframe(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Loads a Feather file into a pandas DataFrame using pathlib paths.

    Args:
        filepath: Pathlib object representing the path to the Feather file.
        **kwargs: Additional keyword arguments to pass to pandas.read_feather().

    Returns:
        pandas DataFrame containing the Feather file data.
    """
    logger.info("Loading Feather file from %s into a dataframe", filepath)
    return pd.read_feather(filepath, **kwargs)


def calculate_specificity(conf_matrix: np.array) -> float:
    """
    Calculates specificity for each class in a binary or multiclass classification and returns the average.

    Args:
        conf_matrix: The confusion matrix of the model.

    Returns:
        The average specificity across all classes.
    """
    class_specificities = []
    for i, row in enumerate(conf_matrix):
        true_negatives = sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))
        false_positives = sum(np.delete(row, i))
        total_negatives = true_negatives + false_positives
        class_specificity = true_negatives / total_negatives if total_negatives > 0 else 0
        class_specificities.append(class_specificity)

    return np.mean(class_specificities)
