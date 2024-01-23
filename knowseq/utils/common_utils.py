import os

import pandas as pd


def load_csv_to_dataframe(path_components, index_col=None, header='infer', **kwargs):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    path_components (list of str): List of components in the file path.
    index_col (int, str, sequence of int / str, or False, optional): Column(s) to use as the row labels.
    header (int, list of int, default 'infer'): Row number(s) to use as the column names.
    **kwargs: Additional keyword arguments to pass to pandas.read_csv().

    Returns:
    pandas.DataFrame: DataFrame containing the CSV data.

    Raises:
    FileNotFoundError: If the CSV file is not found at the specified path.
    pd.errors.ParserError: If there is an issue with CSV parsing.
    """
    filepath = os.path.normpath(os.path.join(*path_components))
    try:
        return pd.read_csv(filepath, index_col=index_col, header=header, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing CSV file: {e}")


def get_nested_value(data_structure, keys, default=None):
    """
    Access nested elements in a data structure.

    Args:
        data_structure (dict or similar): The data structure to access.
        keys (list or tuple): Sequence of keys to access the nested element.
        default (optional): Default value to return if any key is not found.

    Returns:
        The value from the nested data structure or the default value if not found.

    Raises:
        KeyError: If a key is not found and default is None.
    """
    current_level = data_structure
    for key in keys:
        try:
            current_level = current_level[key]
        except (KeyError, TypeError, IndexError) as e:
            if default is None:
                raise KeyError(f"Key not found: {e}") from None
            return default
    return current_level
