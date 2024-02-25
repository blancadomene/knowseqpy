"""
Unit tests for the `common` module in the `knowseqpy` package.
"""
import logging
import os
import unittest
from unittest import mock
from unittest.mock import patch, mock_open

import pandas as pd

from knowseqpy.utils.common import csv_to_dataframe, csv_to_list, get_nested_value, dataframe_to_feather, \
    feather_to_dataframe


class CommonTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    @mock.patch("pandas.read_csv")
    def test_load_csv_to_dataframe(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        path_components = ["test_fixtures", "golden", "cqn.csv"]
        df = csv_to_dataframe(path_components, index_col=0)

        self.assertIsInstance(df, pd.DataFrame)

        mock_read_csv.assert_called_once_with(os.path.join(*path_components), index_col=0, header=None)

    def test_csv_to_list(self):
        with patch("knowseqpy.utils.common.open", mock_open(read_data=None), create=True):
            with patch("knowseqpy.utils.common.csv.reader",
                       return_value=[["row1item1", "row1item2"], ["row2item1", "row2item2"],
                                     ["row3item1", "row3item2"]]) as mock_csv_reader:
                result = csv_to_list(["path", "to", "file.csv"])

                mock_csv_reader.assert_called()
                self.assertEqual(first=result,
                                 second=[["row1item1", "row1item2"],
                                         ["row2item1", "row2item2"], ["row3item1", "row3item2"]])

    @mock.patch("knowseqpy.utils.common.pd.DataFrame.to_feather")
    def test_dataframe_to_feather(self, mock_to_feather):
        data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        path_components = ["tests", "data", "test.feather"]

        dataframe_to_feather(data, path_components)

        expected_filepath = os.path.normpath(os.path.join(*path_components))
        mock_to_feather.assert_called_once_with(expected_filepath)

    @mock.patch("knowseqpy.utils.common.pd.read_feather")
    def test_feather_to_dataframe(self, mock_read_feather):
        mock_read_feather.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        path_components = ["tests", "data", "test.feather"]
        df = feather_to_dataframe(path_components)

        self.assertIsInstance(df, pd.DataFrame)

        expected_filepath = os.path.normpath(os.path.join(*path_components))
        mock_read_feather.assert_called_once_with(expected_filepath)


def test_get_nested_value(self):
    data = {'a': {'b': {'c': 1}}}

    self.assertEqual(get_nested_value(data_dict=data, keys=['a', 'b', 'c']), 1)
    self.assertEqual(get_nested_value(data_dict=data, keys=['a', 'x'], default="Not Found"), "Not Found")
    with self.assertRaises(KeyError):
        get_nested_value(data, ['a', 'b', 'x'])


if __name__ == "__main__":
    unittest.main()
