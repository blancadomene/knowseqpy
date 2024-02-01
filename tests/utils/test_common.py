"""
Unit tests for the `common` module in the `knowseq` package.
"""
import logging
import os
import unittest
from unittest import mock
from unittest.mock import patch, mock_open

import pandas as pd

from knowseq.utils.common import csv_to_dataframe, csv_to_list, get_nested_value


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
        with patch("knowseq.utils.common.open", mock_open(read_data=None), create=True):
            with patch("knowseq.utils.common.csv.reader",
                       return_value=[["row1item1", "row1item2"], ["row2item1", "row2item2"],
                                     ["row3item1", "row3item2"]]) as mock_csv_reader:
                result = csv_to_list(["path", "to", "file.csv"])

                mock_csv_reader.assert_called()
                self.assertEqual(first=result,
                                 second=[["row1item1", "row1item2"],
                                         ["row2item1", "row2item2"], ["row3item1", "row3item2"]])


def test_get_nested_value(self):
    data = {'a': {'b': {'c': 1}}}

    self.assertEqual(get_nested_value(data_dict=data, keys=['a', 'b', 'c']), 1)
    self.assertEqual(get_nested_value(data_dict=data, keys=['a', 'x'], default="Not Found"), "Not Found")
    with self.assertRaises(KeyError):
        get_nested_value(data, ['a', 'b', 'x'])


if __name__ == "__main__":
    unittest.main()
