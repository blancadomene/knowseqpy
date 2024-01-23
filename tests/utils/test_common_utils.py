import os
import unittest
from unittest import mock

import pandas as pd

from knowseq.utils.common_utils import get_nested_value, load_csv_to_dataframe


class CommonUtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    @mock.patch("pandas.read_csv")
    def test_load_csv_to_dataframe_2(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        path_components = ["test_fixtures", "golden", "cqn_breast.csv"]
        df = load_csv_to_dataframe(path_components, index_col=0)

        self.assertIsInstance(df, pd.DataFrame)

        mock_read_csv.assert_called_once_with(os.path.join(*path_components), index_col=0, header="infer")

    def test_get_nested_value(self):
        data = {'a': {'b': {'c': 1}}}

        self.assertEqual(get_nested_value(data, ['a', 'b', 'c']), 1)
        self.assertEqual(get_nested_value(data, ['a', 'x'], default="Not Found"), "Not Found")
        with self.assertRaises(KeyError):
            get_nested_value(data, ['a', 'b', 'x'])


if __name__ == "__main__":
    unittest.main()
