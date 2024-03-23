import logging
import os
import unittest
from pathlib import Path

import pandas as pd

from knowseqpy.utils import csv_to_dataframe, dataframe_to_feather, feather_to_dataframe, get_project_path, \
    get_test_path


class TestCommon(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    def test_get_project_path(self):
        expected_path = Path(__file__).resolve().parents[2]
        self.assertEqual(get_project_path(), expected_path)

    def test_get_test_path(self):
        expected_path = Path(__file__).resolve().parents[2] / "tests"
        self.assertEqual(get_test_path(), expected_path)

    def test_csv_to_dataframe(self):
        test_csv_path = get_test_path() / "test_fixtures" / "samples_info_breast.csv"
        df = csv_to_dataframe([str(test_csv_path)])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_dataframe_to_feather_and_back(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        feather_path = get_test_path() / "test.feather"
        dataframe_to_feather(df, feather_path)
        loaded_df = feather_to_dataframe(feather_path)

        pd.testing.assert_frame_equal(df, loaded_df)

        os.remove(feather_path)


if __name__ == "__main__":
    unittest.main()
