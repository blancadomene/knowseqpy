import logging
import unittest

import pandas as pd

from knowseqpy.normalization import cpm
from knowseqpy.utils import csv_to_dataframe


class CpmTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.golden_cpm = csv_to_dataframe(
            path_components=["../test_fixtures", "golden", "cpm_breast.csv"], index_col=0, header=0)

    def test_cpm(self):
        golden_dge = csv_to_dataframe(
            path_components=["../test_fixtures", "golden", "read_dge_counts_breast.csv"], index_col=0, header=0)
        res_cpm = cpm(golden_dge)

        pd.testing.assert_frame_equal(self.golden_cpm, res_cpm, check_exact=False, atol=0.1, rtol=0.1)

    def test_non_numeric_columns(self):
        non_numeric_dge = pd.DataFrame({"Gene1": ["abc", 20], "Gene2": [30, 50]})

        with self.assertRaises(ValueError):
            cpm(non_numeric_dge, remove_non_numeric=False)

    def test_empty_dataframe(self):
        res_cpm = cpm(pd.DataFrame())

        pd.testing.assert_frame_equal(pd.DataFrame(), res_cpm)

    def test_all_zero_counts(self):
        zero_counts_df = pd.DataFrame({"Gene1": [0, 0], "Gene2": [0, 0]})

        with self.assertRaises(ValueError):
            cpm(zero_counts_df)

    def test_dataframe_with_nan(self):
        nan_df = pd.DataFrame({"Gene1": [1000, None, 500], "Gene2": [None, 1200, 500]})
        res_cpm = cpm(nan_df).reset_index(drop=True)
        expected_df = cpm(pd.DataFrame({"Gene1": [500, ], "Gene2": [500, ]})).reset_index(drop=True)

        pd.testing.assert_frame_equal(expected_df, res_cpm)

    def test_remove_non_numeric_true(self):
        mixed_df = pd.DataFrame({"Gene1": [1000, 1500], "Metadata": ["Sample1", "Sample2"], "Gene2": [800, 1200]})
        res_cpm = cpm(mixed_df, remove_non_numeric=True)
        expected = cpm(pd.DataFrame({"Gene1": [1000, 1500], "Gene2": [800, 1200]}))

        pd.testing.assert_frame_equal(expected, res_cpm)

    def test_large_numbers(self):
        large_counts_df = pd.DataFrame({"Gene1": [1e9, 2e9], "Gene2": [1.5e9, 2.5e9]})
        result = cpm(large_counts_df)
        total_counts = large_counts_df.sum()
        expected = (large_counts_df / total_counts) * 1e6

        pd.testing.assert_frame_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
