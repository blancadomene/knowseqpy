import unittest

import os
import pandas as pd

from knowseq.normalization import cpm


class CpmTest(unittest.TestCase):
    def setUp(self):
        golden_cpm_path = os.path.normpath(os.path.join("test_fixtures", "golden", "cpm_breast.csv"))
        self.golden_cpm = pd.read_csv(golden_cpm_path, index_col=0)

    def test_valid_read_cpm(self):
        read_dge_path = os.path.normpath(os.path.join("test_fixtures", "golden", "read_dge_counts_breast.csv"))
        golden_dge = pd.read_csv(read_dge_path, index_col=0)

        res_cpm = cpm(golden_dge)

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(self.golden_cpm, res_cpm, check_dtype=False, check_like=True, check_exact=False,
                                      atol=0.1, rtol=0.1)

    def test_non_numeric_columns(self):
        non_numeric_dge_path = os.path.normpath(os.path.join("test_fixtures", "read_dge_counts_non_numeric_values.csv"))
        non_numeric_dge = pd.read_csv(non_numeric_dge_path, index_col=0)
        with self.assertRaises(ValueError):
            cpm(non_numeric_dge)


if __name__ == '__main__':
    unittest.main()
