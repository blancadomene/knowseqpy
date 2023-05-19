import unittest

import pandas as pd

from counts_to_matrix import counts_to_matrix


class counts_to_matrix_test(unittest.TestCase):
    def test_file_not_exists(self):
        self.assertRaises(FileNotFoundError, counts_to_matrix, "/file/doesnt/exist")

    def test_missing_cols_csv(self):
        self.assertRaises(Exception, counts_to_matrix, "test_fixtures/data_info_missing_cols.csv")

    def test_valid_tsv(self):
        golden_counts_matrix = pd.read_csv("test_fixtures/golden/golden_counts_matrix_breast.csv", index_col=0)
        counts_matrix, labels = counts_to_matrix("test_fixtures/data_info_breast.tsv", sep="\t", ext=".count")

        pd.testing.assert_frame_equal(golden_counts_matrix, counts_matrix, check_dtype=False, check_like=True)

    def test_valid_csv(self):
        golden_counts_matrix = pd.read_csv("test_fixtures/golden/golden_counts_matrix_breast.csv", index_col=0)
        counts_matrix, labels = counts_to_matrix("test_fixtures/data_info_breast.csv", ext=".count")

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(golden_counts_matrix, counts_matrix, check_dtype=False, check_like=True)


if __name__ == '__main__':
    unittest.main()
