import unittest

import os
import pandas as pd

from counts_to_matrix import counts_to_matrix


class CountsToMatrixTest(unittest.TestCase):
    def test_file_not_exists(self):
        file_path = os.path.normpath(os.path.join(os.getcwd(), "../file/doesnt/exist"))
        self.assertRaises(FileNotFoundError, counts_to_matrix, file_path)

    def test_missing_cols_csv(self):
        file_path = os.path.normpath(os.path.join(os.getcwd(), "../test_fixtures/data_info_missing_cols.csv"))
        self.assertRaises(Exception, counts_to_matrix, file_path)

    def test_valid_tsv(self):
        golden_path = os.path.normpath(os.path.join(os.getcwd(), "../test_fixtures/golden/counts_matrix_breast.csv"))
        golden_counts_matrix = pd.read_csv(golden_path, index_col=0)

        counts_path = os.path.normpath(os.path.join(os.getcwd(), "../test_fixtures/data_info_breast.tsv"))
        counts_matrix, labels = counts_to_matrix(counts_path, sep="\t", ext=".count")

        pd.testing.assert_frame_equal(golden_counts_matrix, counts_matrix, check_dtype=False, check_like=True)

    def test_valid_csv(self):
        golden_path = os.path.normpath(os.path.join(os.getcwd(), "../test_fixtures/golden/counts_matrix_breast.csv"))
        golden_counts_matrix = pd.read_csv(golden_path, index_col=0)
        counts_path = os.path.normpath(os.path.join(os.getcwd(), "../test_fixtures/data_info_breast.csv"))
        counts_matrix, labels = counts_to_matrix(counts_path, ext=".count")

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(golden_counts_matrix, counts_matrix, check_dtype=False, check_like=True)


if __name__ == '__main__':
    unittest.main()
