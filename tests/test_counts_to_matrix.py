import logging
import os
import unittest

import pandas as pd

from knowseq.counts_to_matrix import counts_to_matrix
from knowseq.utils import csv_to_dataframe


class CountsToMatrixTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.golden_matrix = csv_to_dataframe(
            path_components=["test_fixtures", "golden", "counts_matrix_breast.csv"], index_col=0, header=0)

    """def test_file_not_exists(self):
        file_path = os.path.normpath(os.path.join("file", "doesnt", "exist.csv"))
        self.assertRaises(FileNotFoundError, counts_to_matrix, file_path)

    def test_missing_cols_csv(self):
        file_path = os.path.normpath(os.path.join("test_fixtures", "data_info_missing_cols"))
        self.assertRaises(Exception, counts_to_matrix, file_path)

    def test_valid_tsv(self):
        counts_path = os.path.normpath(os.path.join("test_fixtures", "data_info_breast.tsv"))
        counts_matrix, labels = counts_to_matrix(counts_path, sep="\t", ext=".count")

        pd.testing.assert_frame_equal(self.golden_matrix, counts_matrix, check_dtype=False, check_like=True)
"""

    def test_valid_csv(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        info_path = os.path.join(script_path, "test_fixtures", "samples_info_breast.csv")
        counts_path = os.path.join(script_path, "test_fixtures", "BreastCountFiles")
        counts_matrix, labels = counts_to_matrix(info_path=info_path, counts_path=counts_path)

        pd.testing.assert_frame_equal(self.golden_matrix, counts_matrix, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
