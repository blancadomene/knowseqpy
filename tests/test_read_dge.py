import unittest

import os
import pandas as pd

from knowseq.read_dge import read_dge


class ReadDgeTest(unittest.TestCase):
    def setUp(self):
        golden_dge_path = os.path.normpath(os.path.join("test_fixtures", "golden", "read_dge_counts_breast.csv"))
        self.golden_dge = pd.read_csv(golden_dge_path, index_col=0)

    def test_valid_read_dge(self):
        data_path = os.path.normpath(os.path.join("test_fixtures", "data_info_breast.csv"))
        data = pd.read_csv(data_path, sep=",", dtype="str")

        count_files = data.apply(lambda row: os.path.join(row["Path"], row["Run"] + ".count"), axis=1)
        res_dge = read_dge(count_files, path=os.path.join("test_fixtures"))

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(self.golden_dge, res_dge, check_dtype=False, check_like=True)

    def test_duplicated_row_names(self):
        data_path = os.path.normpath(os.path.join("test_fixtures", "data_info_breast_duplicated_rows.csv"))
        data = pd.read_csv(data_path, sep=",", dtype="str")

        count_files = data.apply(lambda row: os.path.join(row["Path"], row["Run"] + ".count"), axis=1)
        with self.assertRaises(Exception):
            read_dge(count_files, path=os.path.join("test_fixtures"))


if __name__ == '__main__':
    unittest.main()
