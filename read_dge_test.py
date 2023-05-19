import unittest

import pandas as pd

from read_dge import read_dge


class read_dge_test(unittest.TestCase):

    def test_valid_read_dge(self):
        golden_read_dge = pd.read_csv("test_fixtures/golden/read_dge_counts_breast.csv", index_col=0)

        data = pd.read_csv("test_fixtures/data_info_breast.csv", sep=",", dtype="str")
        count_files = data["Path"] + "/" + data["Run"] + ".count"
        dge = read_dge(count_files, path="test_fixtures")

        # Note: R bug makes first row the column name, so it doesn't appear in the provided example
        dge.drop("ENSG00000000003.13", inplace=True)

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(golden_read_dge, dge, check_dtype=False, check_like=True)


if __name__ == '__main__':
    unittest.main()
