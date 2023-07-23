import unittest

import os
import pandas as pd

from cpm import cpm
from read_dge import read_dge


class CpmTest(unittest.TestCase):
    def setUp(self):
        golden_cpm_path = os.path.normpath(os.path.join("test_fixtures", "golden", "cpm_breast.csv"))
        self.golden_cpm = pd.read_csv(golden_cpm_path, index_col=0)

    def test_valid_read_cpm(self):
        data_path = os.path.normpath(os.path.join("test_fixtures", "data_info_breast.csv"))
        data = pd.read_csv(data_path, sep=",", dtype="str")

        count_files = data.apply(lambda row: os.path.join(row["Path"], row["Run"] + ".count"), axis=1)
        count_files = count_files.str.replace(os.sep, "/", regex=False)  # Replace backslashes so it matches the golden
        res_dge = read_dge(count_files, path=os.path.join("test_fixtures"))
        res_cpm = cpm(res_dge)

        # NOTE: The original lib has a bug that makes first row the column name of the dataframe, therefore the golden
        # is not correct. We will skip the first row of the result.
        res_cpm = res_cpm.iloc[1:]

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(self.golden_cpm, res_cpm, check_dtype=False, check_like=True, check_exact=False,
                                      atol=0.1, rtol=0.1)


if __name__ == '__main__':
    unittest.main()
