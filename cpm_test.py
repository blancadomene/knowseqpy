import unittest

import pandas as pd

from read_dge import read_dge
from cpm import cpm

class ReadDGE(unittest.TestCase):

    def test_valid_read_cpm(self):
        golden_cpm = pd.read_csv("test_fixtures/golden/cpms_breast.csv", index_col=0)

        data = pd.read_csv("test_fixtures/data_info_breast.csv", sep=",", dtype="str")
        count_files = data["Path"] + "/" + data["Run"] + ".count"
        dge = read_dge(count_files, path="test_fixtures")
        res = cpm(dge)

        # NOTE: The original lib has a bug that removed the first row in the dataframe, therefore the golden
        # is not correct. We will skip the first row of the result.
        res = res.iloc[1:]

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(golden_cpm, res, check_dtype=False, check_like=True, check_exact=False, atol=0.1, rtol=0.1)
        # return pd.read_csv("test_fixtures/golden/cpms_breast.csv", index_col=0)


if __name__ == '__main__':
    unittest.main()
