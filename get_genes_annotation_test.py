import unittest

import pandas as pd

from get_genes_annotation import get_genes_annotation


class GetGenesAnnotation(unittest.TestCase):

    def test_valid_get_genes_annotation(self):
        golden_annotation = pd.read_csv("test_fixtures/GRCh38Annotation.csv", index_col=0)

        data = pd.read_csv("test_fixtures/data_info_breast.csv", sep=",", dtype="str")
        count_files = data["Path"] + "/" + data["Run"] + ".count"
        dge = read_dge(count_files, path="test_fixtures")
        cpm = cpm()

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(golden_cpm, cpm, check_dtype=False, check_like=True)
        # return pd.read_csv("test_fixtures/golden/cpms_breast.csv", index_col=0)


if __name__ == '__main__':
    unittest.main()
