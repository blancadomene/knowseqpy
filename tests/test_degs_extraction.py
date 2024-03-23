import logging
import unittest

import pandas as pd

from knowseqpy.degs_extraction import degs_extraction
from knowseqpy.utils import csv_to_dataframe, get_test_path


class TestDegsExtraction(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.test_path = get_test_path()
        self.golden_degs_matrix = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "degs_matrix.csv"], index_col=0,
            header=0)

    def test_rna_seq_qa(self):
        batch_df = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "batch_matrix_sva.csv"], header=0, index_col=0)
        quality_labels = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "qa_labels.csv"]).iloc[:, 0]

        res_degs = degs_extraction(batch_df, labels=quality_labels, lfc=3.5, p_value=0.001)
        pd.testing.assert_frame_equal(res_degs[0], self.golden_degs_matrix, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == "__main__":
        unittest.main()
