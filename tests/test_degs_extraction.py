import logging
import os
import unittest

import pandas as pd

from knowseq.degs_extraction import degs_extraction
from knowseq.utils import csv_to_dataframe


class DegsExtractionTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.golden_degs_matrix = csv_to_dataframe(
            path_components=["test_fixtures", "golden", "degs_matrix_breast.csv"], index_col=0, header=0)

    def test_rna_seq_qa(self):
        batch_df = csv_to_dataframe(
            path_components=["test_fixtures", "golden", "batch_matrix_sva_breast.csv"], header=0, index_col=0)

        quality_labels_path = os.path.normpath(
            os.path.join("test_fixtures", "golden", "qa_labels_breast.csv"))
        quality_labels = pd.read_csv(quality_labels_path, header=None).iloc[:, 0]
        res_degs = degs_extraction(batch_df, labels=quality_labels, lfc=3.5, p_value=0.001)
        pd.testing.assert_frame_equal(res_degs[0], self.golden_degs_matrix, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == '__main__':
        unittest.main()
