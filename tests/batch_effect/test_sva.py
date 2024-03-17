import logging
import unittest

import pandas as pd

from knowseqpy.batch_effect import sva
from knowseqpy.utils import csv_to_dataframe


class BatchEffectRemovalTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.golden_batch = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "batch_matrix_sva.csv"], index_col=0, header=0)

    def test_sva(self):
        qa_matrix = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "qa_matrix.csv"], header=0, index_col=0)
        qa_labels = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "qa_labels.csv"]).iloc[:, 0]

        qa_matrix.index.name = "external_gene_name"
        qa_labels.name = "Sample.Type"

        res_qa = sva(qa_matrix, labels=qa_labels)
        pd.testing.assert_frame_equal(res_qa, self.golden_batch, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == '__main__':
        unittest.main()
