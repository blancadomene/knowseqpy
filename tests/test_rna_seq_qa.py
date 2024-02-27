import logging
import unittest

import pandas as pd

from knowseqpy.rna_seq_qa import rna_seq_qa
from knowseqpy.utils import csv_to_dataframe


class RnaSeqQaTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.golden_qa = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "qa_matrix.csv"], index_col=0, header=0)

    def test_rna_seq_qa(self):
        gene_expression = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "gene_expression_matrix.csv"], index_col=0, header=0)

        res_qa, _ = rna_seq_qa(gene_expression)
        pd.testing.assert_frame_equal(res_qa, self.golden_qa, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == '__main__':
        unittest.main()
