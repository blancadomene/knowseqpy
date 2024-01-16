import os
import unittest

import pandas as pd

from knowseq.rna_seq_qa import rna_seq_qa


class RnaSeqQaTest(unittest.TestCase):
    def setUp(self):
        qa_path = os.path.normpath(os.path.join("test_fixtures", "golden", "qa_matrix_breast.csv"))
        self.golden_qa = pd.read_csv(qa_path, index_col=0, header=0)

    def test_rna_seq_qa(self):
        gene_expression_path = os.path.normpath(
            os.path.join("test_fixtures", "golden", "gene_expression_matrix_breast.csv"))
        gene_expression = pd.read_csv(gene_expression_path, header=0, index_col=0)

        res_qa, _ = rna_seq_qa(gene_expression)
        pd.testing.assert_frame_equal(res_qa, self.golden_qa, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == '__main__':
        unittest.main()
