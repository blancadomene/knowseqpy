import unittest

import os
import pandas as pd

from knowseq.calculate_gene_expression_values import calculate_gene_expression_values


class CalculateGeneExpressionValuesTest(unittest.TestCase):
    def setUp(self):
        counts_matrix_path = os.path.normpath(os.path.join("test_fixtures", "golden", "counts_matrix_breast.csv"))
        self.counts_matrix = pd.read_csv(counts_matrix_path, header=0, index_col=0)
        annotation_path = os.path.normpath(os.path.join("test_fixtures", "annotation_breast.csv"))
        self.annotation = pd.read_csv(annotation_path, header=0, index_col="ensembl_gene_id")
        golden_gene_expression_path = os.path.normpath(
            os.path.join("test_fixtures", "golden", "gene_expression_matrix_breast.csv"))
        self.golden_gene_expression = pd.read_csv(golden_gene_expression_path, header=0, index_col=0)

    def test_calculate_gene_expression(self):
        res_gene_expression = calculate_gene_expression_values(self.counts_matrix, self.annotation)
        pd.testing.assert_frame_equal(res_gene_expression, self.golden_gene_expression, check_dtype=False,
                                      check_like=True, check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == '__main__':
        unittest.main()
