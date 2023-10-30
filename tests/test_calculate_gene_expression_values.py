import unittest

import os
import pandas as pd

from knowseq.calculate_gene_expression_values import calculate_gene_expression_values


class CalculateGeneExpressionValuesTest(unittest.TestCase):
    def setUp(self):
        dge_path = os.path.normpath(os.path.join(os.getcwd(), "test_fixtures/golden/counts_matrix.csv"))
        self.counts_matrix = pd.read_csv("test_fixtures/input/counts_matrix.csv", index_col=0)
        dge_path = os.path.normpath(os.path.join(os.getcwd(), "test_fixtures/golden/counts_matrix.csv"))
        self.annotation = pd.read_csv("test_fixtures/input/annotation.csv", index_col=0)
        dge_path = os.path.normpath(os.path.join(os.getcwd(), "test_fixtures/golden/counts_matrix.csv"))
        self.golden_expr = pd.read_csv("test_fixtures/golden/gene_expression_matrix_breast.csv", index_col=0)

    def test_gene_expression_ensembl_id(self):
        genes_names = True
        not_human = False
        ensembl_id = True

        expected_output = self.golden_expr
        result = calculate_gene_expression_values(self.counts_matrix, self.annotation, genes_names, not_human,
                                                  ensembl_id=ensembl_id)
        pd.testing.assert_frame_equal(result, expected_output)

    def test_gene_expression_gene_names(self):
        genes_names = True
        not_human = False
        ensembl_id = False

        expected_output = self.golden_expr
        result = calculate_gene_expression_values(self.counts_matrix, self.annotation, genes_names, not_human,
                                                  ensembl_id=ensembl_id)
        pd.testing.assert_frame_equal(result, expected_output)

    if __name__ == '__main__':
        unittest.main()
