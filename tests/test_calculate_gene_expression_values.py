import logging
import unittest

import pandas as pd

from knowseq.calculate_gene_expression_values import calculate_gene_expression_values
from knowseq.utils import csv_to_dataframe


class CalculateGeneExpressionValuesTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.counts_matrix = csv_to_dataframe(
            path_components=["test_fixtures", "golden", "counts_matrix_breast.csv"], index_col=0, header=0)
        self.annotation = csv_to_dataframe(
            path_components=["test_fixtures", "annotation_breast.csv"], header=0, index_col="ensembl_gene_id")
        self.golden_gene_expression = csv_to_dataframe(
            path_components=["test_fixtures", "golden", "gene_expression_matrix_breast.csv"], header=0, index_col=0)

    def test_calculate_gene_expression(self):
        res_gene_expression = calculate_gene_expression_values(self.counts_matrix, self.annotation)
        res_gene_expression.to_csv('MY_gene_expression_breast.csv', index=True, header=True) # TODO

        pd.testing.assert_frame_equal(res_gene_expression, self.golden_gene_expression, check_dtype=False,
                                      check_like=True, check_exact=False, atol=0.1, rtol=0.1)

    if __name__ == '__main__':
        unittest.main()
