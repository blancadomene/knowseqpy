import unittest

import pandas as pd

from calculate_gene_expression_values import calculate_gene_expression_values


class CalculateGeneExpressionValuesTest(unittest.TestCase):
    def test_calculate_gene_expression_values(self):
        golden_gene_expression = pd.read_csv("../test_fixtures/golden/gene_expression_matrix_breast.csv", index_col=0)

    if __name__ == '__main__':
        unittest.main()
