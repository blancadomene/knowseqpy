import logging
import unittest

import pandas as pd

from src.calculate_gene_expression_values import calculate_gene_expression_values
from src.utils import csv_to_dataframe


class CalculateGeneExpressionValuesTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.counts_df = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "counts_matrix.csv"], index_col=0, header=0)
        self.gene_annotation = csv_to_dataframe(
            path_components=["test_fixtures", "gene_annotation_breast.csv"], header=0, index_col="ensembl_gene_id")
        self.golden_gene_expression = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "gene_expression_matrix.csv"], header=0, index_col=0)

    # TODO: fix
    def test_calculate_gene_expression(self):
        res_gene_expression = calculate_gene_expression_values(self.counts_df, self.gene_annotation)

        pd.testing.assert_frame_equal(res_gene_expression, self.golden_gene_expression, check_dtype=False,
                                      check_like=True, check_exact=False, atol=0.1, rtol=0.1)

    def test_gene_names_flag(self):
        # Test with gene names
        res_with_gene_names = calculate_gene_expression_values(self.counts_df, self.gene_annotation, genes_names=True)
        # Ensure that the index consists of gene names
        self.assertTrue(all(isinstance(name, str) for name in res_with_gene_names.index))

        # Test without gene names (use Ensembl IDs)
        res_with_ensembl_ids = calculate_gene_expression_values(self.counts_df, self.gene_annotation, genes_names=False)
        # Ensure that the index consists of Ensembl IDs
        self.assertTrue(all(isinstance(name, str) for name in res_with_ensembl_ids.index))

    def test_invalid_gene_length_path(self):
        with self.assertRaises(FileNotFoundError):
            calculate_gene_expression_values(self.counts_df, self.gene_annotation,
                                             not_human_gene_length_csv="non_existent_path.csv")

    def test_missing_annotation_columns(self):
        modified_annotation = self.gene_annotation.drop(columns=['percentage_gene_gc_content'], errors='ignore')
        with self.assertRaises(KeyError):
            calculate_gene_expression_values(self.counts_df, modified_annotation)

    if __name__ == '__main__':
        unittest.main()
