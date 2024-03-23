import logging
import unittest

import pandas as pd

from knowseqpy.feature_selection import discriminant_analysis
from knowseqpy.utils import csv_to_dataframe, get_test_path


class TestDiscriminantAnalysis(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.test_path = get_test_path()
        self.golden_degs_matrix = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "degs_matrix.csv"],
            index_col=0, header=0).transpose()
        self.quality_labels = csv_to_dataframe(
            [self.test_path, "test_fixtures", "golden_breast", "qa_labels.csv"]).iloc[:, 0]
        self.fs_ranking_golden = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "fs_ranking_mrmr.csv"])

    # TODO: Fix
    def test_discriminant_analysis(self):
        da_fs_ranking = discriminant_analysis(data=self.golden_degs_matrix, labels=self.quality_labels,
                                              vars_selected=self.golden_degs_matrix.columns.tolist())

        da_ranking_df = pd.DataFrame(da_fs_ranking)

        self.assertTrue(da_ranking_df.iloc[:, 0].equals(self.fs_ranking_golden.iloc[:, 0]))

    def test_max_genes_limit(self):
        max_genes = 10
        ranked_genes = discriminant_analysis(data=self.golden_degs_matrix, labels=self.quality_labels,
                                             vars_selected=self.golden_degs_matrix.columns.tolist(),
                                             max_genes=max_genes)

        self.assertEqual(len(ranked_genes), max_genes,
                         "The number of genes returned should match the max_genes parameter.")

    def test_vars_selected_validity(self):
        vars_selected = ["GeneNotInData1", "GeneNotInData2"]

        with self.assertRaises(KeyError):
            discriminant_analysis(self.golden_degs_matrix, self.quality_labels, vars_selected)

    def test_max_genes_exceeding_available(self):
        vars_selected = self.golden_degs_matrix.columns.tolist()
        max_genes = len(vars_selected) + 10

        ranked_genes = discriminant_analysis(data=self.golden_degs_matrix, labels=self.quality_labels,
                                             vars_selected=vars_selected, max_genes=max_genes)
        self.assertLessEqual(len(ranked_genes), len(vars_selected), "Should not return more genes than available.")

    if __name__ == "__main__":
        unittest.main()
