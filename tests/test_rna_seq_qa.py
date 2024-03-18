import logging
import unittest

import pandas as pd

from knowseqpy.rna_seq_qa import rna_seq_qa
from knowseqpy.utils import csv_to_dataframe, get_test_path


class RnaSeqQaTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.test_path = get_test_path()
        self.golden_qa = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "qa_matrix.csv"], index_col=0, header=0)
        self.gene_expression = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "gene_expression_matrix.csv"],
            index_col=0, header=0)

    def test_rna_seq_qa(self):
        res_qa, _ = rna_seq_qa(self.gene_expression)
        pd.testing.assert_frame_equal(res_qa, self.golden_qa, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)

    def test_identification_of_known_outliers(self):
        known_outliers = ["A1EN-RNA-Tumor.count", "A0DB-RNA-Tumor.count", "A0C3-RNA-Normal.count",
                          "A13E-RNA-Tumor.count", "A2C8-RNA-Normal.count"]
        _, outliers_detected = rna_seq_qa(self.gene_expression)

        self.assertTrue(all(outlier in outliers_detected for outlier in known_outliers))

    def test_no_outliers_scenario(self):
        known_outliers = ["A1EN-RNA-Tumor.count", "A0DB-RNA-Tumor.count", "A0C3-RNA-Normal.count",
                          "A13E-RNA-Tumor.count", "A2C8-RNA-Normal.count", "A204-RNA-Tumor.count"]
        gene_expression_no_outliers = self.gene_expression.copy()
        gene_expression_no_outliers = gene_expression_no_outliers.drop(columns=known_outliers)

        _, outliers_detected = rna_seq_qa(gene_expression_no_outliers)
        self.assertEqual(len(outliers_detected), 0)

    def test_consistency_across_runs(self):
        first_run_results, first_run_outliers = rna_seq_qa(self.gene_expression)
        second_run_results, second_run_outliers = rna_seq_qa(self.gene_expression)

        pd.testing.assert_frame_equal(first_run_results, second_run_results, check_dtype=False)
        self.assertListEqual(first_run_outliers, second_run_outliers)

    if __name__ == "__main__":
        unittest.main()
