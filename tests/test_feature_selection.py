import logging
import unittest

import pandas as pd

from knowseqpy.feature_selection import feature_selection
from knowseqpy.utils import csv_to_dataframe


class FeatureSelectionTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.fs_ranking_golden = csv_to_dataframe(
            path_components=["test_fixtures", "golden", "fs_ranking_mrmr_breast.csv"])

    def test_feature_selection(self):
        golden_degs_matrix = csv_to_dataframe(path_components=["test_fixtures", "golden", "degs_matrix_breast.csv"],
                                              index_col=0, header=0).transpose()
        quality_labels = csv_to_dataframe(["test_fixtures", "golden", "qa_labels_breast.csv"]).iloc[:, 0]

        fs_ranking = feature_selection(data=golden_degs_matrix, labels=quality_labels, mode="da",
                                       vars_selected=golden_degs_matrix.columns.tolist())

        fs_ranking_df = pd.DataFrame(fs_ranking)
        fs_ranking_df.to_csv('datos.csv', index=False, header=False)

        self.assertTrue(fs_ranking_df.iloc[:, 0].equals(self.fs_ranking_golden.iloc[:, 0]))

    if __name__ == '__main__':
        unittest.main()
