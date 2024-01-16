import os
import unittest

import pandas as pd

from knowseq.feature_selection import feature_selection


class FeatureSelectionTest(unittest.TestCase):
    def setUp(self):
        fs_ranking_path = os.path.normpath(
            os.path.join("test_fixtures", "golden", "fs_ranking_mrmr_breast.csv"))
        self.fs_ranking_golden = pd.read_csv(fs_ranking_path, header=None)

    def test_rna_seq_qa(self):
        degs_matrix_path = os.path.normpath(os.path.join("test_fixtures", "golden", "degs_matrix_breast.csv"))
        golden_degs_matrix = pd.read_csv(degs_matrix_path, index_col=0, header=0)
        golden_degs_matrix_transposed = golden_degs_matrix.transpose()

        quality_labels_path = os.path.normpath(
            os.path.join("test_fixtures", "golden", "qa_labels_breast.csv"))
        quality_labels = pd.read_csv(quality_labels_path, header=None).iloc[:, 0]

        fs_ranking = feature_selection(data=golden_degs_matrix_transposed, labels=quality_labels, mode="mrmr",
                                       vars_selected=golden_degs_matrix_transposed.columns.tolist())

        fs_ranking_df = pd.DataFrame(fs_ranking)
        # fs_ranking_df.to_csv('datos.csv', index=False, header=False)

        self.assertTrue(fs_ranking_df.iloc[:, 0].equals(self.fs_ranking_golden.iloc[:, 0]))

    if __name__ == '__main__':
        unittest.main()
