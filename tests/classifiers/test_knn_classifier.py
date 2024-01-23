import csv
import os
import unittest

import pandas as pd

from knowseq.classifiers import knn_classifier
from knowseq.utils.plotting_utils import plot_confusion_matrix


class KnnClassifierTest(unittest.TestCase):
    def setUp(self):
        golden_cqn_path = os.path.normpath(os.path.join("../test_fixtures", "golden", "cqn_breast.csv"))
        self.golden_cqn = pd.read_csv(golden_cqn_path, index_col=0)

    def test_kenn_classifier(self):
        degs_matrix_path = os.path.normpath(os.path.join("../test_fixtures", "golden", "degs_matrix_breast.csv"))
        golden_degs_matrix = pd.read_csv(degs_matrix_path, index_col=0, header=0)
        golden_degs_matrix_transposed = golden_degs_matrix.transpose()

        quality_labels_path = os.path.normpath(
            os.path.join("../test_fixtures", "golden", "qa_labels_breast.csv"))
        quality_labels = pd.read_csv(quality_labels_path, header=None).iloc[:, 0]

        fs_ranking_path = os.path.normpath(
            os.path.join("../test_fixtures", "golden", "fs_ranking_mrmr_breast.csv"))

        with open(fs_ranking_path, newline='') as f:
            reader = csv.reader(f)
            fs_ranking = list(reader)
            fs_ranking_list = [element[0] for element in fs_ranking]

        knn_res = knn_classifier(golden_degs_matrix_transposed, quality_labels, fs_ranking_list)

        plot_confusion_matrix(knn_res["confusion_matrix"], class_labels=knn_res["unique_labels"].tolist())

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        """pd.testing.assert_frame_equal(self.golden_cqn, cqn_values, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)"""


if __name__ == '__main__':
    unittest.main()
