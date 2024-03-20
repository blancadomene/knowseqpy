import logging
import unittest
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from knowseqpy.classifiers import knn
from knowseqpy.utils import csv_to_dataframe, csv_to_list, get_test_path
from knowseqpy.utils.plotting import plot_samples_heatmap, plot_confusion_matrix, plot_boxplot


class KnnClassifierTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.test_path = get_test_path()

    def test_kenn_classifier(self):
        golden_degs_df = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "degs_matrix.csv"],
            index_col=0, header=0).transpose()

        quality_labels = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "qa_labels.csv"],
            header=None).iloc[:, 0]

        fs_ranking = csv_to_list(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "fs_ranking_mrmr.csv"])
        fs_ranking_list = [row[0] for row in fs_ranking if row]

        knn_res = knn(golden_degs_df, quality_labels, fs_ranking_list)

        plot_boxplot(golden_degs_df, quality_labels, fs_ranking_list, top_n_features=3)
        plot_confusion_matrix(knn_res["confusion_matrix"], unique_labels=knn_res["unique_labels"].tolist())
        plot_samples_heatmap(golden_degs_df, quality_labels, fs_ranking_list, top_n_features=4)

        x_train = pd.DataFrame(golden_degs_df).apply(pd.to_numeric, errors="coerce").fillna(0)
        x_train = x_train[fs_ranking_list]
        x_train = StandardScaler().fit_transform(x_train)

        label_codes, unique_labels = pd.factorize(quality_labels)
        model = knn_res["model"]
        y_pred = model.predict(x_train)
        y_train = label_codes

        acc = accuracy_score(y_train, y_pred)
        self.assertGreaterEqual(acc, 0.95)

    def test_knn_small_dataset(self):
        data, labels, vars_selected = self.load_small_dataset()
        results = knn(data, labels, vars_selected)
        self.assertTrue(results['model'].n_neighbors <= len(data))


if __name__ == "__main__":
    unittest.main()
