import logging
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from knowseqpy.classifiers import knn
from knowseqpy.utils import csv_to_dataframe, csv_to_list, get_test_path


class TestKnnClassifier(unittest.TestCase):
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

        x_train = pd.DataFrame(golden_degs_df).apply(pd.to_numeric, errors="coerce").fillna(0)
        x_train = x_train[fs_ranking_list]
        x_train = StandardScaler().fit_transform(x_train)

        label_codes, unique_labels = pd.factorize(quality_labels)
        model = knn_res["model"]
        y_pred = model.predict(x_train)
        y_train = label_codes

        acc = accuracy_score(y_train, y_pred)
        self.assertGreaterEqual(acc, 0.95)

    def test_different_cv_strategies(self):
        data = pd.DataFrame(np.random.rand(100, 10), columns=[f"Gene{i}" for i in range(10)])
        labels = pd.Series(np.random.randint(2, size=100))
        vars_selected = list(data.columns)[:5]
        cv_strategies = [KFold(n_splits=5), RepeatedStratifiedKFold(n_splits=5, n_repeats=2)]

        for cv_strategy in cv_strategies:
            result = knn(data, labels, vars_selected, cv_strategy=cv_strategy)
            self.assertIn("accuracy", result)
            self.assertIn("model", result)
            self.assertIn("confusion_matrix", result)
            self.assertTrue(result["accuracy"] >= 0)

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        labels = pd.Series(dtype="int")
        vars_selected = []

        with self.assertRaises(ValueError):
            knn(data, labels, vars_selected)

    def test_invalid_vars_selected(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=[f"Gene{i}" for i in range(5)])
        labels = pd.Series(np.random.randint(2, size=10))
        vars_selected = ["Gene5", "Gene6"]

        with self.assertRaises(KeyError):
            knn(data, labels, vars_selected)


if __name__ == "__main__":
    unittest.main()
