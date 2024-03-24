import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from knowseqpy.feature_selection import random_forest


class TestRandomForestFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.mock_data = pd.DataFrame({
            "gene1": np.random.rand(10),
            "gene2": np.random.rand(10),
            "gene3": np.random.rand(10),
            "gene4": np.random.rand(10)
        })
        self.mock_labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.vars_selected = ["gene1", "gene2", "gene3", "gene4"]

    @patch("knowseqpy.feature_selection.random_forest.RandomForestClassifier")
    def test_random_forest_returns_correct_gene_count(self, mock_rf):
        mock_rf.return_value.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])

        all_genes = random_forest(self.mock_data, self.mock_labels, self.vars_selected)
        self.assertEqual(len(all_genes), len(self.vars_selected))

        max_genes = 2
        limited_genes = random_forest(self.mock_data, self.mock_labels, self.vars_selected, max_genes=max_genes)
        self.assertEqual(len(limited_genes), max_genes)


if __name__ == "__main__":
    unittest.main()
