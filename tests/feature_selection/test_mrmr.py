import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from knowseqpy.feature_selection import mrmr


class TestMrmr(unittest.TestCase):
    def setUp(self):
        self.mock_data = pd.DataFrame({
            "gene1": np.random.rand(10),
            "gene2": np.random.rand(10),
            "gene3": np.random.rand(10)
        })
        self.mock_labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.vars_selected = ["gene1", "gene2"]

    @patch("knowseqpy.feature_selection.mrmr.mrmr_classif")
    def test_mrmr_calls_mrmr_classif_correctly(self, mock_mrmr_classif):
        mock_mrmr_classif.return_value = ["gene1", "gene2"]

        expected_max_genes = len(self.vars_selected)

        result = mrmr(self.mock_data, self.mock_labels, self.vars_selected)

        mock_mrmr_classif.assert_called_once()
        args, kwargs = mock_mrmr_classif.call_args

        self.assertTrue("X" in kwargs and "y" in kwargs)
        self.assertEqual(kwargs["K"], expected_max_genes)
        self.assertEqual(kwargs["relevance"], "f")
        self.assertEqual(kwargs["redundancy"], "c")
        self.assertEqual(kwargs["denominator"], "mean")
        self.assertEqual(result, ["gene1", "gene2"])


if __name__ == "__main__":
    unittest.main()
