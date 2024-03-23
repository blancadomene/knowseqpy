import logging
import unittest

import numpy as np
import pandas as pd

from knowseqpy.outlier_detection import median_deviation


class TestMedianDeviation(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        np.random.seed(1234)
        self.data = pd.DataFrame({
            f"Sample{i}": np.random.normal(i, 1, 100) for i in range(10)
        })
        self.data["SampleOutlier"] = np.random.normal(100, 1, 100)

    def test_detect_outlier(self):
        outliers = median_deviation(self.data)
        self.assertIn("SampleOutlier", outliers)

    def test_no_outliers(self):
        no_outlier_data = self.data.drop(columns=["SampleOutlier"])
        outliers = median_deviation(no_outlier_data)
        self.assertEqual(len(outliers), 0)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        outliers = median_deviation(empty_df)
        self.assertEqual(len(outliers), 0)


if __name__ == "__main__":
    unittest.main()
