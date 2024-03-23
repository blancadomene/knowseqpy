import logging
import unittest

import numpy as np
import pandas as pd

from knowseqpy.outlier_detection import manhattan_distance


class TestManhattanDistance(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        np.random.seed(1234)
        normal_data = np.random.normal(0, 1, (100, 10))
        outlier_data = np.random.normal(0, 10, (100, 1))
        self.data = pd.DataFrame(np.hstack((normal_data, outlier_data))).transpose()

    def test_detect_outlier(self):
        outliers = manhattan_distance(self.data)
        self.assertIn(79, outliers)

    def test_no_outliers(self):
        no_outlier_data = np.random.normal(0, 1, (100, 10))
        no_outlier_df = pd.DataFrame(no_outlier_data).transpose()
        outliers = manhattan_distance(no_outlier_df)
        self.assertEqual(len(outliers), 0)


if __name__ == "__main__":
    unittest.main()
