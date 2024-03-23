import logging
import unittest

import numpy as np
import pandas as pd

from knowseqpy.outlier_detection import kolmogorov_smirnov


class TestKolmogorovSmirnov(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        np.random.seed(1234)

    def test_no_outliers(self):
        no_outlier_data = pd.DataFrame({
            "Sample1": np.random.normal(0, 1, 100),
            "Sample2": np.random.normal(0, 1, 100),
        })
        outliers = kolmogorov_smirnov(no_outlier_data)
        self.assertEqual(len(outliers), 0)

    def test_empty_dataframe(self):
        with self.assertRaises(ValueError):
            kolmogorov_smirnov(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
