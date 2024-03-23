import logging
import os
import unittest

import numpy as np
import pandas as pd

from knowseqpy.read_dge import read_dge
from knowseqpy.utils import csv_to_dataframe


class TestReadDge(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.counts_path = os.path.join(self.script_path, "test_fixtures", "count_files_breast")
        self.golden_dge = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "read_dge_counts.csv"], index_col=0, header=0)

    def test_read_count_file_successful(self):
        info_path = os.path.join(self.script_path, "test_fixtures", "samples_info_breast.csv")

        data_info_df = pd.read_csv(info_path, sep=",", dtype="str", usecols=["Internal.ID", "Sample.Type"])

        res_dge = read_dge(data_info=data_info_df, counts_path=self.counts_path)

        pd.testing.assert_frame_equal(self.golden_dge, res_dge, check_dtype=False, check_like=True)

    def test_duplicate_row_names_raises_value_error(self):
        data_info_duplicated_df = pd.DataFrame({
            "Internal.ID": ["test_duplicated_rows"],
            "Sample.Type": ["Solid Tissue Normal"]
        })

        with self.assertRaises(ValueError):
            read_dge(data_info=data_info_duplicated_df, counts_path=self.counts_path)

    def test_file_not_found_raises_file_not_found_error(self):
        data_info_missing_df = pd.DataFrame([{
            "Internal.ID": "missing_file",
            "Sample.Type": "Missing"
        }])
        with self.assertRaises(FileNotFoundError):
            read_dge(data_info=data_info_missing_df, counts_path=self.counts_path)

    def test_all_zeros(self):
        data_info_df_zeros = pd.DataFrame([{
            "Internal.ID": "test_all_zeros",
            "Sample.Type": "Zero Count Test"
        }])
        res_dge = read_dge(data_info=data_info_df_zeros, counts_path=self.counts_path)
        self.assertTrue(np.all(res_dge.values == 0))


if __name__ == "__main__":
    unittest.main()
