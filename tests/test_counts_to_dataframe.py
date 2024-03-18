import logging
import unittest
from unittest.mock import patch

import pandas as pd

from knowseqpy.counts_to_dataframe import counts_to_dataframe
from knowseqpy.utils import csv_to_dataframe, get_test_path


class CountsToMatrixTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.test_path = get_test_path()
        self.golden_counts = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "counts.csv"], index_col=0, header=0)
        self.golden_labels = csv_to_dataframe(
            path_components=[self.test_path, "test_fixtures", "golden_breast", "counts_labels.csv"], index_col=0,
            header=0)[
            "Sample.Type"]

    def test_file_not_exists(self):
        info_path = self.test_path / "file" / "doesnt" / "exist.csv"
        self.assertRaises(FileNotFoundError, counts_to_dataframe, info_path, "some_other_path")

    def test_missing_cols_csv(self):
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                "Wrong.Column": ["sample1"],
                "Another.Wrong.Column": ["type1"]
            })

            with self.assertRaises(Exception):
                counts_to_dataframe(info_path="dummy_path.csv", counts_path="dummy_counts_path")

    def test_valid_csv(self):
        info_path = self.test_path / "test_fixtures" / "samples_info_breast.csv"
        counts_path = self.test_path / "test_fixtures" / "count_files_breast"

        counts, labels = counts_to_dataframe(info_path=info_path, counts_path=counts_path)

        pd.testing.assert_frame_equal(self.golden_counts, counts, check_dtype=False)
        pd.testing.assert_series_equal(self.golden_labels, labels, check_dtype=False)

    def test_row_filtering_logic(self):
        info_path = self.test_path / "test_fixtures" / "samples_info_breast.csv"
        counts_path = self.test_path / "test_fixtures" / "count_files_breast"
        rows_to_skip = ["__no_feature", "__ambiguous"]
        counts, labels = counts_to_dataframe(info_path=info_path, counts_path=counts_path, rows_to_skip=rows_to_skip)
        for row in rows_to_skip:
            self.assertNotIn(row, counts.index)

    def test_gene_id_truncation(self):
        info_path = self.test_path / "test_fixtures" / "samples_info_breast.csv"
        counts_path = self.test_path / "test_fixtures" / "count_files_breast"
        counts, _ = counts_to_dataframe(info_path=info_path, counts_path=counts_path)
        for gene_id in counts.index:
            self.assertNotIn(".", gene_id, "Gene IDs should be truncated before the period.")


if __name__ == "__main__":
    unittest.main()
