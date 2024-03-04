import logging
import os
import unittest

import pandas as pd

from src.normalization import cqn
from src.utils import csv_to_dataframe


class CqnTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        self.golden_cqn = csv_to_dataframe(
            path_components=["../test_fixtures", "golden_breast", "cqn.csv"], index_col=0, header=0)

    def test_cqn_values(self):
        counts_matrix_path = os.path.normpath(os.path.join("../test_fixtures", "golden_breast", "counts_matrix.csv"))
        counts_matrix = pd.read_csv(counts_matrix_path, header=0, index_col=0)
        annotation_path = os.path.normpath(os.path.join("../test_fixtures", "gene_annotation_breast.csv"))
        annotation = pd.read_csv(annotation_path, header=0, index_col="ensembl_gene_id")
        gene_length_path = os.path.normpath(os.path.join("../..", "external_data", "genes_length_homo_sapiens.csv"))
        gene_length = pd.read_csv(gene_length_path, header=0, index_col="Gene_stable_ID")

        # Remove duplicates from annotation (we will only need (ensembl_gene_id,percentage_gene_gc_content) pairs)
        genes_annot = annotation.groupby(annotation.index).first()

        # We get the annotations and length only for those genes that are present in counts_matrix
        # Also we only take the column for each variable we will use later on
        common_joined = genes_annot.join(counts_matrix, how="inner").join(gene_length, how="inner")
        common_genes_annot = common_joined["percentage_gene_gc_content"]
        common_gene_length = common_joined["Gene_length"]
        common_counts_matrix = common_joined.drop(columns=list(genes_annot.columns) + list(gene_length.columns))

        res_cqn_y, res_cqn_offset = cqn(counts=common_counts_matrix,
                                        x=common_genes_annot,
                                        lengths=common_gene_length,
                                        size_factors=counts_matrix.sum(axis=0),
                                        tau=0.5,
                                        sqn=True)
        cqn_values = res_cqn_y + res_cqn_offset

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(self.golden_cqn, cqn_values, check_dtype=False, check_like=True,
                                      check_exact=False, atol=0.1, rtol=0.1)


if __name__ == '__main__':
    unittest.main()
