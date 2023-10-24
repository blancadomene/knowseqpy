import unittest

import os
import pandas as pd

from knowseq.normalization import cqn


class CqnTest(unittest.TestCase):
    def setUp(self):
        golden_cqn_path = os.path.normpath(os.path.join("test_fixtures", "golden", "cqn_breast.csv"))
        self.golden_cqn = pd.read_csv(golden_cqn_path, index_col=0)

    def test_cqn_values(self):
        counts_matrix_path = os.path.normpath(os.path.join("test_fixtures", "golden", "counts_matrix_breast.csv"))
        counts_matrix = pd.read_csv(counts_matrix_path, header=0, index_col=0)
        annotation_path = os.path.normpath(os.path.join("test_fixtures", "annotation_breast.csv"))
        annotation = pd.read_csv(annotation_path, header=0, index_col="ensembl_gene_id")
        gene_length_path = os.path.normpath(os.path.join("..", "external_data", "genes_length_homo_sapiens.csv"))
        gene_length = pd.read_csv(gene_length_path, header=0, index_col="Gene_stable_ID")

        # Remove duplicates from annotation
        genes_annot = annotation.groupby(annotation.index).first()

        # We get the annotations and length only for those genes that are present in counts_matrix
        # Also we only take the column for each variable we will use later on
        common_joined = genes_annot.join(counts_matrix, how="inner").join(gene_length, how="inner")
        genes_annot = common_joined["percentage_gene_gc_content"]
        gene_length = common_joined["Gene_length"]
        counts_matrix = common_joined.drop(columns=["percentage_gene_gc_content", "Gene_length"])

        res_cqn = cqn(counts=counts_matrix,
                      x=genes_annot,
                      lengths=gene_length,
                      size_factors=counts_matrix.sum(axis=1),
                      tau=0.5,
                      sqn=True)
        cqn_values = res_cqn['y'] + res_cqn['offset']

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(self.golden_cqn, cqn_values, check_dtype=False, check_like=True)


if __name__ == '__main__':
    unittest.main()
