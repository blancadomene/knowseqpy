import os
import unittest

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
        genes_length_path = os.path.normpath(os.path.join("..", "external_data", "genes_length_homo_sapiens.csv"))
        genes_length = pd.read_csv(genes_length_path, header=0, index_col="Gene_stable_ID")

        # Remove duplicates from annotation (we will only need (ensembl_gene_id,percentage_gene_gc_content) pairs)
        genes_annot = annotation.groupby(annotation.index).first()

        # We get the annotations and length only for those genes that are present in counts_matrix
        # Also we only take the column for each variable we will use later on
        common_joined = genes_annot.join(counts_matrix, how="inner").join(genes_length, how="inner")
        common_genes_annot = common_joined["percentage_gene_gc_content"]
        common_gene_length = common_joined["Gene_length"]
        common_counts_matrix = common_joined.drop(columns=list(genes_annot.columns) + list(genes_length.columns))

        res_cqn_y, res_cqn_offset = cqn(counts=common_counts_matrix,
                                        x=common_genes_annot,
                                        lengths=common_gene_length,
                                        size_factors=counts_matrix.sum(axis=0),
                                        tau=0.5,
                                        sqn=True)
        # cqn_values = res_cqn_y + res_cqn_offset
        res_cqn_offset = res_cqn_offset[res_cqn_y.columns]
        # pd.options.display.float_format = '{:.12f}'.format
        pd.set_option("display.precision", 14)
        cqn_values = res_cqn_y.add(res_cqn_offset, fill_value=0)

        # Check that both dataframes contain the same data, but ignoring the dtype and order of rows and columns
        pd.testing.assert_frame_equal(self.golden_cqn, cqn_values, check_dtype=False, check_like=True,
                                      check_exact=False,
                                      atol=0.1, rtol=0.1)


if __name__ == '__main__':
    unittest.main()
