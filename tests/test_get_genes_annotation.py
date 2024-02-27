import logging
import os
import unittest

import pandas as pd

from knowseqpy.get_genes_annotation import get_genes_annotation
from knowseqpy.utils import csv_to_dataframe


class GetGenesAnnotationTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        """self.golden_annotation_38 = csv_to_dataframe(
            path_components=["test_fixtures", "golden_breast", "gene_annotations_38.csv"], index_col=0, header=0)"""

    def test_get_genes_annotation_valid_input_genome38(self):
        values = ["KRT19", "BRCA1"]
        atrb_filter = "external_gene_name"
        
        golden_data_df = pd.DataFrame({
            "ensembl_gene_id": ["ENSG00000171345", "ENSG00000012048"],
            "external_gene_name": ["KRT19", "BRCA1"],
            "percentage_gene_gc_content": [57.44, 44.09],
            "entrezgene_id": [3880, 672]
        })
        
        res_annotation = get_genes_annotation(values, attribute_filter=atrb_filter)
        res_annotation = res_annotation[golden_data_df.columns]

        self.assertIsInstance(res_annotation, pd.DataFrame)

        pd.testing.assert_frame_equal(golden_data_df, res_annotation.reset_index(drop=True),
                                      check_dtype=False, check_like=False)

    def test_get_genes_annotation_valid_input_genome37(self):
        values = ["KRT19", "BRCA1"]
        atrb_filter = "external_gene_name"
        reference_genome = 37

        golden_data_df = pd.DataFrame({
            "ensembl_gene_id": ["ENSG00000171345", "ENSG00000012048"],
            "external_gene_name": ["KRT19", "BRCA1"],
            "percentage_gene_gc_content": [57.44, 42.93],
            "entrezgene_id": [3880, 672]
        })

        res_annotation = get_genes_annotation(values, attribute_filter=atrb_filter, reference_genome=reference_genome)
        self.assertIsInstance(golden_data_df, pd.DataFrame)

        pd.testing.assert_frame_equal(golden_data_df, res_annotation.reset_index(drop=True), check_dtype=False)

    def test_get_genes_annotation_invalid_input_raises_ValueError(self):
        # Test with invalid values input
        values = "KRT19"
        attributes = ["ensembl_gene_id", "external_gene_name", "percentage_gene_gc_content", "entrezgene_id"]
        atrb_filter = "external_gene_name"
        not_human_dataset = ""
        reference_genome = 38

        with self.assertRaises(ValueError):
            get_genes_annotation(values, attributes, atrb_filter, not_human_dataset, reference_genome)

        # Test with invalid attributes input
        values = ["KRT19", "BRCA1"]
        attributes = "ensembl_gene_id"  # Not a list
        atrb_filter = "external_gene_name"
        not_human_dataset = ""
        reference_genome = 38

        with self.assertRaises(ValueError):
            get_genes_annotation(values, attributes, atrb_filter, not_human_dataset, reference_genome)


if __name__ == "__main__":
    unittest.main()
