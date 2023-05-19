import unittest
from unittest.mock import patch
import pandas as pd
from get_genes_annotation import get_genes_annotation
import requests

class get_genes_annotation_test(unittest.TestCase):
    def test_get_genes_annotation_validInput_genome38(self):
        golden = pd.read_csv("test_fixtures/golden/gene_annotations_38.csv", index_col=0)
        values = ["KRT19", "BRCA1"]
        filter = "external_gene_name"

        result = get_genes_annotation(values, filter=filter)
        self.assertIsInstance(result, pd.DataFrame)

        # We reset the index since we don't care about it
        pd.testing.assert_frame_equal(golden.reset_index(drop=True), result.reset_index(drop=True), check_dtype=False)

    def test_get_genes_annotation_validInput_genome37(self):
        golden = pd.read_csv("test_fixtures/golden/gene_annotations_37.csv", index_col=0)
        values = ["KRT19", "BRCA1"]
        filter = "external_gene_name"
        referenceGenome = 37

        result = get_genes_annotation(values, filter=filter, referenceGenome=referenceGenome)
        self.assertIsInstance(result, pd.DataFrame)

        # We reset the index since we don't care about it
        pd.testing.assert_frame_equal(golden.reset_index(drop=True), result.reset_index(drop=True), check_dtype=False)

    # TODO: Add tests for non human

    def test_get_genes_annotation_invalidInput_raisesValueError(self):
        # Test with invalid values input
        values = "KRT19"  # Not a list
        attributes = ["ensembl_gene_id", "external_gene_name", "percentage_gene_gc_content", "entrezgene_id"]
        filter = "external_gene_name"
        notHSapiens = False
        notHumandataset = ""
        referenceGenome = 38

        with self.assertRaises(ValueError):
            get_genes_annotation(values, attributes, filter, notHSapiens, notHumandataset, referenceGenome)

        # Test with invalid attributes input
        values = ["KRT19", "BRCA1"]
        attributes = "ensembl_gene_id"  # Not a list
        filter = "external_gene_name"
        notHSapiens = False
        notHumandataset = ""
        referenceGenome = 38

        with self.assertRaises(ValueError):
            get_genes_annotation(values, attributes, filter, notHSapiens, notHumandataset, referenceGenome)

        # TODO: Add more test cases for other parameters if needed

if __name__ == "__main__":
    unittest.main()
