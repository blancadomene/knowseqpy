import os
import unittest

import pandas as pd

from knowseq.get_genes_annotation import get_genes_annotation


class GetGenesAnnotationTest(unittest.TestCase):
    def setUp(self):
        golden_annot_38_path = os.path.normpath(os.path.join("test_fixtures", "golden", "gene_annotations_38.csv"))
        self.golden_annot_38 = pd.read_csv(golden_annot_38_path, index_col=0)

    def test_get_genes_annotation_validInput_genome38(self):
        values = ["KRT19", "BRCA1"]
        at_filter = "external_gene_name"
        result = get_genes_annotation(values, attribute_filter=at_filter)

        self.assertIsInstance(result, pd.DataFrame)

        # We reset the index since we don't care about it
        pd.testing.assert_frame_equal(self.golden_annot_38.reset_index(drop=True), result.reset_index(drop=True), check_dtype=False)

    def test_get_genes_annotation_validInput_genome37(self):
        file_path = os.path.normpath(os.path.join("test_fixtures", "golden", "gene_annotations_37.csv"))
        golden = pd.read_csv(file_path, index_col=0)
        values = ["KRT19", "BRCA1"]
        at_filter = "external_gene_name"
        reference_genome = 37

        result = get_genes_annotation(values, attribute_filter=at_filter, reference_genome=reference_genome)
        self.assertIsInstance(result, pd.DataFrame)

        # We reset the index since we don't care about it
        pd.testing.assert_frame_equal(golden.reset_index(drop=True), result.reset_index(drop=True), check_dtype=False)

    # TODO: Add tests for non human

    def test_get_genes_annotation_invalidInput_raisesValueError(self):
        # Test with invalid values input
        values = "KRT19"  # Not a list
        attributes = ["ensembl_gene_id", "external_gene_name", "percentage_gene_gc_content", "entrezgene_id"]
        at_filter = "external_gene_name"
        not_human_dataset = ""
        reference_genome = 38

        with self.assertRaises(ValueError):
            get_genes_annotation(values, attributes, at_filter, not_human_dataset, reference_genome)

        # Test with invalid attributes input
        values = ["KRT19", "BRCA1"]
        attributes = "ensembl_gene_id"  # Not a list
        at_filter = "external_gene_name"
        not_human_dataset = ""
        reference_genome = 38

        with self.assertRaises(ValueError):
            get_genes_annotation(values, attributes, at_filter, not_human_dataset, reference_genome)


if __name__ == "__main__":
    unittest.main()
