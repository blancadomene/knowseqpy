import logging
import unittest

import pandas as pd

from knowseqpy.get_genes_annotation import get_genes_annotation


class GetGenesAnnotationTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    def test_get_genes_annotation_valid_input_genome38(self):
        values = ["KRT19", "BRCA1"]
        atrb_filter = "external_gene_name"

        golden_data_df = pd.DataFrame({
            "ensembl_gene_id": ["ENSG00000171345", "ENSG00000012048"],
            "external_gene_name": ["KRT19", "BRCA1"],
            "percentage_gene_gc_content": [57.44, 44.09],
            "entrezgene_id": [3880, 672]
        })
        golden_data_df.set_index("ensembl_gene_id", inplace=True)

        res_annotation = get_genes_annotation(values, attribute_filter=atrb_filter)
        res_annotation = res_annotation[golden_data_df.columns]

        self.assertIsInstance(res_annotation, pd.DataFrame)

        pd.testing.assert_frame_equal(golden_data_df, res_annotation,
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
        values = "KRT19"
        attributes = ["ensembl_gene_id", "external_gene_name", "percentage_gene_gc_content", "entrezgene_id"]
        atrb_filter = "external_gene_name"
        not_human_dataset = ""
        reference_genome = 38

        with self.assertRaises(ValueError):
            get_genes_annotation(values, attributes, atrb_filter, not_human_dataset, reference_genome)

    def test_get_genes_annotation_non_human_dataset(self):
        values = ["FBgn0267431", "FBgn0003360"]
        attributes = ["ensembl_gene_id", "external_gene_name"]
        atrb_filter = "ensembl_gene_id"
        not_human_dataset = "dmelanogaster_gene_ensembl"

        result_annotation = get_genes_annotation(values, attributes=attributes, attribute_filter=atrb_filter,
                                                 not_hsapiens_dataset=not_human_dataset)

        self.assertIsInstance(result_annotation, pd.DataFrame)
        self.assertTrue(not result_annotation.empty)
        self.assertIn("FBgn0267431", result_annotation["ensembl_gene_id"].values)

    def test_get_genes_annotation_non_human_dataset(self):
        values = ["FBgn0267431", "FBgn0003360"]  # Example Drosophila gene IDs
        attributes = ["ensembl_gene_id", "external_gene_name"]
        atrb_filter = "ensembl_gene_id"
        not_human_dataset = "dmelanogaster_gene_ensembl"

        result_annotation = get_genes_annotation(values, attributes=attributes, attribute_filter=atrb_filter,
                                                 not_hsapiens_dataset=not_human_dataset)

        self.assertIsInstance(result_annotation, pd.DataFrame)
        self.assertTrue(not result_annotation.empty)
        self.assertIn("FBgn0267431", result_annotation["ensembl_gene_id"].values)

    def test_get_genes_annotation_empty_result(self):
        values = ["NonExistentGene"]
        atrb_filter = "external_gene_name"

        result_annotation = get_genes_annotation(values, attribute_filter=atrb_filter)

        self.assertIsInstance(result_annotation, pd.DataFrame)
        self.assertTrue(result_annotation.empty)

    def test_get_genes_annotation_all_genome(self):
        values = ["allGenome"]
        atrb_filter = "ensembl_gene_id"
        expected_columns = {"ensembl_gene_id", "external_gene_name", "percentage_gene_gc_content", "entrezgene_id"}

        result_annotation = get_genes_annotation(values, attribute_filter=atrb_filter)

        self.assertIsInstance(result_annotation, pd.DataFrame)
        self.assertTrue(not result_annotation.empty)
        self.assertEqual(set(result_annotation.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
