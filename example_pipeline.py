"""
Example pipeline using breast samples.
"""

import logging
import os
from datetime import datetime

from knowseqpy.calculate_gene_expression_values import calculate_gene_expression_values
from knowseqpy.counts_to_matrix import counts_to_matrix
from knowseqpy.get_genes_annotation import get_genes_annotation
from knowseqpy.rna_seq_qa import rna_seq_qa


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        filename=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_knowseq_logs.log",
        filemode="w"
    )

    # Set and read paths
    script_path = os.path.dirname(os.path.abspath(__file__))
    info_path = os.path.join(script_path, "tests", "test_fixtures", "samples_info_breast.csv")
    counts_path = os.path.join(script_path, "tests", "test_fixtures", "breast_count_files")

    # Execute counts to matrix conversion
    counts_df, labels_ser = counts_to_matrix(info_path=info_path, counts_path=counts_path)

    # Number of samples per class
    print(labels_ser.value_counts())

    gene_annotation_df = get_genes_annotation(values=counts_df.index)

    print(gene_annotation_df)

    gene_expression_df = calculate_gene_expression_values(counts_df, gene_annotation_df)

    qa_df, qa_labels = rna_seq_qa(gene_expression_df)


if __name__ == '__main__':
    main()
