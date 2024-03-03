"""
Example pipeline using breast samples.
"""

import logging
import os
from datetime import datetime

from knowseqpy.batch_effect_removal import batch_effect_removal
from knowseqpy.calculate_gene_expression_values import calculate_gene_expression_values
from knowseqpy.classifiers import knn
from knowseqpy.counts_to_matrix import counts_to_matrix
from knowseqpy.degs_extraction import degs_extraction
from knowseqpy.feature_selection import feature_selection
# from knowseqpy.feature_selection import feature_selection
from knowseqpy.get_genes_annotation import get_genes_annotation
from knowseqpy.rna_seq_qa import rna_seq_qa
from knowseqpy.utils import plot_boxplot, plot_confusion_matrix, plot_samples_heatmap


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
    counts_path = os.path.join(script_path, "tests", "test_fixtures", "count_files_breast")

    # Execute counts to matrix conversion
    counts_df, labels_ser = counts_to_matrix(info_path=info_path, counts_path=counts_path)

    # Number of samples per class
    print(labels_ser.value_counts())

    gene_annotation_df = get_genes_annotation(values=counts_df.index)

    print(gene_annotation_df)

    gene_expression_df = calculate_gene_expression_values(counts_df, gene_annotation_df)

    qa_df, outliers = rna_seq_qa(gene_expression_df)
    qa_labels = labels_ser.drop(outliers)

    batch_df = batch_effect_removal(qa_df, labels=qa_labels, method="sva")

    degs_df = degs_extraction(batch_df, labels=qa_labels, lfc=3.5, p_value=0.001)[0]

    fs_ranking = feature_selection(data=degs_df, labels=qa_labels, mode="mrmr", vars_selected=degs_df.columns.tolist())

    knn_res = knn(degs_df, qa_labels, fs_ranking)

    plot_boxplot(degs_df, qa_labels, fs_ranking, top_n_features=3)
    plot_confusion_matrix(knn_res["confusion_matrix"], unique_labels=knn_res["unique_labels"].tolist())
    plot_samples_heatmap(degs_df, qa_labels, fs_ranking, top_n_features=4)


if __name__ == '__main__':
    main()
