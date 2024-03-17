"""
Example pipeline using breast samples.
"""

from pathlib import Path

from knowseqpy import (counts_to_dataframe, get_genes_annotation, calculate_gene_expression_values, rna_seq_qa,
                       batch_effect_removal, degs_extraction, feature_selection)
from knowseqpy.classifiers import knn
from knowseqpy.utils import plot_boxplot, plot_confusion_matrix, plot_samples_heatmap


def main():
    # Set and read paths
    script_path = Path(__file__).resolve().parent.parent
    info_path = script_path / "tests" / "test_fixtures" / "samples_info_breast.csv"
    counts_path = script_path / "tests" / "test_fixtures" / "count_files_breast"

    # Execute counts to matrix conversion
    counts, labels = counts_to_dataframe(info_path=info_path, counts_path=counts_path)

    # Number of samples per class
    print(labels.value_counts())

    gene_annotation = get_genes_annotation(values=counts.index)

    print(gene_annotation)

    gene_expression = calculate_gene_expression_values(counts, gene_annotation)

    qa, outliers = rna_seq_qa(gene_expression)
    qa_labels = labels.drop(outliers)

    batch = batch_effect_removal(qa, labels=qa_labels, method="sva")

    degs = degs_extraction(batch, labels=qa_labels, lfc=3.5, p_value=0.001)[0].transpose()
    selected_features = feature_selection(data=degs, labels=qa_labels, mode="da",
                                          vars_selected=degs.columns.tolist())

    knn_res = knn(degs, qa_labels, selected_features)

    plot_boxplot(degs, qa_labels, selected_features, top_n_features=3)
    plot_confusion_matrix(knn_res["confusion_matrix"], unique_labels=knn_res["unique_labels"].tolist())
    plot_samples_heatmap(degs, qa_labels, selected_features, top_n_features=4)


if __name__ == '__main__':
    main()
