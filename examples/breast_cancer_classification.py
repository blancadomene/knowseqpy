"""
Example pipeline using breast samples.
"""

from pathlib import Path
import random

from knowseqpy import (calculate_gene_expression_values, counts_to_dataframe, degs_extraction, get_genes_annotation,
                       rna_seq_qa)
from knowseqpy.batch_effect import sva
from knowseqpy.classifiers import knn
from knowseqpy.feature_selection import discriminant_analysis
from knowseqpy.utils import plot_boxplot, plot_confusion_matrix, plot_samples_heatmap


def main():
    # Set seed for reproducible results
    random.seed(1234)

    # Set and read paths
    script_path = Path(__file__).resolve().parent.parent
    info_path = script_path / "tests" / "test_fixtures" / "samples_info_breast.csv"
    counts_path = script_path / "tests" / "test_fixtures" / "count_files_breast"

    # Load and preprocess count files to create a counts df
    counts, labels = counts_to_dataframe(info_path=info_path, counts_path=counts_path)

    #  Number of samples per class to understand the dataset's distribution
    print(labels.value_counts())

    # Annotate genes: Fetch gene annotations for the genes in the dataset
    gene_annotation = get_genes_annotation(values=counts.index)

    print(gene_annotation)

    # Normalize and calculate expression values from counts
    gene_expression = calculate_gene_expression_values(counts=counts, gene_annotation=gene_annotation)

    # Perform QA to identify and remove outliers from the dataset
    cleaned_df, outliers = rna_seq_qa(gene_expression_df=gene_expression)
    qa_labels = labels.drop(outliers)

    # Apply chosen method to correct for batch effects in the data
    batch = sva(expression_df=cleaned_df, labels=qa_labels)

    # Identify differentially expressed genes with specified criteria
    degs = degs_extraction(batch, labels=qa_labels, lfc=3.5, p_value=0.001)[0].transpose()

    # Select features (genes) for downstream analysis based on specified criteria
    selected_features = discriminant_analysis(data=degs, labels=qa_labels, vars_selected=degs.columns.tolist())

    # Classifier training and prediction
    knn_res = knn(degs, qa_labels, selected_features)

    # Visualization
    plot_boxplot(data=degs, labels=qa_labels, fs_ranking=selected_features, top_n_features=3)
    plot_confusion_matrix(conf_matrix=knn_res["confusion_matrix"], unique_labels=knn_res["unique_labels"].tolist())
    plot_samples_heatmap(data=degs, labels=qa_labels, fs_ranking=selected_features, top_n_features=4)


if __name__ == "__main__":
    main()
