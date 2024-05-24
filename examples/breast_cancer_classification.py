"""
Example pipeline using breast samples.
"""
import os
import random
from pathlib import Path

from knowseqpy import (calculate_gene_expression_values, counts_to_dataframe, degs_extraction, get_genes_annotation,
                       rna_seq_qa)
from knowseqpy.batch_effect import sva
from knowseqpy.classifiers import knn, gradient_boosting, rf, svm
from knowseqpy.feature_selection import linear_discriminant_analysis
from knowseqpy.utils import plot_boxplot, plot_confusion_matrix, plot_samples_heatmap

SCRIPT_PATH = Path(__file__).resolve().parent.parent
INFO_PATH = Path(os.getenv("SAMPLES_INFO_BREAST_PATH", f"{SCRIPT_PATH}/tests/test_fixtures/samples_info_breast.csv"))
COUNTS_PATH = Path(os.getenv("COUNTS_BREAST_PATH", f"{SCRIPT_PATH}/tests/test_fixtures/count_files_breast"))


def main():
    # Set seed for reproducible results
    random.seed(1234)

    # Load and preprocess count files to create a counts df
    counts, labels = counts_to_dataframe(info_path=INFO_PATH, counts_path=COUNTS_PATH)

    # Number of samples per class to understand the dataset's distribution
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
    batch_cleaned_df = sva(expression_df=cleaned_df, labels=qa_labels)

    # Identify differentially expressed genes with specified criteria
    degs = degs_extraction(data=batch_cleaned_df, labels=qa_labels, lfc=3.5, p_value=0.001)[0].transpose()

    # Select features (genes) for downstream analysis based on specified criteria
    selected_features = linear_discriminant_analysis(data=degs, labels=qa_labels, vars_selected=degs.columns.tolist())

    # Classifier training and prediction
    knn_res = knn(degs, qa_labels, selected_features)
    rf_res = rf(degs, qa_labels, selected_features)
    svm_res = svm(degs, qa_labels, selected_features)
    ga_res = gradient_boosting(degs, qa_labels, selected_features)

    # Visualization
    plot_boxplot(data=degs, labels=qa_labels, fs_ranking=selected_features, top_n_features=3)
    plot_confusion_matrix(conf_matrix=knn_res["confusion_matrix"], unique_labels=knn_res["unique_labels"].tolist())
    plot_samples_heatmap(data=degs, labels=qa_labels, fs_ranking=selected_features, top_n_features=4)


if __name__ == "__main__":
    main()
