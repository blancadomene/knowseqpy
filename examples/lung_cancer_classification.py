"""
Example pipeline using breast samples.
"""
import os
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from knowseqpy import counts_to_dataframe, get_genes_annotation, calculate_gene_expression_values, rna_seq_qa, \
    degs_extraction
from knowseqpy.batch_effect import sva
from knowseqpy.classifiers import decision_tree, logistic_regression
from knowseqpy.evaluate_model import evaluate_model
from knowseqpy.feature_selection import random_forest
from knowseqpy.utils import plot_boxplot, plot_confusion_matrix, plot_samples_heatmap
from knowseqpy.utils.plotting import plot_decision_boundary, plot_decision_tree

SCRIPT_PATH = Path(__file__).resolve().parent.parent

INFO_PATH_DEFAULT = f"{SCRIPT_PATH}/tests/test_fixtures/samples_info_lung.csv"
INFO_PATH_ENV = os.getenv("SAMPLES_INFO_BREAST_PATH")
INFO_PATH = Path(INFO_PATH_ENV if INFO_PATH_ENV else INFO_PATH_DEFAULT)

COUNTS_PATH_DEFAULT = f"{SCRIPT_PATH}/tests/test_fixtures/count_files_lung"
COUNTS_PATH_ENV = os.getenv("COUNTS_BREAST_PATH")
COUNTS_PATH = Path(COUNTS_PATH_ENV if COUNTS_PATH_ENV else COUNTS_PATH_DEFAULT)


def main():
    # Set seed for reproducible results
    seed = 1234
    random.seed(seed)

    # Preprocess the info file, renaming the count name column and the sample types
    df = pd.read_csv(INFO_PATH)
    df["Sample.Type"] = df["Sample.Type"].replace({
        "Adenocarcinoma": "Primary Tumor",
        "SquamousCellCarcinoma": "Primary Tumor",
        "Healthy": "Solid Tissue Normal"
    })
    df.rename(columns={"File.Name": "Internal.ID"}, inplace=True)
    preprocessed_info_path = f"{INFO_PATH}_preprocessed"
    df.to_csv(preprocessed_info_path, index=True)

    # Load and preprocess count files to create a counts df
    counts, labels = counts_to_dataframe(info_path=preprocessed_info_path, counts_path=COUNTS_PATH, ext="")

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
    selected_features = random_forest(data=degs, labels=qa_labels, vars_selected=degs.columns.tolist())
    selected_features = selected_features[:10]

    # Split data into training and testing sets (80% training, 20% testing)
    degs_train, degs_test, labels_train, labels_test = train_test_split(degs, qa_labels, test_size=0.2,
                                                                        random_state=seed, stratify=qa_labels)

    # Classifier training and prediction using a decision tree
    dt_model = decision_tree(data=degs_train, labels=labels_train, vars_selected=selected_features)
    dt_pred = evaluate_model(model=dt_model, data_test=degs_test, labels_test=labels_test,
                             vars_selected=selected_features)

    # Visualization
    plot_decision_tree(dt_model, qa_labels, selected_features)
    plot_boxplot(data=degs, labels=qa_labels, fs_ranking=selected_features, top_n_features=10)
    plot_confusion_matrix(conf_matrix=dt_pred["confusion_matrix"], unique_labels=dt_pred["unique_labels"].tolist())
    plot_samples_heatmap(data=degs, labels=qa_labels, fs_ranking=selected_features, top_n_features=10)

    # Try another classifier: logistic regression
    selected_features = selected_features[:2]
    dt_model = logistic_regression(data=degs_train, labels=labels_train, vars_selected=selected_features)
    dt_pred = evaluate_model(model=dt_model, data_test=degs_test, labels_test=labels_test,
                             vars_selected=selected_features)
    plot_decision_boundary(model=dt_model, data=degs_train, labels=labels_train, vars_selected=selected_features)
    plot_confusion_matrix(conf_matrix=dt_pred["confusion_matrix"], unique_labels=dt_pred["unique_labels"].tolist())


if __name__ == "__main__":
    main()
