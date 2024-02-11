import logging
import os
import subprocess

import pandas as pd
from patsy.highlevel import dmatrix
from sklearn.model_selection import KFold

from knowseq.utils import dataframe_to_feather, feather_to_dataframe

logger = logging.getLogger(__name__)


# TODO: CV?
# TODO: Add notes about Rscript path
def degs_extraction(data: pd.DataFrame, labels: pd.Series, max_genes: int = float("inf"), p_value: float = 0.05,
                    lfc: float = 1.0, cv: bool = False, k_folds: int = 5) -> list[pd.DataFrame]:
    """
    Performs the analysis to extract Differentially Expressed Genes (DEGs) among classes to compare.

    Args:
        data: DataFrame containing genes in rows and samples in columns.
        labels: Series containing labels for each sample in data.
        p_value: p-value threshold for determining DEGs. Defaults to 0.05.
        lfc: Log Fold Change threshold for determining DEGs. Defaults to 1.0.
        max_genes: Maximum number of genes as output. Defaults to all genes.
        cv: If True, runs Cross-Validation DEGs extraction. Defaults to False.
        k_folds: Number of folds for Cross-Validation. Defaults to 5.

    Returns:
        Dictionary containing DEGs analysis results.
    """

    labels = labels.astype("category")

    cv_datasets = []
    if cv:
        logger.info("Applying DEGs extraction with Cross-Validation")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for _, test_index in kf.split(data):
            fold_data = data.iloc[test_index]
            fold_labels = labels.iloc[test_index]
            cv_datasets.append((fold_data, fold_labels))
    else:
        cv_datasets.append((data, labels))

    # Perform DEGs analysis for each dataset
    cv_degs_results = []
    for data, labels in cv_datasets:
        if len(labels.cat.categories) == 2:
            logger.info("Two classes detected, applying biclass analysis")
            cv_degs_results.append(_biclass_analysis(data, labels, p_value, lfc, max_genes))
        elif len(labels.cat.categories) > 2:
            logger.info("More than two classes detected, applying multiclass analysis")
            cv_degs_results.append(_multiclass_analysis())
        else:
            raise ValueError("Number of classes in labels must be at least 2.")

    return cv_degs_results


def _biclass_analysis(data: pd.DataFrame, labels: pd.Series, p_value: float, lfc: float,
                      max_genes: int) -> pd.DataFrame:
    """
    Performs biclass DEGs analysis by calling an R script for lmFit and eBayes.

    Args:
        data: DataFrame containing genes in rows and samples in columns.
        labels: Series containing labels for each sample in data.
        p_value: p-value threshold for determining DEGs.
        lfc: Log Fold Change threshold for determining DEGs.
        max_genes: Maximum number of genes as output.

    Returns:
        Analysis results including DEGs table and matrix.
    """

    degs_table = run_limma_deg_analysis(data, labels, p_value, lfc, max_genes)

    # TODO: Note: Test this additional processing
    if max_genes != float("inf") and len(degs_table) > max_genes:
        degs_table = degs_table.head(max_genes)

    return degs_table


def run_limma_deg_analysis(data, labels, p_value, lfc, max_genes) -> pd.DataFrame:
    """Runs differential expression analysis using R's limma package.

    Args:
        data: A pandas DataFrame containing gene expression data.
        labels: Series containing labels for each sample in data.
        p_value: p-value threshold for determining DEGs.
        lfc: Log Fold Change threshold for determining DEGs.
        max_genes: Maximum number of genes as output.

    Returns:
        A pandas DataFrame containing the limma top_table results.
    """

    labels = labels.astype("category")

    transposed_data = data.copy().T
    transposed_data["sample_class"] = labels.values
    design_matrix = dmatrix(formula_like="1 + C(sample_class)", data=transposed_data, return_type="dataframe").astype(
        int)

    script_path = os.path.dirname(os.path.abspath(__file__))
    expression_data_path = os.path.join(script_path, "r_scripts", "expression_data.feather")
    design_matrix_path = os.path.join(script_path, "r_scripts", "design_matrix.feather")
    limma_results_path = os.path.join(script_path, "r_scripts", "limma_results.feather")

    dataframe_to_feather(data, [script_path, "r_scripts", "expression_data.feather"])
    dataframe_to_feather(design_matrix, [script_path, "r_scripts", "design_matrix.feather"])

    command = [
        "Rscript",
        os.path.join(script_path, "r_scripts", "LimmaDEGsExtractionWorkflow.R"),
        expression_data_path,
        design_matrix_path,
        limma_results_path,
        str(p_value),
        str(lfc),
        str(max_genes)
    ]

    subprocess.run(command, check=True)

    results = feather_to_dataframe([script_path, "r_scripts", "limma_results.feather"])
    results.set_index("row_name", inplace=True)
    results.index.name = None

    os.remove(expression_data_path)
    os.remove(design_matrix_path)
    os.remove(limma_results_path)

    return results


def _multiclass_analysis():
    raise NotImplementedError("Multiclass analysis function has not been implemented yet")
