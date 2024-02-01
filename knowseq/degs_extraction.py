import logging
import sys

import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

logger = logging.getLogger(__name__)


def degs_extraction(data: pd.DataFrame, labels: pd.Series, max_genes: int = float("inf"), p_value: float = 0.05,
                    lfc: float = 1.0, cv: bool = False, k_folds: int = 5) -> list[dict]:
    """
    Performs the analysis to extract Differentially Expressed Genes (DEGs) among classes to compare.

    Args:
        data: DataFrame containing genes in rows and samples in columns.
        labels: Series containing labels for each sample in data.
        p_value: P-value threshold for determining DEGs. Defaults to 0.05.
        lfc: Log Fold Change threshold for determining DEGs. Defautls to 1.0.
        max_genes: Maximum number of genes as output. # TODO: Defaults to ?
        cv: If True, runs Cross-Validation DEGs extraction. Defaults to False.
        k_folds: Number of folds for Cross-Validation. Defaults to 5.

    Returns:
        Dictionary containing DEGs analysis results.
    """

    # TODO label_codes = pd.factorize(labels)
    labels = labels.astype("category")

    cv_datasets = []
    if cv:
        # TODO: test CV
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


def _biclass_analysis(data: pd.DataFrame, labels: pd.Series, p_value: float, lfc: float, max_genes: int) -> dict:
    """
    Performs biclass DEGs analysis using ANOVA.

    Args:
        data: Expression data.
        labels: Labels for each sample.
        p_value: p-value threshold.
        lfc: Log Fold Change threshold.
        max_genes: Maximum number of genes.

    Returns:
        Analysis results including DEGs table and matrix.
    """
    data_with_labels = data.copy().T
    data_with_labels["sample_class"] = labels.values
    # al reves?
    # predictors = ["Q('{}')".format(col) for col in data.columns]  # Wrap each column name with Q()
    # formula = " + ".join(predictors) + " ~ sample_class"
    formula = " + ".join(data_with_labels.columns[:-1]) + " ~ sample_class"

    sys.setrecursionlimit(1000000)

    model = ols(formula, data=data_with_labels).fit()
    anova_results = anova_lm(model, typ=2)

    significant_genes = anova_results[anova_results['PR(>F)'] <= p_value]

    # Calculate Log Fold Change
    mean_diff = data_with_labels.groupby("class").mean().diff().iloc[-1].abs()
    significant_genes_lfc = mean_diff[mean_diff >= lfc]

    significant_genes = significant_genes.loc[significant_genes_lfc.index]

    # Limit to specified number of genes
    if len(significant_genes) > max_genes:
        significant_genes = significant_genes.head(max_genes)

    return {
        'DEGs_Table': significant_genes,
        'DEGs_Matrix': data_with_labels.loc[significant_genes.index]
    }


""" golden_degs_labels = csv_to_dataframe(
    path_components=["test_fixtures", "golden", "qa_labels_breast.csv"], index_col=0, header=0)
golden_degs_matrix = csv_to_dataframe(
    path_components=["test_fixtures", "golden", "degs_matrix_breast.csv"], index_col=0, header=0)

return {
    'DEGs_Table': golden_degs_labels,
    'DEGs_Matrix': golden_degs_matrix
}"""


def _multiclass_analysis():
    raise NotImplementedError("Multiclass analysis function has not been implemented yet")
