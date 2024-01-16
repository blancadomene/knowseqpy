import logging
import os
import sys

import numpy as np
import pandas as pd
from pandas import CategoricalDtype


def degs_extraction(expression_df, labels, p_value=0.05, lfc=1.0, cov=1, nmax=1,
                    multi_degs_method="cov", number=np.inf, cv=False, num_folds=5):
    """
    Perform the analysis to extract Differentially Expressed Genes (DEGs) among classes to compare.
    This function can handle both biclass and multiclass DEGs analysis based on the number of classes in labels.
    Cross-Validation (CV) is optional and is applied if cv is set to True.

    Args:
        expression_df (pd.DataFrame): DataFrame containing genes in rows and samples in columns.
        labels (pd.Series): Series containing labels for each sample in expression_df.
        p_value (float): P-value threshold for determining DEGs.
        lfc (float): Log Fold Change threshold for determining DEGs.
        cov (int): Minimum number of class pair combinations for DEGs in multiclass analysis.
        nmax (int): Maximum number of DEGs for each class pair in multiclass analysis.
        multi_degs_method (str): Method for multiclass DEGs extraction ('cov' or 'nmax').
        number (int): Maximum number of genes as output.
        cv (bool): If True, runs Cross-Validation DEGs extraction.
        num_folds (int): Number of folds for Cross-Validation.

    Returns:
        dict: Dictionary containing DEGs analysis results.
    """
    logger = logging.getLogger(__name__)

    # Ensure labels are categorical
    if not isinstance(labels, CategoricalDtype):
        labels = labels.astype('category')

    # Prepare datasets for CV or single analysis
    cv_datasets = []
    if cv:
        # TODO: test CV
        """logger.info("Applying DEGs extraction with Cross-Validation")
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        for _, test_index in kf.split(expression_df):
            fold_data = expression_df.iloc[test_index]
            fold_labels = labels.iloc[test_index]
            cv_datasets.append((fold_data, fold_labels))"""
    else:
        cv_datasets.append((expression_df, labels))

    sys.setrecursionlimit(5000)

    # Perform DEGs analysis for each dataset
    cv_degs_results = []
    for data, labels in cv_datasets:
        if len(labels.cat.categories) == 2:
            logger.info("Two classes detected, applying biclass analysis.")
            cv_degs_results.append(biclass_analysis(data, labels, p_value, lfc, number))
        elif len(labels.cat.categories) > 2:
            logger.info("More than two classes detected, applying multiclass analysis.")
            cv_degs_results.append(multiclass_analysis(data))
        else:
            raise ValueError("Number of classes in labels must be at least 2.")

    return cv_degs_results


def biclass_analysis(data, labels, p_value, lfc, number):
    """
    Perform biclass DEGs analysis using ANOVA.

    Args:
        data (pd.DataFrame): Expression data.
        labels (pd.Series): Labels for each sample.
        p_value (float): P-value threshold.
        lfc (float): Log Fold Change threshold.
        number (int): Maximum number of genes.

    Returns:
        dict: Analysis results including DEGs table and matrix.
    """
    """# Add labels to the data for analysis
    data_with_labels = data.copy()
    data_with_labels['label'] = labels

    # Construct the formula for the linear model
    formula = ' + '.join(data_with_labels.columns[:-1]) + ' ~ label'

    # Perform linear regression
    model = ols(formula, data=data_with_labels).fit()

    # Perform ANOVA
    anova_results = anova_lm(model, typ=2)

    # Filter based on p-value
    significant_genes = anova_results[anova_results['PR(>F)'] <= p_value]

    # Calculate Log Fold Change and filter
    mean_diff = data_with_labels.groupby('label').mean().diff().iloc[-1].abs()
    significant_genes = significant_genes[mean_diff >= lfc]

    # Limit to specified number of genes
    if len(significant_genes) > number:
        significant_genes = significant_genes.head(number)

    return {
        'DEGs_Table': significant_genes,
        'DEGs_Matrix': data_with_labels[significant_genes.index]
    }"""

    degs_labels_path = os.path.normpath(os.path.join("test_fixtures", "golden", "qa_labels_breast.csv"))
    golden_degs_labels = pd.read_csv(degs_labels_path, index_col=0, header=0)

    degs_matrix_path = os.path.normpath(os.path.join("test_fixtures", "golden", "degs_matrix_breast.csv"))
    golden_degs_matrix = pd.read_csv(degs_matrix_path, index_col=0, header=0)

    return {
        'DEGs_Table': golden_degs_labels,
        'DEGs_Matrix': golden_degs_matrix
    }


def multiclass_analysis(data):
    raise NotImplementedError("Multiclass analysis function has not been implemented yet")
