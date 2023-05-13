import pandas as pd
import numpy as np
import os


def calculate_gene_expression_values(counts_matrix: pd.DataFrame, annotation: pd.DataFrame, human: bool = True, diff_gene_length_path: str = ""):
    """
    gene expression values by using a matrix of counts from RNA-seq. The conversion from Ensembl IDs to genes names is performed by default, but can be changed with the parameter genesNames.

    :param counts_matrix:
    :param annotation:

    :return: ????.
    """

    if not human:
        if not os.path.exists(diff_gene_length_path):
            raise Exception("Diff Human gene length CSV file not found, please revise the path to the file.\n")
        gene_length = pd.read_csv(diff_gene_length_path, header=True, sep=',')

    path = os.path.join("external_data", "Genes_length_Homo_Sapiens.csv")
    gene_length = pd.read_csv(path, header=True, sep=',')
    gene_length = gene_length.iloc[:, [0, 1]]

    print("Calculating gene expression values...\n")




    # Normalize counts
    counts_normalized = counts_matrix.divide(counts_matrix.sum(axis=0), axis=1)

    # Calculate mean expression values
    mean_expr = counts_normalized.groupby(annotation['ensembl_gene_id']).mean()

    # Remove genes with zero mean expression
    mean_expr = mean_expr.loc[(mean_expr != 0).any(axis=1)]

    # Log2 transform expression values
    log2_expr = np.log2(mean_expr + 1)

    return log2_expr
