import pandas as pd
import numpy as np
import os
import logging
import statsmodels.api as sm
from sklearn.preprocessing import quantile_transform

from knowseq.normalization import cqn

# Initialize logging
logging.basicConfig(level=logging.INFO)


def calculate_gene_expression_values(countsMatrix: pd.DataFrame, annotation: pd.DataFrame, genesNames: bool = True,
                                     not_human=False, not_human_gene_length_csv="", ensembl_id=True):
    """
    Calculates the gene expression values by using a matrix of counts from RNA-seq.

    Parameters:
        countsMatrix (pd.DataFrame): The counts matrix with genes in rows and samples in columns.
        annotation (pd.DataFrame): A DataFrame containing gene annotations.
        genesNames (bool): Whether to use gene names instead of Ensembl IDs.
        not_human (bool): Whether the data is not from humans.
        notHumanGeneLengthCSV (str): Path to the gene length file if notHuman is True.
        Ensembl_ID (bool): Whether the counts matrix contains Ensembl IDs.

    Returns:
        pd.DataFrame: A DataFrame containing the gene expression values.
    """

    # Loading gene length data
    if not not_human:
        # Load default human gene length data
        gene_length = pd.read_csv("genes_length_homo_sapiens.csv")  # TODO: File path to be updated
    else:
        if os.path.exists(not_human_gene_length_csv):
            gene_length = pd.read_csv(not_human_gene_length_csv)
        else:
            raise FileNotFoundError("Not Human gene length CSV file not found, please revise the path to the file.")

    logging.info("Calculating gene expression values...")

    if ensembl_id:
        my_g_cannot = annotation.set_index('ensembl_gene_id')['percentage_gene_gc_content']
        my_g_cannot = my_g_cannot.loc[countsMatrix.index]
        my_genes = countsMatrix.index.intersection(annotation['ensembl_gene_id'])
        my_length = gene_length.set_index('Gene_stable_ID').loc[my_genes, 'Gene_length']

        my_cqn = cqn(countsMatrix.loc[my_genes], x=my_g_cannot, lengths=my_length, sizeFactors=countsMatrix.sum(axis=1),
                     tau=0.5, sqn=True)
        cqn_values = my_cqn['y'] + my_cqn['offset']
        expression_matrix = cqn_values - np.min(cqn_values) + 1

        if genesNames:
            row_names = annotation.set_index('ensembl_gene_id').loc[my_genes, 'external_gene_name']
            expression_matrix.index = row_names
            expression_matrix = expression_matrix.loc[~expression_matrix.index.duplicated()]
    else:
        pass

    return expression_matrix
