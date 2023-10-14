import logging

import os
import pandas as pd

from cqn import cqn


def calculate_gene_expression_values(counts_matrix: pd.DataFrame,
                                     annotation: pd.DataFrame,
                                     genes_names: bool = True,
                                     not_human: bool = False,
                                     not_human_gene_length_csv: str = "",
                                     ensembl_id: bool = True):
    """
    Calculates the gene expression values by using a matrix of counts from RNA-seq.

    :param counts_matrix: The original counts matrix with gene Ensembl ID in the rows and samples in the columns.
    :param annotation: A DataFrame containing Ensembl IDs, gene names, and gene gc content.
    :param genes_names: Indicates if the row names of the expression matrix are gene names (True) or Ensembl IDs (False). Defaults to True.
    :param not_human: Indicates whether the gene length file is for human (False) or another species (True). Defaults to False.
    :param not_human_gene_length_csv: Path to the CSV file containing gene lengths for the species if not_human is True.
    :param ensembl_id: Indicates whether the counts matrix contains Ensembl IDs (True) or gene names (False). Defaults to True.


    :return: DataFrame containing gene annotations, including Ensembl gene ID, gene symbol and the percentage gene GC content
    """

    if not_human:
        assert os.path.exists(
            not_human_gene_length_csv), "Not Human gene length CSV file not found, please revise the path to the file."
        gene_length = pd.read_csv(not_human_gene_length_csv)
    else:
        # The path for "Genes_length_Homo_Sapiens.csv" needs to be updated
        gene_length = pd.read_csv("path_to_Genes_length_Homo_Sapiens.csv")
        gene_length = gene_length.drop(columns=['column_3', 'column_4'])

    logging.info("Calculating gene expression values...\n")

    if ensembl_id:
        my_gc_not = annotation.set_index('ensembl_gene_id')['percentage_gene_gc_content']
        my_gc_not = my_gc_not.loc[counts_matrix.index]
        my_genes = counts_matrix.index.intersection(annotation['ensembl_gene_id'])

        my_length = gene_length.set_index('Gene_stable_ID').loc[my_genes, 'length']
        my_length = my_length.dropna()
        my_genes = my_length.index

        row_names = annotation.set_index('ensembl_gene_id').loc[my_genes, 'external_gene_name']

        my_gc_not = my_gc_not.loc[my_genes]

        # Call to cqn
        cqn_result = cqn(counts_matrix.loc[my_genes].values, my_gc_not.values, my_length.values)
        cqn_values = cqn_result['y'] + cqn_result['offset']
        expression_matrix = pd.DataFrame(cqn_values, index=counts_matrix.loc[my_genes].index,
                                         columns=counts_matrix.columns)
        expression_matrix = expression_matrix - expression_matrix.min().min() + 1

        if genes_names:
            expression_matrix.index = row_names

    else:
        annotation = annotation.loc[annotation['external_gene_name'].drop_duplicates().index]
        counts_matrix = counts_matrix.loc[annotation['external_gene_name']]

        my_gc_not = annotation.set_index('external_gene_name')['percentage_gene_gc_content']
        my_gc_not = my_gc_not.loc[counts_matrix.index]
        my_genes = counts_matrix.index.intersection(annotation['external_gene_name'])

        my_length = gene_length.set_index('Gene_name').loc[my_genes, 'length']
        my_length = my_length.dropna()
        my_genes = my_length.index

        row_names = annotation.set_index('external_gene_name').loc[my_genes, 'external_gene_name']

        my_gc_not = my_gc_not.loc[my_genes]

        # Call to cqn
        cqn_result = cqn(counts_matrix.loc[my_genes].values, my_gc_not.values, my_length.values)
        cqn_values = cqn_result['y'] + cqn_result['offset']
        expression_matrix = pd.DataFrame(cqn_values, index=counts_matrix.loc[my_genes].index,
                                         columns=counts_matrix.columns)
        expression_matrix = expression_matrix - expression_matrix.min().min() + 1

    return expression_matrix
