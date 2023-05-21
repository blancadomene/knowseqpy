import pandas as pd
import os

from cqn import cqn

def calculate_gene_expression_values(counts_matrix: pd.DataFrame,
                                     annotation: pd.DataFrame,
                                     genes_names: bool = True,
                                     not_human: bool = False,
                                     not_human_gene_length_csv: str = "",
                                     ensembl_id: bool = True):
    assert isinstance(genes_names, bool), "genesNames parameter can only takes the values TRUE or FALSE."
    assert isinstance(counts_matrix, pd.DataFrame), "The class of countsMatrix parameter must be DataFrame."
    assert isinstance(annotation, pd.DataFrame), "The class of annotation parameter must be DataFrame."
    assert isinstance(not_human, bool), "notHuman parameter can only takes the values TRUE or FALSE."

    if not_human:
        assert os.path.exists(
            not_human_gene_length_csv), "Not Human gene length CSV file not found, please revise the path to the file."
        gene_length = pd.read_csv(not_human_gene_length_csv)
    else:
        # The path for "Genes_length_Homo_Sapiens.csv" needs to be updated
        gene_length = pd.read_csv("path_to_Genes_length_Homo_Sapiens.csv")
        gene_length = gene_length.drop(columns=['column_3', 'column_4'])

    print("Calculating gene expression values...\n")

    if ensembl_id:
        my_gc_not = annotation.set_index('ensembl_gene_id')['percentage_gene_gc_content']
        my_gc_not = my_gc_not.loc[counts_matrix.index]
        my_genes = counts_matrix.index.intersection(annotation['ensembl_gene_id'])

        mylength = gene_length.set_index('Gene_stable_ID').loc[my_genes, 'length']
        mylength = mylength.dropna()
        my_genes = mylength.index

        rownames = annotation.set_index('ensembl_gene_id').loc[my_genes, 'external_gene_name']

        my_gc_not = my_gc_not.loc[my_genes]

        # Call to cqn
        cqn_result = cqn(counts_matrix.loc[my_genes].values, my_gc_not.values, mylength.values)
        cqn_values = cqn_result['y'] + cqn_result['offset']
        expression_matrix = pd.DataFrame(cqn_values, index=counts_matrix.loc[my_genes].index,
                                         columns=counts_matrix.columns)
        expression_matrix = expression_matrix - expression_matrix.min().min() + 1

        if genes_names:
            expression_matrix.index = rownames

    else:
        annotation = annotation.loc[annotation['external_gene_name'].drop_duplicates().index]
        counts_matrix = counts_matrix.loc[annotation['external_gene_name']]

        my_gc_not = annotation.set_index('external_gene_name')['percentage_gene_gc_content']
        my_gc_not = my_gc_not.loc[counts_matrix.index]
        my_genes = counts_matrix.index.intersection(annotation['external_gene_name'])

        mylength = gene_length.set_index('Gene_name').loc[my_genes, 'length']
        mylength = mylength.dropna()
        my_genes = mylength.index

        rownames = annotation.set_index('external_gene_name').loc[my_genes, 'external_gene_name']

        my_gc_not = my_gc_not.loc[my_genes]

        # Call to cqn
        cqn_result = cqn(counts_matrix.loc[my_genes].values, my_gc_not.values, mylength.values)
        cqn_values = cqn_result['y'] + cqn_result['offset']
        expression_matrix = pd.DataFrame(cqn_values, index=counts_matrix.loc[my_genes].index,
                                         columns=counts_matrix.columns)
        expression_matrix = expression_matrix - expression_matrix.min().min() + 1

    return expression_matrix
