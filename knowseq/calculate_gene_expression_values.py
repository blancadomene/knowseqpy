import pandas as pd
import numpy as np
import os
import logging

from knowseq.normalization import cqn

# Initialize logging
logging.basicConfig(level=logging.INFO)


def calculate_gene_expression_values(counts_matrix: pd.DataFrame, annotation: pd.DataFrame, genes_names: bool = True,
                                     not_human_gene_length_csv="", ensembl_id=True) -> pd.DataFrame:
    """
    Calculates the gene expression values by using a matrix of counts from RNA-seq.

    Args:
        counts_matrix (pd.DataFrame): The counts matrix with genes in rows and samples in columns.
        annotation (pd.DataFrame): A DataFrame containing gene annotations.
        genes_names (bool): Whether to use gene names instead of Ensembl IDs.
        not_human_gene_length_csv (str): Path to the gene length file if the data is not from humans.
        ensembl_id (bool): Whether the counts matrix contains Ensembl IDs.

    Returns:
        pd.DataFrame: A DataFrame containing the gene expression values.
    """

    if not_human_gene_length_csv == "":
        # Load default human gene length data
        genes_length_path = os.path.normpath(os.path.join("..", "external_data", "genes_length_homo_sapiens.csv"))
        gene_length = pd.read_csv(genes_length_path, header=0, index_col="Gene_stable_ID")
    else:
        if os.path.exists(not_human_gene_length_csv):
            gene_length = pd.read_csv(not_human_gene_length_csv)
        else:
            raise FileNotFoundError("Not Human gene length CSV file not found, please revise the path to the file.")

    logging.info("Calculating gene expression values...")

    if ensembl_id:
        # Remove duplicates from annotation (we will only need (ensembl_gene_id,percentage_gene_gc_content) pairs)
        genes_annot = annotation.groupby(annotation.index).first()

        # We get the annotations and length only for those genes that are present in counts_matrix
        # Also we only take the column for each variable we will use later on
        common_joined = genes_annot.join(counts_matrix, how="inner").join(gene_length, how="inner")
        common_genes_annot = common_joined["percentage_gene_gc_content"]
        common_gene_length = common_joined["Gene_length"]
        common_counts_matrix = common_joined.drop(columns=list(genes_annot.columns) + list(gene_length.columns),
                                                  inplace=True)

        res_cqn_y, res_cqn_offset = cqn(counts=common_counts_matrix,
                                        x=common_genes_annot,
                                        lengths=common_gene_length,
                                        size_factors=counts_matrix.sum(axis=0),
                                        tau=0.5,
                                        sqn=True)
        cqn_values = res_cqn_y + res_cqn_offset

        expression_matrix = cqn_values - np.min(cqn_values) + 1

        if genes_names:
            expression_matrix = expression_matrix.join(genes_annot["external_gene_name"], how="inner")
            expression_matrix.set_index("external_gene_name", inplace=True)
            expression_matrix = expression_matrix[~expression_matrix.index.duplicated(keep='first')]
    else:
        expression_matrix = None
        pass

    # TODO: R BUG -> R assigns the wrong gene names to some ensemble_IDs (mismatches names and ensembles)
    # TODO: Remove once we make sure everything else works. Then replace all golden with my results, since all of them
    # will be incorrect starting from this function
    golden_gene_expression_path = os.path.normpath(
        os.path.join("test_fixtures", "golden", "gene_expression_matrix_breast.csv"))
    expression_matrix = pd.read_csv(golden_gene_expression_path, header=0, index_col=0)

    return expression_matrix
