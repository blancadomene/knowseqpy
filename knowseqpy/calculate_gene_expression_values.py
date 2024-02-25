"""
This module offers functionality to calculate gene expression values from RNA-seq data.
It supports both human and non-human gene lengths. It processes a counts matrix and annotation DataFrame
to produce a DataFrame of gene expression values.
"""
import logging
import os
import subprocess

import pandas as pd

from knowseqpy.utils import dataframe_to_feather, feather_to_dataframe

logger = logging.getLogger(__name__)


# TODO: R BUG -> R code for this func assigns the wrong gene names to some ensemble_IDs (mismatches names and ensembles)
#  Remove once we make sure everything else works. Then replace all golden with my results, since all of them
#  will be incorrect starting from this function
def calculate_gene_expression_values(counts_matrix: pd.DataFrame, annotation: pd.DataFrame, genes_names: bool = True,
                                     not_human_gene_length_csv="", ensembl_id=True) -> pd.DataFrame:
    """
    Calculates the gene expression values by using a matrix of counts from RNA-seq.

    Args:
        counts_matrix: The counts matrix with genes in rows and samples in columns.
        annotation: A DataFrame containing gene annotations.
        genes_names: Whether to use gene names instead of Ensembl IDs.
        not_human_gene_length_csv: Path to the gene length file if the data is not from humans.
        ensembl_id: Whether the counts matrix contains Ensembl IDs.

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
        # Remove duplicates from annotation, as we will only need (ensembl_gene_id, percentage_gene_gc_content) pairs
        genes_annot = annotation.groupby(annotation.index).first()

        # We get the annotations and length only for those genes that are present in counts_matrix
        # Also we only take the column for each variable we will use later on
        common_joined = genes_annot.join(counts_matrix, how="inner").join(gene_length, how="inner")
        common_genes_annot = common_joined["percentage_gene_gc_content"]
        common_gene_length = common_joined["Gene_length"]
        common_joined.drop(columns=list(genes_annot.columns) + list(gene_length.columns),
                           inplace=True)

        """
        res_cqn_y, res_cqn_offset = cqn(counts=common_joined,
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
        
        return expression_matrix
        """

        logger.info("Calculating gene expression values using R")
        script_path = os.path.dirname(os.path.abspath(__file__))
        counts_path = os.path.join(script_path, "r_scripts", "counts.feather")
        x_path = os.path.join(script_path, "r_scripts", "x.feather")
        length_path = os.path.join(script_path, "r_scripts", "length.feather")
        results_path = os.path.join(script_path, "r_scripts", "gene_results.feather")

        dataframe_to_feather(common_joined, [script_path, "r_scripts", "counts.feather"])
        dataframe_to_feather(common_genes_annot.to_frame(), [script_path, "r_scripts", "x.feather"])
        dataframe_to_feather(common_gene_length.to_frame(), [script_path, "r_scripts", "length.feather"])

        command = [
            "Rscript",
            os.path.join(script_path, "r_scripts", "cqnWorkflow.R"),
            counts_path,
            x_path,
            length_path,
            results_path
        ]
        try:
            subprocess.run(command, check=True)
            pass
        except subprocess.CalledProcessError as e:
            logger.error("R script execution failed: %s", {e})
            raise RuntimeError("Failed to execute cqn R script.") from e

        # Process R output
        expression_matrix = feather_to_dataframe([script_path, "r_scripts", "gene_results.feather"])
        expression_matrix.set_index("row_name", inplace=True)
        expression_matrix.index.name = None

        # Use gene_names. If gene ID not found in annotation, it's row is removed from the final df
        if genes_names:
            expression_matrix = expression_matrix.join(genes_annot["external_gene_name"], how="inner")
            expression_matrix.set_index("external_gene_name", inplace=True)
            expression_matrix = expression_matrix[~expression_matrix.index.duplicated(keep='first')]

        os.remove(counts_path)
        os.remove(x_path)
        os.remove(length_path)
        os.remove(results_path)

        return expression_matrix

    else:
        raise NotImplementedError("Non ensembl_id gene extraction has not been implemented yet")
