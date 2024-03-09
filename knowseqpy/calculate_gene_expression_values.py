"""
This module offers functionality to calculate gene expression values from RNA-seq data.
It supports both human and non-human gene lengths. It processes a counts and annotation DataFrames
to produce a DataFrame of gene expression values.
"""

import os
import subprocess
from pathlib import Path

import pandas as pd

from src.log import get_logger
from src.utils import dataframe_to_feather, feather_to_dataframe

logger = get_logger().getChild(__name__)


# TODO: R BUG -> R code for this func assigns the wrong gene names to some ensemble_IDs (mismatches names and ensembles)
#  Remove once we make sure everything else works. Then replace all golden_breast with my results, since all of them
#  will be incorrect starting from this function
def calculate_gene_expression_values(counts: pd.DataFrame, gene_annotation: pd.DataFrame, genes_names: bool = True,
                                     not_human_gene_length_csv="", ensembl_id=True) -> pd.DataFrame:
    """
    Calculates the gene expression values by using a matrix of counts from RNA-seq.

    Args:
        counts: The counts pd.DataFrame with genes in rows and samples in columns.
        gene_annotation: A DataFrame containing gene annotations.
        genes_names: Use gene names instead of Ensembl IDs.
        not_human_gene_length_csv: Path to the gene length file if the data is not from humans.
        ensembl_id: Whether the counts pd.DataFrame contains Ensembl IDs.

    Returns:
        pd.DataFrame: A pd.DataFrame containing the gene expression values.
    """
    if not ensembl_id:
        raise NotImplementedError("Non ensembl_id gene extraction has not been implemented yet")

    if not_human_gene_length_csv == "":
        # Load default human gene length data
        genes_length_path = os.path.join(str(Path(__file__).resolve().parents[1]),
                                         "external_data", "genes_length_homo_sapiens.csv")
        gene_length = pd.read_csv(genes_length_path, header=0, index_col="Gene_stable_ID")
    else:
        if not os.path.exists(not_human_gene_length_csv):
            raise FileNotFoundError("Not Human gene length CSV file not found, please revise the path to the file.")

        gene_length = pd.read_csv(not_human_gene_length_csv)

    logger.info("Calculating gene expression values...")

    # Remove duplicates from gene_annotation, as we'll only need (ensembl_gene_id, percentage_gene_gc_content) pairs
    genes_annot = gene_annotation.groupby(gene_annotation.index).first()

    # We get the gene_annotations and length only for those genes that are present in counts_df
    common_joined = genes_annot.join(counts, how="inner").join(gene_length, how="inner")
    common_genes_annot = common_joined["percentage_gene_gc_content"]
    common_gene_length = common_joined["Gene_length"]
    common_joined.drop(columns=list(genes_annot.columns) + list(gene_length.columns),
                       inplace=True)

    logger.info("Calculating gene expression values using R")
    script_path = os.path.dirname(os.path.abspath(__file__))
    counts_path = os.path.join(script_path, "r_scripts", "counts.feather")
    x_path = os.path.join(script_path, "r_scripts", "x.feather")
    length_path = os.path.join(script_path, "r_scripts", "length.feather")
    results_path = os.path.join(script_path, "r_scripts", "gene_results.feather")

    dataframe_to_feather(common_joined, [script_path, "r_scripts", "counts.feather"])
    dataframe_to_feather(common_genes_annot.to_frame(), [script_path, "r_scripts", "x.feather"])
    dataframe_to_feather(common_gene_length.to_frame(), [script_path, "r_scripts", "length.feather"])

    try:
        subprocess.run([
            "Rscript",
            os.path.join(script_path, "r_scripts", "cqnWorkflow.R"),
            counts_path,
            x_path,
            length_path,
            results_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute cqn R script. {e}") from e

    # Process R output
    gene_expression = feather_to_dataframe([script_path, "r_scripts", "gene_results.feather"])
    gene_expression.set_index("row_name", inplace=True)
    gene_expression.index.name = None

    # Use gene_names. If gene ID not found in gene_annotation, its row is removed from the final df
    if genes_names:
        gene_expression = gene_expression.join(genes_annot["external_gene_name"], how="inner")
        gene_expression.set_index("external_gene_name", inplace=True)
        gene_expression = gene_expression[~gene_expression.index.duplicated(keep='first')]

    os.remove(counts_path)
    os.remove(x_path)
    os.remove(length_path)
    os.remove(results_path)

    return gene_expression
