"""
This module offers functionality to calculate gene expression values from RNA-seq data.
It supports both human and non-human gene lengths. It processes a counts and annotation DataFrames
to produce a DataFrame of gene expression values.
"""
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from .utils import dataframe_to_feather, feather_to_dataframe, get_logger, get_project_path

logger = get_logger().getChild(__name__)


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
        genes_length_path = Path(__file__).resolve().parent / "external_data" / "genes_length_homo_sapiens.csv"
        gene_length = pd.read_csv(genes_length_path, header=0, index_col="Gene_stable_ID")
    else:
        if not Path(not_human_gene_length_csv).exists():
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
    with tempfile.TemporaryDirectory() as temp_dir:
        counts_path = Path(temp_dir, "counts.feather")
        x_path = Path(temp_dir, "x.feather")
        length_path = Path(temp_dir, "length.feather")
        results_path = Path(temp_dir, "gene_results.feather")

        dataframe_to_feather(common_joined, counts_path)
        dataframe_to_feather(common_genes_annot.to_frame(), x_path)
        dataframe_to_feather(common_gene_length.to_frame(), length_path)

        try:
            subprocess.run([
                "Rscript",
                get_project_path() / "knowseqpy" / "r_scripts" / "cqnWorkflow.R",
                str(counts_path),
                str(x_path),
                str(length_path),
                str(results_path)
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute cqn R script. {e}") from e

        gene_expression = feather_to_dataframe(results_path)
        gene_expression.set_index("row_name", inplace=True)
        gene_expression.index.name = None

        # Use gene_names. If gene ID not found in gene_annotation, its row is removed from the final df
        if genes_names:
            gene_expression = gene_expression.join(genes_annot["external_gene_name"], how="inner")
            gene_expression.set_index("external_gene_name", inplace=True)
            gene_expression = gene_expression[~gene_expression.index.duplicated(keep="first")]

        return gene_expression
