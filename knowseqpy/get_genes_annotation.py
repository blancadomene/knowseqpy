"""
This module provides functionality to retrieve gene annotations from the Ensembl biomart database. It supports querying
for specific genes or the entire genome, with customizable attributes and filters. Annotations can be fetched for both
human and non-human species, accommodating different reference genomes (GRCh37, GRCh38). Results are returned as pandas
DataFrames, facilitating further data analysis and manipulation.
"""

import io
from pathlib import Path

import pandas as pd
import requests

from .utils import csv_to_dataframe, get_logger

EXTERNAL_DATA_PATH = Path(__file__).resolve().parent / "external_data"
ENSEMBL_URL = "http://www.ensembl.org/biomart/martservice"
GRCH37_ENSEMBL_URL = "https://grch37.ensembl.org/biomart/martservice"

logger = get_logger().getChild(__name__)


def get_genes_annotation(values: list[str], attributes: list[str] = None, attribute_filter: str = "ensembl_gene_id",
                         not_hsapiens_dataset: str = None, reference_genome: int = 38) -> pd.DataFrame:
    """
    Retrieves gene annotations from the Ensembl biomart database.

    Args:
        values: A list of genes that can contain Ensembl IDs, gene names, or "allGenome"
                to indicate all genes.
        attributes: A list of desired attributes or information to retrieve from Ensembl biomart.
                    Default is ['ensembl_gene_id', 'external_gene_name', 'percentage_gene_gc_content', 'entrezgene_id'].
        attribute_filter: An attribute used as a filter to return the rest of the attributes.
                          Default is 'ensembl_gene_id'.
        not_hsapiens_dataset: The dataset identification for non-human annotations. Default is None.
        reference_genome: The reference genome to use. It must be 37 or 38. Default is 38.

    Returns:
        A DataFrame containing the requested gene annotations.

    Raises:
        ValueError: If invalid input is provided for the parameters.
        ValueError: If an error occurs during the query, or the query result is empty or contains an error message.
    """
    # Temporary condition to use the package's GRCh38 annotation. Can remove this once we clarify why
    # external_data\GRCh38Annotation.csv yields different results for approx 360 IDs compared to the BioMart download.
    if reference_genome == 38 and not_hsapiens_dataset is None:
        annotation_df = csv_to_dataframe(path_components=[str(Path(__file__).resolve().parent),
                                                          "external_data", "GRCh38Annotation.csv"], header=0)
        if list(values) == ["allGenome"]:
            return annotation_df

        # Filtered len(annotation_df) can be greater than len(values) since we usually have duplicated ensembl_gene_id
        filtered_annotation_df = annotation_df[annotation_df[attribute_filter].isin(values)]
        filtered_annotation_df.set_index("ensembl_gene_id", inplace=True)
        return filtered_annotation_df

    if not attributes:
        attributes = ["ensembl_gene_id", "external_gene_name", "percentage_gene_gc_content", "entrezgene_id"]

    if attribute_filter not in attributes:
        attributes += [attribute_filter]

    base_url, dataset_name, filename = _resolve_dataset_details(not_hsapiens_dataset, reference_genome)

    annotation_list = []
    current_batch_values = values
    max_values_per_query = min(len(values), 900)

    while current_batch_values:
        batch_values = current_batch_values[:max_values_per_query]
        query = _build_query(dataset_name, attributes, attribute_filter, batch_values, max_values_per_query)
        annotation_list.append(_fetch_annotation(query, base_url, attributes))
        current_batch_values = current_batch_values[max_values_per_query:]

    annotation_df = pd.concat(annotation_list, ignore_index=True) if annotation_list else pd.DataFrame()

    if not annotation_df.empty:
        annotation_df.to_csv(f"{EXTERNAL_DATA_PATH}/{filename}", index=False)

    return annotation_df


def _resolve_dataset_details(not_hsapiens_dataset: str, reference_genome: int) -> (str, str, str):
    """
    Determines the dataset name and base URL based on the provided parameters.

    Args:
        not_hsapiens_dataset: The dataset identification for non-human annotations.
        reference_genome: The reference genome to use.

    Returns:
        A tuple containing the base URL, the dataset name, and the filename.
    """
    if not_hsapiens_dataset:
        if not_hsapiens_dataset == "" or not isinstance(not_hsapiens_dataset, str):
            raise ValueError("The 'not_hsapiens_dataset' parameter must be a non-empty string.")

        return ENSEMBL_URL, not_hsapiens_dataset, f"{not_hsapiens_dataset}.csv"

    if reference_genome == 38:
        logger.info("Using reference genome 38")
        dataset_name = "hsapiens_gene_ensembl"
        return ENSEMBL_URL, dataset_name, f"{dataset_name}.csv"

    logger.info("Using reference genome 37")
    dataset_name = "hsapiens_gene_ensembl"
    return GRCH37_ENSEMBL_URL, dataset_name, f"{dataset_name}.csv"


def _build_query(dataset_name: str, attributes: list[str], attribute_filter: str,
                 values: list[str], max_values: int) -> str:
    """
    Builds the query XML for the Ensembl biomart request.

    Args:
        dataset_name: The name of the dataset to query.
        attributes: A list of attributes to retrieve.
        attribute_filter: The filter to apply to the query.
        values: The values to filter by.
        max_values: The maximum number of values to include in a single query.

    Returns:
        A string representing the XML query.
    """
    query = f"<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE Query>" \
            f"<Query virtualSchemaName='default' formatter='CSV' header='0' uniqueRows='0' count='' " \
            f"datasetConfigVersion='0.6'> <Dataset name='{dataset_name}' interface='default'>"

    if "allGenome" not in values or len(values) > 1:
        query += f"<Filter name='{attribute_filter}' value='" + ','.join(values[:max_values]) + "' />"

    for attribute in attributes:
        query += f"<Attribute name='{attribute}' />"

    query += "</Dataset></Query>"
    return query


def _fetch_annotation(query: str, base_url: str, attributes: list[str]) -> pd.DataFrame:
    """
    Fetches the gene annotation based on the constructed query and parses it into a DataFrame.

    Args:
        query: The query XML string.
        base_url: The base URL for the Ensembl biomart service.
        attributes: The list of attributes to include in the DataFrame.

    Returns:
        A DataFrame containing the gene annotations.

    Raises:
        ValueError: If there's an error with the query or network issues occur.
    """
    response = requests.get(f"{base_url}?query={query}", timeout=10)

    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text), sep=",", header=None)
        df.columns = attributes
        if df.empty or "ERROR" in df.iloc[0, 0]:
            err = "Error in query. Please check attributes and filter."
            raise ValueError(err)

        return df

    err = f"Failed to fetch data, HTTP status code: {str(response.status_code)}"
    raise ValueError(err)
