import io
import os
from typing import List

import pandas as pd
import requests

EXTERNAL_DATA_PATH = os.path.normpath(os.path.join(os.getcwd(), "../external_data"))
ENSEMBL_URL = "http://www.ensembl.org/biomart/martservice"
GRCH37_ENSEMBL_URL = "https://grch37.ensembl.org/biomart/martservice"

def get_genes_annotation(values: List[str],
                         attributes: List[str] = ["ensembl_gene_id","external_gene_name","percentage_gene_gc_content","entrezgene_id"],
                         filter: str = "ensembl_gene_id",
                         notHSapiens: bool = False,
                         notHumandataset: str = "",
                         referenceGenome: int = 38) -> pd.DataFrame:
    """
    Retrieves gene annotations from the Ensembl biomart database.

    :param values: A list of genes that can contain Ensembl IDs, gene names, or the string "allGenome" to indicate all genes.
    :param attributes: A list of desired attributes or information to retrieve from Ensembl biomart. Default is ['ensembl_gene_id', 'external_gene_name', 'percentage_gene_gc_content', 'entrezgene_id'].
    :param filter: An attribute used as a filter to return the rest of the attributes. Default is 'ensembl_gene_id'.
    :param notHSapiens: Indicates whether to retrieve human annotations (False) or annotations from a different species available in BiomaRt (True). Default is False.
    :param notHumandataset: The dataset identification for non-human annotations (if notHSapiens is True). Default is an empty string.
    :param referenceGenome: The reference genome to use. It must be 37 or 38. Default is 38.

    :return: A DataFrame containing the requested gene annotations.

    :raises ValueError: If invalid input is provided for the parameters.
    :raises ValueError: If an error occurs during the query, or the query result is empty or contains an error message.

    Examples:
        >>> myAnnotation = getGenesAnnotation(["KRT19", "BRCA1"], filter="external_gene_name", notHSapiens=False)
    """
    if not isinstance(attributes, list) or not all(isinstance(attr, str) for attr in attributes):
        raise ValueError("The 'attributes' parameter must be a list of attribute names.")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("The 'values' parameter must be a list of gene IDs.")
    if not isinstance(filter, str):
        raise ValueError("The 'filter' parameter must be a string.")
    if not isinstance(notHSapiens, bool):
        raise ValueError("The 'notHSapiens' parameter can only take the values True or False.")
    if referenceGenome not in [37, 38]:
        raise ValueError("The 'referenceGenome' parameter must be 37 or 38.")

    if filter not in attributes:
        attributes += [filter]

    base = ENSEMBL_URL
    if notHSapiens:
        if notHumandataset == "" or not isinstance(notHumandataset, str):
            raise ValueError("The 'notHumandataset' parameter must be a non-empty string.")
        dataset_name = notHumandataset
        filename = f"{notHumandataset}.csv"
    else:
        print("Getting annotation of the Homo Sapiens...\n")
        if referenceGenome == 38:
            print("Using reference genome 38.")
            my_annotation = pd.read_csv(f"{EXTERNAL_DATA_PATH}/GRCh38Annotation.csv")
            if filter in my_annotation.columns and set(attributes).issubset(my_annotation.columns):
                my_annotation = my_annotation[my_annotation[filter].isin(values)]
                my_annotation = my_annotation[attributes]
                return my_annotation
            else:
                dataset_name = 'hsapiens_gene_ensembl'
                filename = f"{dataset_name}.csv"
        else:
            print("Using reference genome 37.")
            base = GRCH37_ENSEMBL_URL
            dataset_name = 'hsapiens_gene_ensembl'
            filename = f"{dataset_name}.csv"

    print(f"Downloading annotation {dataset_name}...")
    act_values = values.copy()
    max_values = min(len(values), 900)
    my_annotation = pd.DataFrame()

    while len(act_values) > 0:
        query = f'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>' \
                f'<Query virtualSchemaName="default" formatter="CSV" header="0" uniqueRows="0" count="" ' \
                f'datasetConfigVersion="0.6">' \
                f'<Dataset name="{dataset_name}" interface="default">'

        if len(values) > 1 or values != "allGenome":
            query += f'<Filter name="{filter}" value="'
            query += ",".join(act_values[:max_values])
            query += '" />'

        for attribute in attributes:
            query += f'<Attribute name="{attribute}" />'
        query += '</Dataset></Query>'

        response = requests.get(f"{base}?query={query}")
        act_my_annotation = pd.read_csv(io.StringIO(response.text), sep=",", header=None)
        act_my_annotation.columns = attributes

        if act_my_annotation.empty or "ERROR" in act_my_annotation.iloc[0, 0]:
            raise ValueError("Error in query. Please check attributes and filter.")

        if my_annotation.empty:
            my_annotation = act_my_annotation
        else:
            my_annotation = pd.concat([my_annotation, act_my_annotation], ignore_index=True)

        if len(act_values) <= max_values:
            act_values = []
        else:
            act_values = act_values[max_values:]
        max_values = min(900, len(act_values))

    if len(values) > 1 or values != "allGenome":
        my_annotation = my_annotation[my_annotation[filter].isin(values)]

    if filename:
        my_annotation.to_csv(f"{EXTERNAL_DATA_PATH}/{filename}", index=False)

    return my_annotation

