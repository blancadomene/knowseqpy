import os

import pandas as pd


def get_genes_annotation(values: pd.Series, filter: str = "ensembl_gene_id", diff_dataset_id: str = None,
                         ref_genome: int = 38):
    """
    ???.
    :param filter: The attribute used as filter to return the rest of the attributes. Set to ensembl_gene_id by default.
    :param diff_dataset_id: Dataset identification from biomaRt::listDatasets(useMart("ensembl")). Set to None by default.
    :param ref_genome: Integer that indicates used reference genome. It must be 37 or 38. Set to 38 by default.

    :return: ????.
    """
    if ref_genome not in (37, 38):
        raise Exception("Introduced reference_genome is not available. Value must be 37 or 38.")
    """
    dir = os. path()
    base = 'http://www.ensembl.org/biomart/martservice'
    server = "https://rest.ensembl.org"

    if not diff_dataset_id:
        print("Getting annotation of the Homo Sapiens...\n")
        if ref_genome == 38:
            print("Using reference genome 38.\n")
            ext = "/map/human/GRCh37/X:1000000..1000100:1/GRCh38?"
        annotation = pd.read_csv(dir + "/GRCh38Annotation.csv")
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'external_data', "GRCh38Annotation.csv")  # TODO: Delete and read from API
    df = pd.read_csv(path)
    df = df[df[filter].isin(values)]
    return df.loc[df['ensembl_gene_id'].isin(values)]
