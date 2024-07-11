
# KnowSeqPy: *Python Library for Feature Extraction and Biomarker Detection in Cancer Diagnosis*
In bioinformatics, the ability to efficiently analyze and interpret large genomic data sets is essential for enriching our understanding of genetic diseases. This thesis introduces a Python translation of the KnowSeq R package, initially developed for RNA-seq data analysis. This translation benefits from Python’s widespread adoption in the data science community, its versatile ecosystem, and the increasing demand for bioinformatics tools that are accessible to researchers with varying levels of programming expertise.

This README provides instructions on installation, usage, and running the provided code and Docker container.

## Installation

### Python Dependencies

To install the required Python packages, you can use the `requirements.txt` file. Make sure you have Python 3.12 installed.

```
pip install -r requirements.txt
```

The `requirements.txt` file includes the following packages:

- keras
- numpy
- mrmr_selection
- pandas
- patsy
- plotly
- pyarrow
- requests
- scikit-learn
- scikeras
- scipy
- statsmodels
- tensorflow
- sphinx-rtd-theme

### R Dependencies

To install the R dependencies, ensure you have R installed on your system. You can install the required R packages using the following commands:

```R
install.packages("arrow")
install.packages("tibble")
install.packages("BiocManager")
BiocManager::install("limma")
BiocManager::install("cqn")
BiocManager::install("sva")
```
You may also need to install the following system dependencies for R to work correctly:

```
apt install -y r-base libcurl4-openssl-dev libssl-dev libxml2-dev cmake build-essential g++ python3-dev
```

### Downloading the project
After installing the dependencies, clone the project repository from GitHub:
```
git clone https://github.com/blancadomene/knowseqpy.git
```

## Docker

A Docker image is provided to simplify the setup process. The image includes all necessary dependencies.

### Building the Docker Image

To build the Docker image, use the provided `Dockerfile`.

```
docker build -t knowseqpy:latest .
```

### Using the Docker Image from Docker Hub
If you prefer to use the pre-built image available on Docker Hub, you can pull it directly:

```
docker pull bdomene/knowseqpy:latest
```

### Running the Docker Container

To run the Docker container, use the following command. Ensure you mount the required directories as specified.

```
docker run -it \
  -v $(pwd)/tests/test_fixtures/count_files_breast:/knowseqpy/external_data/breast/count_files_breast \
  -v $(pwd)/tests/test_fixtures/samples_info_breast.csv:/knowseqpy/external_data/breast/samples_info_breast.csv \
  knowseqpy:latest
```

This will start the container with the necessary data mounted for running the examples.

## Usage

To use KnowSeqPy, you can import specific modules or classes directly. Here are some examples:

```python
from knowseqpy import counts_to_dataframe
from knowseqpy.classifiers import decision_tree
```

Follow the documentation and examples provided to perform your bioinformatics analyses.

## Examples

Example scripts are included in the [`examples`](https://github.com/blancadomene/knowseqpy/tree/main/examples) directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
