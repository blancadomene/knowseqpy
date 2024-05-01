FROM ubuntu:24.04

WORKDIR /knowseqpy

# Setup and update APT
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
RUN apt update

# Install R and compile deps
RUN apt install -y r-base
RUN apt install -y libcurl4-openssl-dev
RUN apt install -y libssl-dev
RUN apt install -y libxml2-dev
RUN apt install -y cmake
RUN apt install -y build-essential
RUN apt install -y g++
RUN apt install -y python3-dev

# Install R and R libs
RUN R -e 'install.packages("arrow")'
RUN R -e 'install.packages("tibble")'
RUN R -e 'install.packages("BiocManager")'
RUN R -e 'BiocManager::install("limma")'
RUN R -e 'BiocManager::install("cqn")'
RUN R -e 'BiocManager::install("sva")'

# Install Python and configure venv
# See: https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
RUN apt install -y python3.12
RUN apt install -y python3.12-venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy lib files and set env to use lib
COPY knowseqpy ./knowseqpy
COPY setup.py ./setup.py
RUN pip install .

# Copy examples and configure external data
# To run the examples, you should mount the following paths into the container:
#    -v $(pwd)/tests/test_fixtures/count_files_breast:/knowseqpy/external_data/breast/count_files_breast
#    -v $(pwd)/tests/test_fixtures/samples_info_breast.csv:/knowseqpy/external_data/breast/samples_info_breast.csv
COPY examples ./examples
VOLUME /knowseqpy/external_data/breast
ENV SAMPLES_INFO_BREAST_PATH=/knowseqpy/external_data/breast/samples_info_breast.csv
ENV COUNTS_BREAST_PATH=/knowseqpy/external_data/breast/count_files_breast

WORKDIR /knowseqpy/examples
ENTRYPOINT ["/bin/bash"]