FROM bdomene/knowseqpy:latest

# Delete the knowseqpy pkg so we can use the project mounted in the devcontainer
RUN pip uninstall -y knowseqpy
RUN rm -r /knowseqpy

# Unset env for external data so we load it from the mounted workspace in the examples
ENV SAMPLES_INFO_BREAST_PATH=
ENV COUNTS_BREAST_PATH=