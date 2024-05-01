#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='knowseqpy',
    version='1.0',
    description='All in one tool to analyze RNASeq data',
    author='Blanca Domene',
    packages=find_packages(),
    package_data={
        'knowseqpy': [
            'external_data/*.csv',
            'r_scripts/*.R'
        ]
    },
    install_requires=[
        "numpy",
        "mrmr_selection",
        "pandas",
        "patsy",
        "plotly",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "requests",
        "pyarrow"
    ],
 )