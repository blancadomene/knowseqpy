import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp, zscore
from statsmodels.robust.scale import mad


def RNAseqQA(expressionMatrix, outdir="SamplesQualityAnalysis", toPNG=True, toPDF=True, toRemoval=False):
    if not isinstance(expressionMatrix, pd.DataFrame):
        raise ValueError("The class of expressionMatrix parameter must be a DataFrame.")
    if not isinstance(toPNG, bool):
        raise ValueError("toPNG parameter can only take the values True or False.")
    if not isinstance(toPDF, bool):
        raise ValueError("toPDF parameter can only take the values True or False.")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Performing samples quality analysis...")

    expressionMatrix = expressionMatrix.loc[~expressionMatrix.index.duplicated(keep='first')]
    outliers = {}
    found_outliers = [-1]
    removed_outliers = []

    print("Running Distances Outliers Detection test...")
    distance_matrix = pd.DataFrame(squareform(pdist(expressionMatrix.T, 'cityblock')) / len(expressionMatrix),
                                   columns=expressionMatrix.columns, index=expressionMatrix.columns)
    distance_sum = distance_matrix.sum()
    dist_data = pd.DataFrame({'x': distance_sum, 'y': distance_matrix.columns})

    if toPNG:
        sns.heatmap(distance_matrix)
        plt.title('Distances between arrays')
        plt.savefig(os.path.join(outdir, 'distance-plot.png'), dpi=300)
        plt.close()

    q3 = dist_data['x'].quantile(0.75)
    dist_limit = q3 + 1.5 * (dist_data['x'].quantile(0.75) - dist_data['x'].quantile(0.25))

    outliers['Distance'] = {'limit': dist_limit, 'outliers': distance_sum[distance_sum > dist_limit]}

    print("Done!")

    print("Running Kolmogorov-Smirnov test...")
    ks = expressionMatrix.apply(lambda x: ks_2samp(x, expressionMatrix.values.flatten())[0], axis=1)

    ks_data = pd.DataFrame({'x': ks, 'y': np.arange(len(ks))})

    q3 = ks_data['x'].quantile(0.75)
    ks_limit = q3 + 1.5 * (ks_data['x'].quantile(0.75) - ks_data['x'].quantile(0.25))

    outliers['KS'] = {'limit': ks_limit, 'outliers': ks[ks > ks_limit]}

    print("Done!")

    print("Running MAD Outliers Detection test...")
    rowExpression = expressionMatrix.mean(axis=1)

    outliersMA = []
    for i in rowExpression.index:
        exprMatrix = rowExpression.drop(i)
        upperBound = exprMatrix.median() + 3 * mad(exprMatrix)
        lowerBound = exprMatrix.median() - 3 * mad(exprMatrix)

        if rowExpression.loc[i] < lowerBound or rowExpression.loc[i] > upperBound:
            outliersMA.append(i)

        outliers['MAD'] = {'limit': "-", 'outliers': outliersMA}

        print("Done!")

        if not toRemoval:
            return outliers

        found_outliers = list(
            set(outliers['Distance']['outliers']).intersection(outliers['KS']['outliers'], outliers['MAD']['outliers']))
        expressionMatrix = expressionMatrix.drop(found_outliers, errors='ignore')
        removed_outliers += found_outliers

        return {'matrix': expressionMatrix, 'outliers': removed_outliers}

