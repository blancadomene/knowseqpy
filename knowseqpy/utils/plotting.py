"""
Data Visualization Module using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

from .logger import get_logger

# TODO: Can't export to png bc of the use of deprecated setDaemon API in Kaleido (freezes execution).
#       Opened issue: https://github.com/plotly/Kaleido/issues/171

logger = get_logger().getChild(__name__)


def plot_boxplot(data: pd.DataFrame, labels: pd.Series, fs_ranking: list, top_n_features: int = 5,
                 png_filename: str = None):
    """
    Plots a boxplot of the top N features from the feature selection ranking.

    Args:
        data: Data containing samples in rows and features in columns.
        labels: Labels for the samples.
        fs_ranking: List of feature names ordered by ranking.
        top_n_features: Number of top features to include in the boxplot. Defaults to 5.
        png_filename: Filename for the exported PNG file.
                      If not None, the plot is exported with the given name. Defaults to None.
    """

    top_fs_ranking = fs_ranking[:top_n_features]
    top_data = data[top_fs_ranking]
    labeled_top_data = top_data.merge(labels.rename("class").set_axis(top_data.index).to_frame(),
                                      left_index=True, right_index=True)
    melted_data = labeled_top_data.melt(id_vars="class", value_vars=top_fs_ranking, var_name="Gene",
                                        value_name="Expression")
    fig = px.box(melted_data,
                 x="Gene",
                 y="Expression",
                 color="class",
                 color_discrete_map={"Primary Tumor": "red", "Solid Tissue Normal": "green"}
                 )
    fig.update_traces(quartilemethod="linear")
    fig.update_layout(title_text="Genes Boxplot", xaxis_title="Samples", yaxis_title="Expression value")

    if png_filename:
        fig.write_image(png_filename)

    fig.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, unique_labels: list, png_filename: str = None):
    """
    Plots a confusion matrix of classified samples.

    Args:
        conf_matrix (array-like): The confusion matrix to be plotted.
        unique_labels: Labels for the samples.
        png_filename: Filename for the exported PNG file.
                      If not None, the plot is exported with the given name. Defaults to None.
    """
    fig = ff.create_annotated_heatmap(z=conf_matrix, x=unique_labels, y=unique_labels, colorscale="blues")
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Labels",
        yaxis_title="True Labels",
        xaxis={"side": "bottom"}
    )
    if png_filename:
        fig.write_image(png_filename)

    fig.show()


def plot_samples_heatmap(data: pd.DataFrame, labels: pd.Series, fs_ranking: list, top_n_features: int = 5,
                         png_filename: str = None):
    """
    Plots a heatmap of the top N features from the feature selection ranking.

    Args:
        data: Data containing samples in rows and features in columns.
        labels: Labels for the samples.
        fs_ranking: List of feature names ordered by ranking.
        top_n_features: Number of top features to include in the heatmap. Defaults to 5.
        png_filename: Filename for the exported PNG file.
                      If not None, the plot is exported with the given name. Defaults to None.
    """
    top_fs_ranking = fs_ranking[:top_n_features]
    top_data = data[top_fs_ranking]
    labeled_top_data = top_data.merge(labels.rename("class").set_axis(top_data.index).to_frame(),
                                      left_index=True, right_index=True)
    labeled_top_data = labeled_top_data.sort_values("class")

    fig = go.Figure(data=go.Heatmap(
        z=labeled_top_data.drop(columns=["class"]).values,
        x=top_fs_ranking,
        y=labeled_top_data.index,
        colorscale=[(0, "red"), (0.5, "black"), (1, "green")],
        showscale=True,
        colorbar={"title": "Expression"}
    ))

    fig.update_layout(
        title="Samples Heatmap",
        xaxis_title="Features",
        yaxis_title="Samples",
        yaxis={"tickmode": "array",
               "ticktext": ["Primary Tumor", "Solid Tissue Normal"],
               "tickvals": [labeled_top_data.shape[0] * 0.25, labeled_top_data.shape[0] * 0.75]},
        autosize=True
    )

    if png_filename:
        fig.write_image(png_filename)

    fig.show()


def plot_decision_boundary(model: Pipeline, data: pd.DataFrame, labels: pd.Series, vars_selected: list):
    """
    Plots the decision boundary of a logistic regression model.

    Args:
        model: A trained logistic regression model pipeline.
        data: The feature matrix.
        labels: True labels for the data.
        vars_selected: Selected features for classification.
    """
    data = data[vars_selected]

    # Create a mesh grid for plotting
    x_min, x_max = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1
    y_min, y_max = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict the classification for each point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Convert labels to numeric codes
    label_codes, unique_labels = pd.factorize(labels, sort=True)

    # Create the plot
    fig = go.Figure()

    # Add the decision boundary as a heatmap
    fig.add_trace(go.Heatmap(
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        z=Z,
        showscale=False,
        colorscale=[[0, 'red'], [1, 'green']],
        opacity=0.3
    ))


    # Add contour lines for better boundary visualization
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        z=Z,
        showscale=False,
        contours_coloring='lines',
        line_width=1
    ))

    # Add scatter plot for the data points
    for label, color in zip(unique_labels, ['red', 'green']):
        fig.add_trace(go.Scatter(
            x=data.iloc[:, 0][labels == label],
            y=data.iloc[:, 1][labels == label],
            mode='markers',
            name=f'Class {label}',
            marker=dict(size=10, color=color)
        ))

    # Add layout details
    fig.update_layout(
        title='Decision Boundary of Logistic Regression',
        xaxis_title=vars_selected[0],
        yaxis_title=vars_selected[1]
    )

    fig.show()