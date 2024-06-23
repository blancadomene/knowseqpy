"""
Data Visualization Module using Plotly.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.tree import _tree, DecisionTreeClassifier

from .logger import get_logger

# TODO: Can't export to png bc of the use of deprecated setDaemon API in Kaleido (freezes execution).
#       Opened issue: https://github.com/plotly/Kaleido/issues/171

logger = get_logger().getChild(__name__)


def plot_boxplot(data: pd.DataFrame, labels: pd.Series, fs_ranking: list, top_n_features: int = 5,
                 png_filename: str = None) -> None:
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


def plot_confusion_matrix(conf_matrix: np.ndarray, unique_labels: list, png_filename: str = None) -> None:
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
                         png_filename: str = None) -> None:
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


def plot_decision_boundary(model: Pipeline, data: pd.DataFrame, labels: pd.Series, vars_selected: list) -> None:
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
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Convert labels to numeric codes
    label_codes, unique_labels = pd.factorize(labels, sort=True)

    # Create the plot
    fig = go.Figure()

    # Add the decision boundary as a heatmap
    fig.add_trace(go.Heatmap(
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        z=z,
        showscale=False,
        colorscale=[[0, 'red'], [1, 'green']],
        opacity=0.3
    ))

    # Add contour lines for better boundary visualization
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        z=z,
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


"""
def plot_decision_tree_plt(model: Pipeline, labels: pd.Series, vars_selected: list) -> None:
    "
    Plots the decision tree from the trained model pipeline using plotly.

    Args:
        model: A trained model pipeline that includes preprocessing and the classifier.
        labels: True labels for the test data.
        vars_selected: Selected genes for classification.

    Returns:
        None. Displays the plot of the decision tree.
    "
    from matplotlib import pyplot as plt
    y_true_test, unique_labels = pd.factorize(labels, sort=True)

    plt.figure(figsize=(20, 10))
    plot_tree(model.named_steps.decisiontreeclassifier, filled=True, feature_names=vars_selected,
              class_names=unique_labels.astype(str), rounded=True)
    plt.show()
"""


def get_tree_data(tree: DecisionTreeClassifier, feature_names: List[str], class_names: List[str]) -> Tuple[
    List[Tuple], List[Tuple], int]:
    """
    Extracts and structures data from a decision tree for visualization.

    This function traverses the decision tree recursively from the root to the leaves, collecting
    data about each node and the connecting edges. These details include the splitting feature,
    threshold value, Gini impurity, number of samples at the node, and classification outcomes,
    which are used to visualize the tree structure.

    Args:
        tree: The decision tree classifier from which to extract structure data.
        feature_names: Names of the features used in the decision tree.
        class_names: Names of the possible output classes in the decision tree.

    Returns:
        Tuple[List[Tuple], List[Tuple], int]: A tuple containing:
            - A list of node details.
            - A list of edge details, specifying the connection between nodes and the condition.
            - The maximum depth of the tree.
    """

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    node_depth = [0]
    nodes = []
    edges = []

    def recurse(node, depth, x_offset):
        node_depth[0] = max(node_depth[0], depth)
        value = tree_.value[node][0]
        samples = tree_.n_node_samples[node]
        class_idx = value.argmax()
        class_name = class_names[class_idx]
        color = 'red' if class_name == 'Primary Tumor' else 'green'

        if node == 0:
            color = 'lightblue'

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            gini = tree_.impurity[node]

            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]

            node_text = (f"{name} <= {threshold:.2f}<br>gini = {gini:.2f}<br>"
                         f"samples = {samples}<br>values = {value if gini != 0.5 else [70, 70]}")
            if node != 0:  # Exclude class name for root node
                node_text += f"<br>class = {class_name}"

            nodes.append((node, name, threshold, value, samples, depth, "internal", class_name, color, x_offset, gini,
                          node_text))
            edges.append((node, left_child, "True"))
            edges.append((node, right_child, "False"))

            recurse(left_child, depth + 1, x_offset - 1 / 2 ** (depth + 1))
            recurse(right_child, depth + 1, x_offset + 1 / 2 ** (depth + 1))
        else:
            class_name = "Primary Tumor" if color == 'red' else "Solid Tissue Normal"
            node_text = f"samples = {samples}<br>class = {class_name}"
            nodes.append((node, "", "", value, samples, depth, "leaf", class_name, color, x_offset, None, node_text))

    recurse(0, 0, 0)
    return nodes, edges, node_depth[0]


def plot_decision_tree(model: Pipeline, labels: pd.Series, vars_selected: List[str]) -> None:
    """
    Generates and displays an interactive plot of a decision tree using Plotly.

    Args:
        model: A sklearn Pipeline, which includes the decision tree classifier and preprocessing steps.
        labels: The series of true labels used to train the classifier, required for naming the classes in the plot.
        vars_selected: List of feature names used in the decision tree, which are selected for classification.
    """
    y_true_test, class_names = pd.factorize(labels, sort=True)
    tree = model.named_steps.decisiontreeclassifier
    nodes, edges, max_depth = get_tree_data(tree, vars_selected, class_names)

    x = []
    y = []
    text = []
    for node, name, threshold, value, samples, depth, node_type, class_name, color, x_offset, gini, node_text in nodes:
        x.append(x_offset)
        y.append(depth)
        text.append(node_text)

    fig = go.Figure()

    # Add edges first to ensure they appear at the back
    for parent, child, condition in edges:
        parent_idx = next(i for i, v in enumerate(nodes) if v[0] == parent)
        child_idx = next(i for i, v in enumerate(nodes) if v[0] == child)
        fig.add_trace(go.Scatter(
            x=[x[parent_idx], x[child_idx]],
            y=[y[parent_idx], y[child_idx]],
            mode='lines',
            line=dict(color='lightgrey', width=2),
            opacity=0.7
        ))

    # Add nodes to ensure they appear on top of the lines
    for node, name, threshold, value, samples, depth, node_type, class_name, color, x_offset, gini, node_text in nodes:
        fig.add_trace(go.Scatter(
            x=[x_offset],
            y=[depth],
            text=[node_text],
            mode='markers+text',
            textposition="bottom center",
            marker=dict(size=40, color=color, line=dict(color='black', width=2))
        ))

    # Adding text for conditions ensuring visibility
    for parent, child, condition in edges:
        parent_idx = next(i for i, v in enumerate(nodes) if v[0] == parent)
        child_idx = next(i for i, v in enumerate(nodes) if v[0] == child)
        fig.add_trace(go.Scatter(
            x=[(x[parent_idx] + x[child_idx]) / 2],  # Position text in the middle of the line
            y=[(y[parent_idx] + y[child_idx]) / 2],
            text=[condition],
            mode='text',
            textfont=dict(size=12, color='blue'),
            textposition="top center"
        ))

    fig.update_layout(
        title="Decision Tree",
        xaxis=dict(title="Node", showgrid=False, zeroline=False),
        yaxis=dict(title="Depth", showgrid=False, zeroline=False, autorange="reversed"),
        showlegend=False,
        height=800
    )

    fig.show()
