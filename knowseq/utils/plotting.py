"""
Data Visualization Module using Plotly.
"""
import plotly.figure_factory as ff
import plotly.graph_objects as go


# TODO: Remove to_png? And maybe only check png_filename not empty
#  Can"t export right now anyways bc of the use of deprecated setDaemon API in Kaleido (freezes execution)

def plot_boxplot():
    pass


def plot_confusion_matrix(conf_matrix, sample_labels, to_png: bool = False, png_filename: str = "confusion_matrix.png"):
    """
    Plots a confusion matrix using Plotly's heatmap and optionally exports it to a PNG file.

    Args:
        conf_matrix (array-like): The confusion matrix to be plotted.
        sample_labels (list of str): List of class labels.
        to_png: Whether to export the plot to a PNG file.
        png_filename: Filename for the exported PNG file.
    """
    fig = ff.create_annotated_heatmap(conf_matrix, x=sample_labels, y=sample_labels, colorscale="blues")
    fig.update_layout(title="Confusion Matrix", xaxis={"title": "Predicted Labels"}, yaxis={"title": "True Labels"})

    if to_png:
        fig.write_image(png_filename)

    fig.show()


def plot_samples_heatmap(data, sample_labels, fs_ranking, top_n_features: int, to_png: bool = False,
                         png_filename: str = "samples_heatmap.png"):
    """
    Plots a heatmap of the top N features from the feature selection ranking using Plotly's graph objects.

    Args:
        data (DataFrame): Data containing samples in rows and features in columns.
        sample_labels (Series or list): Labels for the samples.
        fs_ranking (list): List of feature names ordered by ranking.
        top_n_features: Number of top features to include in the heatmap.
        to_png: Whether to export the plot to a PNG file.
        png_filename: Filename for the exported PNG file.
    """
    top_features = fs_ranking[:top_n_features]
    selected_data = data[top_features]
    result = selected_data.merge(sample_labels.rename("class").set_axis(selected_data.index).to_frame(),
                                 left_index=True, right_index=True)
    result = result.sort_values("class")

    fig = go.Figure(data=go.Heatmap(
        z=result.drop(columns=["class"]).values,
        x=top_features,
        y=result.index,
        colorscale=[(0, "red"), (0.5, "black"), (1, "green")],
        showscale=True
    ))

    fig.update_layout(
        title="Samples Heatmap",
        xaxis_title="Features",
        yaxis_title="Samples",
        yaxis={"tickmode": "array",
               "ticktext": ["Primary Tumor", "Solid Tissue Normal"],
               "tickvals": [result.shape[0] * 0.25, result.shape[0] * 0.75]},
        autosize=True
    )

    if to_png:
        fig.write_image(png_filename)

    fig.show()
