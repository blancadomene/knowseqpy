import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


# TODO: add timestamp?

def plot_confusion_matrix(conf_matrix, class_labels=None, to_pdf=False, pdf_filename='confusion_matrix.pdf'):
    """
    Plots a confusion matrix using seaborn heatmap and optionally exports it to a PDF file.

    Parameters:
    conf_matrix (array-like): The confusion matrix to be plotted.
    class_labels (list of str): List of class labels. If None, defaults to integer labels.
    to_pdf (bool): Whether to export the plot to a PDF file.
    pdf_filename (str): Filename for the exported PDF file.
    """
    if class_labels is None:
        class_labels = [str(i) for i in range(len(conf_matrix))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    if to_pdf:
        with PdfPages(pdf_filename) as export_pdf:
            export_pdf.savefig()
            plt.close()
