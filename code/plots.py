import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(file_path, dataset_name, train_losses, valid_losses, train_accuracies, valid_accuracies):
    # TODO: Make plots for loss curves and accuracy curves.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.plot(np.arange(len(train_losses)), train_losses, label='Training Loss', color='blue')
    ax1.plot(np.arange(len(valid_losses)), valid_losses, label='Validation Loss', color='orange')
    ax1.set_title(dataset_name + ': Loss Curve')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax2.plot(np.arange(len(train_accuracies)), train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_title(dataset_name + ': Accuracy Curve')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper left')
    plt.savefig(file_path + dataset_name + '_plot_learning_curves.png', bbox_inches='tight')
