import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(file_path, dataset_name, train_losses, valid_losses, train_accuracies, valid_accuracies, model_type):
    """
    Plot and save the training and validation learning curves of a model to `file_path`
    The dataset name and model type will be used to construct the title of the plot.

    :param file_path: Path where to save the plots to.
    :param dataset_name: Name of dataset.
    :param train_losses: Training losses.
    :param valid_losses: Validation losses.
    :param train_accuracies: Training accuracies.
    :param valid_accuracies: Validation accuracy:
    :param model_type: Model type.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    ax1.plot(np.arange(len(train_losses)), train_losses, label='Training Loss', color='blue')
    ax1.plot(np.arange(len(valid_losses)), valid_losses, label='Validation Loss', color='orange')
    ax1.set_title(f'{dataset_name} - {model_type}: Loss Curve')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax2.plot(np.arange(len(train_accuracies)), train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_title(f'{dataset_name} - {model_type}: Accuracy Curve')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper left')
    file_name = f'{dataset_name}_{model_type}_plot_learning_curves.png'
    plt.savefig(file_path + file_name, bbox_inches='tight')
    plt.close()
