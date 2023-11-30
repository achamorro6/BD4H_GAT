import os
import torch
import torch.nn as nn
from models import TransductiveGAT, GCN
from datasets import get_dataset
from utils import train, evaluate
from plots import plot_learning_curves
import numpy as np

"""
#### Original paper uses the following parameters:
- batch size = 1
- epochs = 100000
- patience = 100
- learning rate = 0.005 however for pubmed 0.01    
- L2 regularization = 0.0005 however for pubmed 0.001
- Transductive Datasets
    - 8 hidden units for each attention head in each layer
    - 8 attention heads for first layer input
    - ['Cora', 'CiteSeer'] 1 attention head for second layer output
    - 'Pubmed' 8 attention head for second layer output
- No residual
- Each feature is activated using LeakyReLU, with a negative slope of 0.2
- multi-head layer: the features of each attention mechanism are aggregated
- non-output layer concatenated
- output layer not concatenated instead aggregated using an average function
- Exponential Linear Unit Activation Function is applied
- Softmax Activation Function is applied
"""


def train_model(dataset_name, model_type):
    PATH_OUTPUT = '../output/'
    PLOT_PATH_OUTPUT = '../plots/'

    # available ['Cora', 'CiteSeer', 'Pubmed', 'PPI']
    dataset = get_dataset(dataset_name)

    NUM_EPOCHS = 10000
    USE_CUDA = True  # Set 'True' if you want to use GPU

    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if dataset_name in ['Cora', 'CiteSeer']:
        # For 'Cora', 'CiteSeer'
        if model_type == 'GAT':
            model = TransductiveGAT(dataset.num_node_features, dataset.num_classes, 8, 1)
            save_file = 'TransductiveGAT1.pth'
        if model_type == 'GCN64':
            model = GCN(dataset.num_node_features, dataset.num_classes, 64)
            save_file = 'TransductiveGCN1.pth'
        l2_reg = 0.0005
        learning_rate = 0.005

    elif dataset_name == 'Pubmed':
        # For 'Pubmed'
        if model_type == 'GAT':
            model = TransductiveGAT(dataset.num_node_features, dataset.num_classes, 8, 8)
            save_file = 'TransductiveGAT2.pth'

        if model_type == 'GCN64':
            model = GCN(dataset.num_node_features, dataset.num_classes, 64)
            save_file = 'TransductiveGCN2.pth'
        l2_reg = 0.001
        learning_rate = 0.01
    else:
        raise AssertionError("Wrong!")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    model.to(device)
    criterion.to(device)

    best_val_acc = 0.0
    best_val_loss = 100.0
    patience_counter = 0
    patience = 100
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    # print(f'Training: {dataset_name} - {model_type}')
    for epoch in range(NUM_EPOCHS):
        # if epoch % 200 == 0 and epoch != 0:
            # print(f'\tEpoch: {epoch}')

        train_loss, train_accuracy = train(model, device, dataset[0], criterion, optimizer)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, dataset[0], criterion, dataset[0].val_mask)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy >= best_val_acc and valid_loss <= best_val_loss
        if is_best:
            best_val_acc = valid_accuracy
            best_val_loss = valid_loss
            torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)
            patience_counter = 0  # reset counter when there's an improvement
        else:
            patience_counter = patience_counter + 1
        if patience_counter == patience:
            break  # early stop since there has been no improvement

    plot_learning_curves(PLOT_PATH_OUTPUT, dataset_name, train_losses, valid_losses, train_accuracies, valid_accuracies,
                         model_type)
    best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
    test_loss, test_accuracy, test_results = evaluate(best_model, device, dataset[0], criterion, dataset[0].test_mask)

    # print('Accuracy: {:.4f}'.format(test_accuracy))
    return test_accuracy


if __name__ == '__main__':

    dataset_names = ['Cora', 'CiteSeer', 'Pubmed']
    model_types = ['GAT', 'GCN64']
    num_trials = 100 # original paper has 100 runs for transductive datasets

    for name in dataset_names:
        for model in model_types:
            test_accuracies = []
            print(f'Training: {name} - {model}')
            for i in range(num_trials):
                test_accuracy = train_model(name, model)
                test_accuracies.append(test_accuracy)

            avg_accuracy = np.mean(test_accuracies)
            std_accuracy = np.std(test_accuracies)
            print(f'Average Accuracy: {avg_accuracy:.1f} +/- {std_accuracy:.1f}')


