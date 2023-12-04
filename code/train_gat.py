import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import TransductiveGAT, InductiveGAT, GCN
from datasets import get_dataset
from utils import train, evaluate
from plots import plot_learning_curves
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.transforms import NormalizeFeatures

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


def custom_collate(batch):
    return Batch.from_data_list(batch)


def train_model(dataset_name, model_type, num_epochs, batch_size):
    """
    Train a model of `model_type` using `dataset_name` as the dataset.

    :param dataset_name: Name of dataset. Valid options are in {'Cora', 'CiteSeer', 'Pubmed', 'PPI'}.
    :param model_type: Model type. Valid options are in {'GAT', 'GCN64', 'GCN64ELU','ConstGat'}.

    :return None:
    :raises ValueError: If `dataset_name` is not in {'Cora', 'CiteSeer', 'Pubmed', 'PPI'} or if
        `model_type` not in {'GAT', 'GCN64'}.
    """
    if dataset_name not in {'Cora', 'CiteSeer', 'Pubmed', 'PPI'}:
        raise ValueError(f"Invalid dataset name {dataset_name}")

    if model_type not in {'GAT', 'GCN64', 'GCN64ELU', 'ConstGat'}:
        raise ValueError(f"Invalid model type {model_type}")

    PATH_OUTPUT = '../output/'  # Specifies the path where to save model to.
    PLOT_PATH_OUTPUT = '../plots/'  # Specifies the path where to save the plots to.

    # available ['Cora', 'CiteSeer', 'Pubmed', 'PPI']
    dataset = get_dataset(dataset_name)

    NUM_EPOCHS = num_epochs  # Maximum number of iterations.
    BATCH_SIZE = batch_size
    USE_CUDA = True  # Set to True if you want to use GPU.
    NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if dataset_name in ['Cora', 'CiteSeer', 'Pubmed']:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate)
        test_loader = train_loader
        val_loader = train_loader

    elif dataset_name == "PPI":
        train_ppi = PPI(f'../data/inductive/{dataset_name}', split="train", transform=NormalizeFeatures())
        train_loader = torch.utils.data.DataLoader(train_ppi, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=custom_collate)

        test_ppi = PPI(f'../data/inductive/{dataset_name}', split="test", transform=NormalizeFeatures())
        test_loader = torch.utils.data.DataLoader(test_ppi, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate)

        val_ppi = PPI(f'../data/inductive/{dataset_name}', split="val", transform=NormalizeFeatures())
        val_loader = torch.utils.data.DataLoader(val_ppi, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate)

    if dataset_name in ['Cora', 'CiteSeer']:
        # For 'Cora', 'CiteSeer'
        if model_type == 'GAT':
            model = TransductiveGAT(dataset.num_node_features, dataset.num_classes, 8, 1)
            save_file = 'Transductive_Cora_Citeseer_GAT.pth'
        elif model_type == 'GCN64':
            model = GCN(dataset.num_node_features, dataset.num_classes, 64)
            save_file = 'Transductive_Cora_Citeseer_GCN.pth'
        elif model_type == 'GCN64ELU':
            model = GCN(dataset.num_node_features, dataset.num_classes, 64)
            save_file = 'Transductive_Cora_Citeseer_GCNELU.pth'

        l2_reg = 0.0005
        learning_rate = 0.005

    elif dataset_name == 'Pubmed':
        # For 'Pubmed'
        if model_type == 'GAT':
            model = TransductiveGAT(dataset.num_node_features, dataset.num_classes, 8, 8)
            save_file = 'TransductivePubmed_GAT.pth'
        elif model_type == 'GCN64':
            model = GCN(dataset.num_node_features, dataset.num_classes, 64, activation_function=F.relu)
            save_file = 'Transductive_Pubmed_GCN.pth'
        elif model_type == 'GCN64ELU':
            model = GCN(dataset.num_node_features, dataset.num_classes, 64, activation_function=F.elu)
            save_file = 'Transductive_Pubmed_GCNELU.pth'

        l2_reg = 0.001
        learning_rate = 0.01

    elif dataset_name == 'PPI':
        # For 'Pubmed'
        if model_type == 'GAT':
            model = InductiveGAT(dataset.num_node_features, dataset.num_classes, 4, 6)
            save_file = 'InductiveGAT_PPI_GAT.pth'
        elif model_type == 'ConstGat':
            model = InductiveGAT(dataset.num_node_features, dataset.num_classes, 1, 6)
            save_file = 'InductiveGAT_PPI_ConstGAT.pth'

        l2_reg = 0  # not applied
        learning_rate = 0.005
    else:
        raise AssertionError("Wrong!")

    if dataset_name in ['Cora', 'CiteSeer', 'Pubmed']:
        criterion = nn.CrossEntropyLoss()
    elif dataset_name == 'PPI':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise AssertionError("Wrong!")

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    best_val_acc = 0.0
    best_val_loss = 100.0
    best_val_f1 = 0.0

    # This is used for early stopping, where if the metric doesn't improve within `patience` epochs, the model
    # is considered converged.
    patience_counter = 0
    patience = 100

    train_losses, train_accuracies, train_f1s = [], [], []
    valid_losses, valid_accuracies, valid_f1s = [], [], []

    training_time_total = 0
    for epoch in range(NUM_EPOCHS):
        # Train the model on the training set.
        training_time_start = time()
        if dataset_name in ['Cora', 'CiteSeer', 'Pubmed']:
            train_loss, train_accuracy, train_f1 = train(model, device, train_loader, criterion, optimizer, 'train')
        elif dataset_name == "PPI":
            train_loss, train_accuracy, train_f1 = train(model, device, train_loader, criterion, optimizer, 'PPI')

        training_time_end = time()
        training_time_total += training_time_end - training_time_start

        # Validate the model using the validation set.
        if dataset_name in ['Cora', 'CiteSeer', 'Pubmed']:
            valid_loss, valid_accuracy, valid_f1, valid_results = evaluate(model, device, val_loader, criterion, 'val')
        elif dataset_name == "PPI":
            valid_loss, valid_accuracy, valid_f1, valid_results = evaluate(model, device, val_loader, criterion, 'PPI')

        print(f'{train_accuracy}, {train_f1}')
        print(f'{valid_accuracy}, {valid_f1}')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        train_f1s.append(train_f1)
        valid_f1s.append(valid_f1)

        # If the current accuracy is higher or equal to the current best accuracy and the current loss or lower or equal
        # to the current lowest loss, the current model is considered the best.
        if dataset_name in ['Cora', 'CiteSeer', 'Pubmed'] and (
                valid_accuracy >= best_val_acc and valid_loss <= best_val_loss):
            best_val_acc = valid_accuracy
            best_val_loss = valid_loss
            torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)
            patience_counter = 0  # Reset counter when there's an improvement.
            continue

        elif dataset_name == 'PPI' and (valid_f1 >= best_val_f1 and valid_loss <= best_val_loss):
            best_val_f1 = valid_f1
            best_val_loss = valid_loss
            torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)
            patience_counter = 0  # Reset counter when there's an improvement.
            continue

        patience_counter = patience_counter + 1

        # If there has been no improvement within the last `patience` epochs, the model is considered converged
        # and the training will be stopped.
        if patience_counter == patience:
            break

    plot_learning_curves(PLOT_PATH_OUTPUT, dataset_name, train_losses, valid_losses, train_accuracies, train_f1s,
                         valid_f1s, valid_accuracies, model_type)

    # Load the best model that was obtained during training.
    best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
    # Test the model performance on the testing set.
    testing_time_start = time()
    if dataset_name in ['Cora', 'CiteSeer', 'Pubmed']:
        test_loss, test_accuracy, test_f1, test_results = evaluate(best_model, device, test_loader, criterion, 'test')
    elif dataset_name == "PPI":
        test_loss, test_accuracy, test_f1, test_results = evaluate(best_model, device, test_loader, criterion, 'PPI')
    testing_time = time() - testing_time_start

    return test_accuracy, test_f1, training_time_total, testing_time


def run_experiment(dataset_names, model_types, num_trials, num_epochs, batch_size):
    for name in dataset_names:
        for model in model_types:
            print(f'Training: {name} - {model}')
            test_accuracies, test_f1s, training_times, testing_times = [], [], [], []
            for n in range(num_trials):
                print(f'Starting trial {n}')
                test_accuracy, test_f1, training_time, testing_time = train_model(name, model, num_epochs, batch_size)

                test_accuracies.append(test_accuracy)
                test_f1s.append(test_f1)
                training_times.append(training_time)
                testing_times.append(testing_time)

            avg_accuracy = np.mean(test_accuracies)
            std_accuracy = np.std(test_accuracies)

            avg_f1 = np.mean(test_f1s)
            std_f1 = np.std(test_f1s)

            avg_training_time = np.mean(training_times)
            std_training_time = np.std(training_times)

            avg_testing_time = np.mean(testing_times)
            std_testing_time = np.std(testing_times)

            print(f'''\
    {name} - {model}:
        Average Accuracy: {avg_accuracy:.1f} +/- {std_accuracy:.1f}
        Average Micro F1: {avg_f1:.3f} +/- {std_f1:.3f}
        Average training time: {avg_training_time:.3f} +/- {std_training_time:.3f}
        Average testing time: {avg_testing_time:.3f} +/- {std_testing_time:.3f}
    ''')


if __name__ == '__main__':
    num_epochs = 10000

    # transductive_trials = 100  # original paper has 100 runs for transductive dataset
    # transductive_batch_size = 1
    # run_experiment(dataset_names=['Cora', 'CiteSeer', 'Pubmed'], model_types=['GAT', 'GCN64', 'GCN64ELU'],
    #                num_trials=transductive_trials, num_epochs=num_epochs, batch_size=transductive_batch_size)

    inductive_trials = 10  # original paper has 10 runs for inductive dataset
    inductive_batch_size = 2
    run_experiment(dataset_names=['PPI'], model_types=['GAT','ConstGat'], num_trials=inductive_trials,
                   num_epochs=num_epochs, batch_size=inductive_batch_size)
