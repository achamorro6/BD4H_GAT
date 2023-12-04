## Overview

We will be exploring Graph Attention Networks (GAT), which is a novel graph deep learning model. Next to this, we will attempt to reproduce the results from the paper that introduced GATs.
The GAT model uses a mechanism to calculate the effects that the features of a node’s neighbor have on the node’s own features. Since vertices that are alike are more likely to reference each other. As an example, if you have multiple scientific articles in various disciplines, a paper on chemistry is more likely to be cited by another paper in chemistry, as opposed to a paper on linguistics. It is by this mechanism that it aims to match or exceed the state-of-the-art models at the time when the GAT model was proposed.

We will be recreating the GAT models that were used for the inductive dataset and transductive datasets. The claim that the paper makes is that the GAT model either outperforms or matches the state-of-the-art model of that time. For the transductive model, we will create two GCN models. The difference between the two models is that one uses ReLU activation and the other uses ELU activation. The reason for this being that the paper claims that ReLU outperformed ELU for all transductive datasets. We plan to verify the accuracy scores obtained by all three models on each transductive dataset by comparing our results with the original results. Therefore, we will also check whether ReLU outperforms ELU.

For the inductive dataset, we will construct the GAT model and check whether our results either match or exceed the paper’s results. In this case, since the inductive dataset is multi-label, we will be using micro-F1 score instead of accuracy.

## Repository Contents

- **code:** Contains the main code.
- **data:** Data files required for the experiments.
- **output:** Output .pth files for saved models generated during training.
- **plots:** Contans generated plots .png generated during training.
- **GAT.yml:** required packages and python version.
- **README.md:** ReadMe overview on project.

## Dependencies

To recreate our setup and run our code use the provided YAML file:

```bash
- conda env create -f GAT.yml
- conda activate environment_name
```

## Data
We have predownloaded four different datasets. We use Cora, Citeseer and Pubmed for transductive learning and use a protein-protein interaction (PPI) dataset for inductive learning. 
To obtain the datasets, we used the datasets that are included in PyTorch Geometric. Geometric in turn downloads the Cora, Citaseer and Pubmed datasets from the Planetoid GitHub repo (https://github.com/kimiyoung/planetoid/tree/master/data). For the PPI dataset, it obtains the data from DGL (https://data.dgl.ai/dataset/ppi.zip).

## How to run code:
To execute our experiment, run the train_gat.py script:

```bash
- python train_gat.py
```
Models are saved in the output directory during training. Generated chars are stored in stored in the plots directory fir review during training. Statistics are printed(Accuracy, Micro F1, training time, testing time, standard devations). 


## Experimental Setup

### Transductive Learning

Based on the original paper, the researchers implemented a Two layer GAT model for the datasets Cora, Citeseer and Pubmed.

#### Cora and Citeseer Model Hyperparameters

- **First Layer:**
  - Eight attention heads were used.
  - Each head computed eight features (64 hidden features total).

- **Second Layer (Classification Layer):**
  - One output attention head was implemented, computing the number of classes from the dataset.

- **Dropout:**
  - A dropout rate of 0.6 was implemented.
  - Dropout was applied before each GAT layer’s input and to both GAT layers.

- **Regularization:**
  - During training, to prevent overfitting, researchers applied L2 regularization  with a coefficient of 0.0005

- **Activation Functions:**
  - An exponential linear unit (ELU) activation function was applied.
  - A softmax activation function was applied for classification.

- **Batch Size:**
  - Researchers used a batch size of 1 graph for training

#### Pubmed Model Hyperparameters

Similar implementation to what has been describe with the following changes:

- **Second Layer (Classification Layer):**
  - Eight output attention heads were implemented.

- **Regularization:**
  - L2 regularization was set to 0.001

#### Transductive Learning: High-Level Feed-Forward Implementation (GAT)

- Dropout
- GAT Layer 1
- ELU activation function
- Dropout
- GAT Layer 2 (classification layer)
- Softmax activation function

#### Transductive Learning: GCN64 Model

Researchers compared the inductive GAT model to a common baseline Graph Convolution Network (GNC) model with 64 hidden parameters and two GNC layers.
We compared both rectified linear unit (ReLU) and exponential linear unit (ELU).

##### GCN64 Model Hyperparameters

- **First Layer:**
  - 64 hidden units in the graph convolutional layer.
  
- **Second Layer (Classification Layer):**
  - 64 hidden units in the graph convolutional layer.

- **Dropout:**
  - A dropout rate of 0.6 was implemented.
  - Dropout was applied before each graph convolutional layer's input and to both layers.

- **Regularization:**
  - L2 regularization was set to 0.0005.

- **Activation Functions:**
  - A rectified linear unit (ReLU) or exponential linear unit (ELU) activation function was applied
  - A softmax activation function was applied for classification. 

#### Transductive Learning: High-Level Feed-Forward Implementation (GCN-64)

- Dropout
- GCN Layer 1
- ReLU/ELU activation function
- Dropout
- GCN Layer 2 (classification layer)
- Softmax activation function

### Inductive Learning

Based on the original paper, the researchers implemented a three-layer GAT model for the PPI dataset.

#### PPI Model Hyperparameters

- **First and Second Layer:**
  - Four attention heads were used.
  - Each head computed 256 features (1024  hidden features total).

- **Third Layer (Classification Layer):**
  - Six output attention head were implemented, each computing 121 features.

- For PPI researchers did not apply L2 regularization or dropouts because the training sets were considered large enough.

- **Activation Functions:**
  - An exponential linear unit (ELU) activation function was applied.
  - A logistic sigmoid activation function was applied for classification.

- **Batch Size:**
  - Researchers used a batch size of 2 graphs for training

#### Inductive Learning: High-Level Feed-Forward Implementation

- GAT Layer 1
- ELU activation function
- GAT Layer 2
- ELU activation function
- GAT Layer 3 (classification layer)
- Logistic sigmoid activation function

### Training and Optimization Configuration

- **Initialization:**
  - All models are initialized using Glorot initialization.
  
- **Training Objective:**
  - Researchers minimized for cross-entropy on training nodes
  
- **Optimizer:**
  - Adam SGD Optimizer is used for training
  
- **Initial Learning Rate:**
  - For Pubmed the learning rate was set to 0.01
  - For the rest of the datasets the learning rate was set to 0.005

- **Early Stopping:**
  - Researchers applied early stopping to the validation nodes, with a patience of 100 epochs in addition to the following:
    - Transductive: Cross-entropy loss with softmax activation and accuracy
    - Inductive: Cross-entropy loss with logistic sigmoid activation and micro-F1 score
