# BD4H_GAT
Created by achamorro6 (achamorro6@gatech.edu) &amp; gbergevoet3(stanbergevoet@gatech.edu)
Experimental Setup

Transductive Learning
Based on the original paper, the researchers implemented a Two layer GAT model for the datasets Cora, Citeseer and Pubmed.

Cora and Citeseer Model Hyperparameters
First Layer:
Eight attention heads were used.
Each head computed eight features (64 hidden features total).

Second layer (classification layer)
One output attention head was implemented, computing the number of classes from the dataset.
Dropout:
A dropout rate of 0.6 was implemented.
Dropout was applied before each GAT layer’s input and to both GAT layers.

Regularization:
During training, to prevent overfitting, researchers applied L2 regularization  with a coefficient of 0.0005
Activation Functions:
An exponential linear unit (ELU) activation function was applied.
A softmax activation function was applied for classification.

Batch Size:
Researchers used a batch size of 1 graph for training
Pubmed Model Hyperparameters
Similar implementation to what has been describe with the following changes:
Second layer (classification layer)
Eight output attention heads were implemented
Regularization:
L2 regularization was set to 0.001 



Transductive Learning: High-Level Feed-Forward Implementation (GAT)

Dropout
Gat Layer 1
ELU activation function
Dropout
Gat Layer 2 (classification layer)
SoftMax activation function

Transductive Learning: GCN64 Model

Researchers compared the inductive GAT model to a common baseline Graph Convolution Network (GNC) model with 64 hidden parameters and two GNC layers

GCN64 Model Hyperparameters 

First Layer:
64 hidden units in the graph convolutional layer.
Second Layer (Classification Layer):
64 hidden units in the graph convolutional layer.
Dropout:
A dropout rate of 0.6 was implemented.
Dropout was applied before each graph convolutional layer's input and to both layers.
Regularization:
L2 regularization was set to  0.0005
Activation Functions:
A rectified linear unit (ReLU) activation function was applied for classification
A softmax activation function was applied for classification. 

Transductive Learning: High-Level Feed-Forward Implementation (GCN-64)
Dropout
GCNLayer 1
ReLU activation function
Dropout
GCNLayer 2 (classification layer)
SoftMax activation function




Inductive Learning
Based on the original paper, the researchers implemented a three-layer GAT model for the PPI dataset

PPI Model Hyperparameters

First and Second layer:
Four attention heads were used.
Each head computed 256 features (1024  hidden features total).

Third layer (classification layer)
Six output attention head were implemented, each computing 121 features

For PPI researchers did not apply L2 regularization or dropouts because the training sets were considered large enough.

Activation Functions:
An exponential linear unit (ELU) activation function was applied.
A logistic sigmoid activation function was applied for classification.
Batch Size:
Researchers used a batch size of 2 graphs for training

Inductive Learning: High-Level Feed-Forward Implementation

Gat Layer 1
ELU activation function
Gat Layer 2
ELU activation function
Gat Layer 3 (classification layer)
Logistic sigmoid activation function


Inductive Learning: Constant Attention Mechanism Model
Researchers established a baseline to evaluate the impact of applying an attention mechanism compared to a closely related Graph Convolution Network (GNC) model.  The researchers achieve this by creating a static attention mechanism, applying a fixed attention coefficient value of 1 to all neighbors. Which has the effect of making all neighbors contain the same weight, and nothing is learned from the attention. The inductive model previously discussed for PPI stayed the same in architecture and parameters, with the exception of making attention coefficient constant. 

Training and Optimization Configuration

Initialization: 
All three models are initialized using Glorot initialization.
Training Objective:
 Researchers minimized for cross-entropy on training nodes
Optimizer:
 Adam SGD Optimizer is used for training
Initial Learning rate:
For Pubmed the learning rate was set to 0.01
For the rest of the datasets the learning rate was set to 0.005
Early Stopping: 
Researchers applied early stopping to the validation nodes, with a patience of 100 epochs in addition to the following:
Transductive: cross-entropy loss with softmax activation and accuracy
Inductive: cross-entropy loss with softmax sigmoid activation and micro-F1 score
