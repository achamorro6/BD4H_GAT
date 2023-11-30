import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class TransductiveGAT(nn.Module):
    def __init__(self, num_features, num_classes, num_input_att_heads, num_output_att_heads):
        super(TransductiveGAT, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.num_input_att_heads = num_input_att_heads  # 1 for constant; 8 for others
        self.num_hidden_units = 8

        self.num_output_att_heads = num_output_att_heads

        self.drop_out = 0.6
        # First Layer:
        self.conv1 = GATConv(in_channels=self.num_features, out_channels=self.num_hidden_units,
                             heads=self.num_input_att_heads, dropout=self.drop_out, concat=True)
        # Second Layer:
        self.conv2 = GATConv(in_channels=self.num_hidden_units * self.num_input_att_heads,
                             out_channels=self.num_classes, heads=self.num_output_att_heads,
                             dropout=self.drop_out, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # apply dropout to the features
        x = torch.dropout(x, self.drop_out, self.training)
        # apply the first GAT layer
        x = self.conv1(x, edge_index)
        # followed by exponential linear unit activation function
        x = F.elu(x)
        # apply dropout
        x = torch.dropout(x, self.drop_out, self.training)
        # apply the second GAT layer
        x = self.conv2(x, edge_index)
        # softmax activation function done in loss calculation
        # x = F.softmax(x, dim=1)
        return x


class GCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_features):
        super(GCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.drop_out = 0.6
        # First Layer:
        self.conv1 = GCNConv(in_channels=self.num_features, out_channels=self.hidden_features)
        # Second Layer:
        self.conv2 = GCNConv(in_channels=self.hidden_features, out_channels=self.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # apply dropout to the features
        x = torch.dropout(x, self.drop_out, self.training)
        # apply the first GAT layer
        x = self.conv1(x, edge_index)
        # followed by exponential linear unit activation function
        x = F.relu(x)
        # apply dropout
        x = torch.dropout(x, self.drop_out, self.training)
        # apply the second GAT layer
        x = self.conv2(x, edge_index)
        # softmax activation function done in loss calculation
        # x = F.softmax(x, dim=1)
        return x
